"""Lead Agent —— 基于 SimpleReActEngine 的 ReAct 循环架构。

LLM 自主决策工具调用 → 执行工具 → 观察结果 → 继续决策
"""
import json
import logging
from typing import AsyncGenerator

from agent.context import AgentContext, SkillContext
from agent.react_engine import SimpleReActEngine
from prompts.system import BASE_INSTRUCTIONS, SKILL_SYSTEM_TEMPLATE
from skills.skill_registry import SkillRegistry
from skills.skill_loader import SkillLoader
from tools.tool_registry import ToolContext, ToolRegistry
from tools import register_all_tools
from llm.service import LLMService
from services.chat_history import ChatHistoryService
from services.session_service import SessionService
from utils.trace_logger import TraceLogger

logger = logging.getLogger(__name__)

_MAX_HISTORY_ROUNDS = 10


def _match_skill_selection(message: str, candidates: list) -> dict:
    """从用户消息中匹配选择的 skill candidate。"""
    if not candidates:
        return None

    msg = message.strip()

    import re
    letter_match = re.match(r'^([A-Ea-e])\s*[.。、]?$', msg)
    if letter_match:
        label = letter_match.group(1).upper()
        for c in candidates:
            if c.get("label", "") == label:
                return c

    msg_lower = msg.lower()
    for c in candidates:
        name = c.get("display_name", "")
        if name and name in msg:
            return c

    for c in candidates:
        if c.get("skill_id", "") in msg:
            return c

    zh_bigrams = set()
    for i in range(len(msg) - 1):
        if '一' <= msg[i] <= '鿿' and '一' <= msg[i + 1] <= '鿿':
            zh_bigrams.add(msg[i:i + 2])

    best_score = 0
    best_cand = None
    for c in candidates:
        name = c.get("display_name", "")
        score = sum(1 for i in range(len(name) - 1)
                    if '一' <= name[i] <= '鿿' and '一' <= name[i + 1] <= '鿿'
                    and name[i:i + 2] in zh_bigrams)
        if score > best_score:
            best_score = score
            best_cand = c

    if best_score >= 2:
        return best_cand

    return None


def _is_fallback_selection(message: str, candidates: list) -> bool:
    fallback_kws = ["不需要", "其他", "直接搜索", "重新搜索", "没有合适", "都不对", "都不符合"]
    return any(kw in message for kw in fallback_kws)


class LeadAgent:

    def __init__(
        self,
        llm_service: LLMService,
        skill_registry: SkillRegistry,
        skill_loader: SkillLoader,
        chat_history: ChatHistoryService,
        session_service: SessionService,
        memory_store=None,
        memory_updater=None,
    ):
        self._llm = llm_service
        self._reg = skill_registry
        self._loader = skill_loader
        self._ch = chat_history
        self._ss = session_service
        self._memory = memory_store
        self._memory_updater = memory_updater

        self._tool_registry = ToolRegistry()
        register_all_tools(self._tool_registry)
        self._engine = SimpleReActEngine(llm_service, self._tool_registry)
        self._skill_system_prompt = self._build_skill_system_prompt()

    def _build_skill_system_prompt(self) -> str:
        all_skills = [s for s in self._reg.get_enabled() if s.source == "builtin"]
        entries = []
        for s in all_skills:
            entries.append(
                f'  <skill>\n'
                f'    <name>{s.name}</name>\n'
                f'    <description>{s.description}</description>\n'
                f'    <path>{s.skill_md_path}</path>\n'
                f'  </skill>'
            )
        return SKILL_SYSTEM_TEMPLATE.format(skill_entries="\n".join(entries))

    def _build_system_prompt(self, memory_block: str = "") -> str:
        parts = [BASE_INSTRUCTIONS]
        if memory_block:
            parts.append(f"\n<memory>\n{memory_block}\n</memory>")
        parts.append(f"\n{self._skill_system_prompt}")
        return "\n".join(parts)

    def _make_trace_callback(self, session_id: str, trace_id: str):
        ch = self._ch
        async def callback(llm_type, step_name, request_messages, response_content,
                           reasoning_content, model, temperature, elapsed_ms, success, error):
            await ch.save_llm_trace(
                session_id=session_id, trace_id=trace_id,
                llm_type=llm_type, step_name=step_name,
                request_messages=request_messages, response_content=response_content,
                reasoning_content=reasoning_content, model=model, temperature=temperature,
                elapsed_ms=elapsed_ms, success=success, error=error,
            )
        return callback

    async def _load_chat_history(self, session_id: str) -> list[dict]:
        """加载并过滤历史消息：压缩 thinking/skill_result，替换 outline 消息。"""
        msgs = await self._ch.get_messages(session_id, limit=_MAX_HISTORY_ROUNDS * 2)
        filtered = []
        for m in msgs:
            meta = m.get("metadata") or {}
            if meta.get("is_thinking"):
                continue
            content = m["content"]
            if m["role"] == "assistant" and m.get("msg_type") == "skill_result":
                content = f"[摘要]{meta.get('summary', '')}" if meta.get("summary") else "[已执行]"
            if m.get("msg_type") == "outline":
                content = "[已生成大纲]"
            filtered.append({"role": m["role"], "content": content})
        return filtered

    async def handle_message(self, session_id: str, user_message: str) -> AsyncGenerator[str, None]:
        await self._ch.ensure_session(session_id)
        trace = TraceLogger(session_id=session_id)
        trace.log("request.start", data={"user_message": user_message})
        trace_cb = self._make_trace_callback(session_id, trace.trace_id)

        # 加载上下文（历史消息、大纲状态、待确认），在用户消息入库前执行
        chat_history = await self._load_chat_history(session_id)

        outline_state = await self._ch.get_outline_state(session_id)
        has_outline = bool(outline_state and outline_state.get("outline_json"))
        current_outline = outline_state["outline_json"] if has_outline else None

        pending_confirm = await self._ss.get_pending_confirm(session_id)

        # 用户消息入库
        await self._ch.add_message(session_id, "user", user_message)

        msgs = await self._ch.get_messages(session_id, limit=2)
        if len(msgs) <= 1:
            title = user_message[:20] + ("..." if len(user_message) > 20 else "")
            await self._ch.update_session_title(session_id, title)

        # 检查 L5 待确认
        if pending_confirm and not self._is_new_task(user_message):
            ctx = AgentContext(session_id=session_id, user_message=user_message,
                               pending_confirm_options=pending_confirm, has_pending_confirm=True)
            async for ev in self._confirm(ctx, user_message, trace_cb):
                yield ev
            remaining = await self._ss.get_pending_confirm(session_id)
            if not remaining:
                yield self._ev("done")
                return
        if pending_confirm:
            await self._ss.delete_pending_confirm(session_id)

        # 检查 skill_candidates
        skill_cands = await self._ss.get_skill_candidates(session_id)
        if skill_cands and not self._is_new_task(user_message):
            candidates = skill_cands.get("candidates", [])
            matched = _match_skill_selection(user_message, candidates)
            if matched:
                await self._ss.delete_skill_candidates(session_id)
                async for ev in self._load_selected_skill(session_id, matched, trace_cb):
                    yield ev
                yield self._ev("done")
                return
            elif _is_fallback_selection(user_message, candidates):
                await self._ss.delete_skill_candidates(session_id)
            else:
                await self._ss.delete_skill_candidates(session_id)

        system_prompt = self._build_system_prompt("")

        tool_ctx = ToolContext(
            session_id=session_id,
            loader=self._loader,
            registry=self._reg,
            chat_history=self._ch,
            session_service=self._ss,
            container=None,
            trace_callback=trace_cb,
            current_outline=current_outline,
            has_outline=has_outline,
        )
        tool_ctx._trace_id = trace.trace_id

        reply_content = ""
        collected_thinking: list[dict] = []
        outline_md = ""
        _outline_building = False

        async for sse_str in self._engine.run(
            session_id=session_id,
            user_message=user_message,
            system_prompt=system_prompt,
            chat_history=chat_history,
            tool_ctx=tool_ctx,
            trace_callback=trace_cb,
        ):
            yield sse_str
            try:
                p = json.loads(sse_str)
                t = p.get("type", "")
                if t == "chat_reply":
                    reply_content = p.get("content", "")
                elif t == "thinking_step":
                    e = {"step": p["step"], "status": p["status"], "detail": p["detail"], "data": p.get("data")}
                    if e["status"] == "done":
                        for idx in range(len(collected_thinking) - 1, -1, -1):
                            if collected_thinking[idx]["step"] == e["step"] and collected_thinking[idx]["status"] == "running":
                                collected_thinking[idx] = e
                                break
                        else:
                            collected_thinking.append(e)
                    else:
                        collected_thinking.append(e)
                elif t == "outline_done":
                    _outline_building = False
                elif t == "outline_chunk":
                    if not _outline_building:
                        outline_md = ""
                        _outline_building = True
                    outline_md += p.get("content", "")
            except Exception as e:
                logger.debug(f"SSE 事件解析失败（跳过）: {e} raw={sse_str[:100]!r}")

        meta: dict = {}
        if collected_thinking:
            meta["thinking"] = collected_thinking
        if outline_md:
            meta["outline_md"] = outline_md

        await self._ch.add_message(
            session_id, "assistant",
            reply_content or "已处理完毕。",
            msg_type="text",
            metadata=meta if meta else None,
        )

        yield self._ev("done")

    def _is_new_task(self, message: str) -> bool:
        new_task_kws = ["帮我分析", "帮我看", "帮我评估", "生成报告", "删除", "保存", "沉淀", "阈值改为", "分析", "评估"]
        return any(kw in message for kw in new_task_kws)

    async def _load_selected_skill(self, session_id: str, candidate: dict, trace_cb=None) -> None:
        from agent.context import SkillResult
        skill_dir = candidate.get("skill_dir", "")
        display_name = candidate.get("display_name", skill_dir)

        yield self._ev("skill_selected", skill_id=candidate.get("skill_id"), display_name=display_name)
        yield self._ev("chat_reply", content=f"正在加载「{display_name}」的大纲...")

        executor = self._loader.get_executor("customer-analysis")
        if not executor or not hasattr(executor, "load_skill_outline"):
            yield self._ev("chat_reply", content="大纲加载器未就绪，请稍后再试。")
            return

        loaded = executor.load_skill_outline(skill_dir)
        if not loaded:
            yield self._ev("thinking_step", step="skill_load", status="done",
                           detail=f"Skill 大纲文件缺失，回退 GraphRAG: {skill_dir}")
            query = candidate.get("display_name", "")
            skill_ctx = SkillContext(
                session_id=session_id,
                user_message=query,
                params={"query": query},
                trace_callback=trace_cb,
            )
            result = None
            async for item in executor.execute(skill_ctx):
                if isinstance(item, SkillResult):
                    result = item
                    if result.success:
                        subtree = result.data.get("subtree")
                        if subtree:
                            anchor = result.data.get("anchor") or {}
                            await self._ch.save_outline_state(session_id, subtree, anchor)
                elif isinstance(item, str):
                    yield item
            if result and result.success:
                yield self._ev("chat_reply", content="大纲已生成，请查看右侧面板。\n\n请查看右侧大纲面板。")
            return

        outline_json, outline_md = loaded
        executor.merge_paragraph(outline_json, skill_dir=skill_dir)
        anchor_info = {"name": outline_json.get("name", ""), "level": outline_json.get("level", 2),
                       "skill_dir": skill_dir}
        await self._ch.save_outline_state(session_id, outline_json, anchor_info)

        yield json.dumps({"type": "outline_chunk", "content": outline_md}, ensure_ascii=False)
        yield json.dumps({"type": "outline_done", "anchor": anchor_info}, ensure_ascii=False)
        yield self._ev("outline_updated", outline_json=outline_json)
        yield self._ev("chat_reply", content=f"已加载「{display_name}」的大纲，请查看右侧面板。\n\n请查看右侧大纲面板。")

    async def _confirm(self, ctx: AgentContext, user_message: str, trace_cb=None) -> AsyncGenerator[str, None]:
        pending = ctx.pending_confirm_options
        if not pending:
            yield self._ev("chat_reply", content="确认已超时，请重新开始。")
            return

        node_id = None
        msg = user_message.strip().upper()
        option_map = {"A": 0, "B": 1, "C": 2}
        if msg in option_map:
            idx = option_map[msg]
            if idx < len(pending):
                node_id = pending[idx].get("id")
        if not node_id:
            for o in pending:
                if o.get("name", "") in user_message or o.get("label", "") in user_message:
                    node_id = o.get("id")
                    break
        if not node_id:
            return

        await self._ss.delete_pending_confirm(ctx.session_id)
        yield self._ev("chat_reply", content="正在生成大纲...")

        executor = self._loader.get_executor("customer-analysis")
        if executor and hasattr(executor, "execute_from_node"):
            from agent.context import SkillResult
            result = None
            async for item in executor.execute_from_node(
                SkillContext(session_id=ctx.session_id, user_message="", trace_callback=trace_cb),
                node_id,
            ):
                if isinstance(item, SkillResult):
                    result = item
                elif isinstance(item, str):
                    yield item
            if result and result.success:
                subtree = result.data.get("subtree")
                await self._ch.save_outline_state(
                    ctx.session_id,
                    subtree,
                    result.data.get("anchor"),
                )
                if subtree:
                    yield self._ev("outline_updated", outline_json=subtree)
                yield self._ev("chat_reply", content="大纲已生成，请查看右侧面板。\n\n请查看右侧大纲面板。")
                await self._ch.add_message(ctx.session_id, "assistant", "已生成大纲", msg_type="outline")

    @staticmethod
    def _ev(t: str, **kw) -> str:
        return json.dumps({"type": t, **kw}, ensure_ascii=False)

"""Lead Agent —— 基于 SimpleReActEngine 的 ReAct 循环架构。

替代旧的 Plan → Execute → Reflect 架构：
  LLM 自主决策工具调用 → 执行工具 → 观察结果 → 继续决策

保留的功能：
  - 中间件链（历史/大纲状态/待确认）
  - L5 层级确认 (_confirm)
  - 状态持久化 (_persist)
  - SSE 事件兼容（tool_call/tool_result 为新增，旧事件仍透传）
  - Memory 注入（<memory> 块）
"""
import json
import logging
from typing import AsyncGenerator, Optional

from agent.context import AgentContext, SkillContext
from agent.middleware.base import MiddlewareChain
from agent.react_engine import SimpleReActEngine
from agent.skill_registry import SkillRegistry
from agent.skill_loader import SkillLoader
from agent.tool_registry import ToolContext, ToolRegistry
from agent.tool_definitions import register_all_tools
from llm.service import LLMService
from services.chat_history import ChatHistoryService
from services.session_service import SessionService
from utils.trace_logger import TraceLogger

logger = logging.getLogger(__name__)


def _match_skill_selection(message: str, candidates: list) -> dict:
    """从用户消息中匹配选择的 skill candidate。
    匹配顺序：字母选项（A-E）→ display_name 包含匹配 → skill_id 匹配 → bigram 关键词匹配。
    返回匹配的 candidate dict，无匹配返回 None。
    """
    if not candidates:
        return None

    msg = message.strip()

    # 字母选项（A/B/C/D/E，大小写不敏感）
    import re
    letter_match = re.match(r'^([A-Ea-e])\s*[.。、]?$', msg)
    if letter_match:
        label = letter_match.group(1).upper()
        for c in candidates:
            if c.get("label", "") == label:
                return c

    # display_name 包含匹配（用户输入了能力名称的一部分）
    msg_lower = msg.lower()
    for c in candidates:
        name = c.get("display_name", "")
        if name and name in msg:
            return c

    # skill_id 直接匹配
    for c in candidates:
        if c.get("skill_id", "") in msg:
            return c

    # bigram 关键词匹配（中文 2-gram）
    zh_bigrams = set()
    for i in range(len(msg) - 1):
        if '\u4e00' <= msg[i] <= '\u9fff' and '\u4e00' <= msg[i + 1] <= '\u9fff':
            zh_bigrams.add(msg[i:i + 2])

    best_score = 0
    best_cand = None
    for c in candidates:
        name = c.get("display_name", "")
        score = sum(1 for i in range(len(name) - 1)
                    if '\u4e00' <= name[i] <= '\u9fff' and '\u4e00' <= name[i + 1] <= '\u9fff'
                    and name[i:i + 2] in zh_bigrams)
        if score > best_score:
            best_score = score
            best_cand = c

    if best_score >= 2:
        return best_cand

    return None


def _is_fallback_selection(message: str, candidates: list) -> bool:
    """检测用户是否选择了 fallback 选项（不需要已有能力，直接搜索）。"""
    fallback_kws = ["不需要", "其他", "直接搜索", "重新搜索", "没有合适", "都不对", "都不符合"]
    return any(kw in message for kw in fallback_kws)


# ─── System Prompt 基础指令 ───
BASE_INSTRUCTIONS = """\
你是智能看网分析助手。帮助用户生成网络分析大纲，并支持专家将看网逻辑固化为可复用大纲模板。
系统最终产物是大纲 JSON，不生成报告。

## 工具调用诚信规则【最高优先级，绝对不得违反】
- 【严禁】在未实际调用工具的情况下，用文字声称已完成工具操作（如"已为您修改了阈值"）
- 【严禁】假装或推测工具结果：必须真实调用工具并等待返回，才能向用户报告操作结果
- 每一句"已完成/已修改/已注入/已执行"，背后必须有对应的实际工具调用记录
- 每轮对话都是独立的工具执行序列：即使上一轮对话中已执行过相同操作，本轮同类请求也必须重新执行完整工具流程
- 如果不确定需要调用哪个工具，宁可多调用一次 get_session_status/get_current_outline 确认，也不得直接用文字回复

## search_skill 失败处理规则
- 若 search_skill 返回 success=false 且错误信息包含"未在知识库中找到"，说明知识库中没有匹配的分析场景，必须直接将错误信息转述给用户，【严禁重试 search_skill 或编造结果】
- 不得因为一次失败就换更宽泛的描述重新搜索，这样会引入无关场景

## 核心工作方式
1. 首先调用 get_session_status 了解当前会话状态
2. 根据用户意图选择合适的工具序列
3. 复杂任务前先调用 read_skill_file 阅读对应技能的工作流指导

## 意图识别规则（先调 get_session_status 了解当前状态）

### 当 has_outline=true（会话已有大纲）时：
- 用户要求删除/不看某节点（"删除XX"/"去掉XX"/"不看XX"）→ 直接调 clip_outline，不要再调 search_skill
- 用户修改参数/阈值/过滤条件（"阈值改为XX"/"改成XX%"/"只看XX行业"/"筛选XX"）→
  ① 必须先调 get_current_outline 获取最新大纲 JSON（不得凭记忆或历史对话猜测 node_id）
  ② 从返回的 JSON 中找到所有包含该参数的 L5 节点的 node_id
  ③ 对每个目标节点分别调用 inject_params(node_id, param_key, param_value, operator)
  ④ 全部注入完成后，告知用户哪些节点已更新
  【注意】即使本轮与上一轮修改的是同一参数，也必须重新执行 ①②③ 全部步骤
- 用户说"保存/沉淀/确认保存" → persist_outline

### 当 has_outline=false（会话无大纲）时：
- 用户输入超过 80 字的看网逻辑文本（描述分析经验/规则）→ understand_intent(expert_input)，完成后停止等待用户指示
- 用户询问网络分析/评估（短句）→ 先调 skill_router(query) 检索已沉淀模板；若有候选则等待用户选择；若无候选则直接调 search_skill(query)

## 关键约束【严格遵守，不得违反】
- 【禁止】has_outline=true 时因用户说"删除X"而调用 search_skill，应直接调 clip_outline
- 【禁止】understand_intent 完成后自动调用 persist_outline，必须等用户明确说"保存"
- persist_outline 必须等用户明确说"保存/沉淀/确认保存"后才调用，绝不主动触发
"""

# ─── Skill System Prompt 段落（由 SkillRegistry 生成）───
SKILL_SYSTEM_TEMPLATE = """\
<skill_system>
调用工具时，遇到复杂任务先用 read_skill_file(<skill_name>) 阅读工作流指导。
只在需要时读取，不要预先读取所有技能。

<available_skills>
{skill_entries}
</available_skills>
</skill_system>"""


class LeadAgent:

    def __init__(
        self,
        llm_service: LLMService,
        middleware_chain: MiddlewareChain,
        skill_registry: SkillRegistry,
        skill_loader: SkillLoader,
        chat_history: ChatHistoryService,
        session_service: SessionService,
        memory_store=None,    # Optional[MemoryStore]
        memory_updater=None,  # Optional[MemoryUpdater]
    ):
        self._llm = llm_service
        self._mw = middleware_chain
        self._reg = skill_registry
        self._loader = skill_loader
        self._ch = chat_history
        self._ss = session_service
        self._memory = memory_store
        self._memory_updater = memory_updater

        # 构建 ToolRegistry
        self._tool_registry = ToolRegistry()
        register_all_tools(self._tool_registry)

        # 构建 ReAct 引擎
        self._engine = SimpleReActEngine(llm_service, self._tool_registry)

        # 预生成技能系统 prompt 段落
        self._skill_system_prompt = self._build_skill_system_prompt()

    def _build_skill_system_prompt(self) -> str:
        """生成 <skill_system> 块，只列出 builtin 技能（custom skill 通过 skill_router 动态路由）。"""
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
        """组装完整 system prompt。"""
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

    async def handle_message(self, session_id: str, user_message: str) -> AsyncGenerator[str, None]:
        await self._ch.ensure_session(session_id)
        trace = TraceLogger(session_id=session_id)
        trace.log("request.start", data={"user_message": user_message})
        trace_cb = self._make_trace_callback(session_id, trace.trace_id)

        # ─── 中间件链：先填充上下文（此时用户消息尚未入库，避免 HistoryMiddleware 重复加载）───
        ctx = AgentContext(session_id=session_id, user_message=user_message, trace_id=trace.trace_id)
        ctx = await self._mw.run_before(ctx)

        # 中间件之后再入库，确保 chat_history 不含当前轮用户消息（react_engine 会单独追加）
        await self._ch.add_message(session_id, "user", user_message)

        # 设置会话标题（首条消息）
        msgs = await self._ch.get_messages(session_id, limit=2)
        if len(msgs) <= 1:
            title = user_message[:20] + ("..." if len(user_message) > 20 else "")
            await self._ch.update_session_title(session_id, title)

        # ─── 检查 L5 待确认 ───
        if ctx.has_pending_confirm and ctx.pending_confirm_options:
            if not self._is_new_task(user_message):
                # 非新任务指令：尝试匹配确认选项（名称匹配 / A/B/C）
                async for ev in self._confirm(ctx, user_message, trace_cb):
                    yield ev
                # _confirm 成功时已删除 pending；若 pending 还在，说明匹配失败
                remaining = await self._ss.get_pending_confirm(ctx.session_id)
                if not remaining:
                    yield self._ev("done")
                    return
            # 新任务指令 或 确认匹配失败：清理 pending，继续正常 ReAct 流程
            await self._ss.delete_pending_confirm(ctx.session_id)

        # ─── 检查 skill_candidates（上一轮 skill_router 推送的候选列表）───
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
                # 用户选择"其他/不需要/直接搜索"
                await self._ss.delete_skill_candidates(session_id)
                # 继续正常 ReAct 流程，不拦截
            # 其他情况（用户重新描述需求等）：清理候选，继续 ReAct
            else:
                await self._ss.delete_skill_candidates(session_id)

        # Memory 注入已禁用：跨会话污染风险高于收益，保留代码结构便于日后重新启用
        system_prompt = self._build_system_prompt("")

        # ─── 构建 ToolContext ───
        tool_ctx = ToolContext(
            session_id=session_id,
            loader=self._loader,
            registry=self._reg,
            chat_history=self._ch,
            session_service=self._ss,
            container=None,
            trace_callback=trace_cb,
            current_outline=ctx.current_outline,
            has_outline=ctx.has_outline,
        )
        tool_ctx._trace_id = trace.trace_id  # 传给 engine 用于 tool_call_traces

        # ─── 运行 ReAct 引擎 ───
        reply_content = ""
        collected_thinking: list[dict] = []
        outline_md = ""
        _outline_building = False  # 追踪是否在新一轮大纲流中

        async for sse_str in self._engine.run(
            session_id=session_id,
            user_message=user_message,
            system_prompt=system_prompt,
            chat_history=ctx.chat_history,
            tool_ctx=tool_ctx,
            trace_callback=trace_cb,
        ):
            yield sse_str
            # 收集最终回复内容和产物，用于持久化
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

        # ─── 保存助手消息 + 产物 metadata ───
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

        # Memory 异步更新已禁用（与注入同步关闭）
        # if self._memory_updater:
        #     await self._memory_updater.enqueue_update(session_id, user_message, reply_content)

        yield self._ev("done")

    def _is_new_task(self, message: str) -> bool:
        """判断用户消息是否为新任务指令（而非 L5/skill 确认回复）。"""
        new_task_kws = ["帮我分析", "帮我看", "帮我评估", "生成报告", "删除", "保存", "沉淀", "阈值改为", "分析", "评估"]
        return any(kw in message for kw in new_task_kws)

    async def _load_selected_skill(self, session_id: str, candidate: dict, trace_cb=None) -> None:
        """用户选择了某个 skill candidate，直接加载其大纲。"""
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
            # 大纲文件缺失，回退到 search_skill GraphRAG 流程
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
        """处理 L5 层级确认。"""
        pending = ctx.pending_confirm_options
        if not pending:
            yield self._ev("chat_reply", content="确认已超时，请重新开始。")
            return

        # 解析用户选择
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
            # 匹配失败，不产生输出，由调用方检测 pending 仍存在后清理并继续 ReAct
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

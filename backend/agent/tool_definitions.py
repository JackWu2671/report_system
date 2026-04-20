"""工具定义 —— ReAct 工具的 schema + 执行函数。

每个工具函数签名：
  async def fn(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]
  yield {"sse": str}          → 透传 SSE 事件
  yield {"result": SkillResult}  → 工具执行结果（最后一次）

工具分类：
  基础工具：read_skill_file, get_session_status
  运行态：search_skill, skill_router, get_current_outline, clip_outline, inject_params
  设计态：understand_intent, persist_outline
"""

import json
import logging
import os
import re
from typing import AsyncGenerator

from agent.context import SkillContext, SkillResult
from agent.tool_registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)


def _query_matches_anchor(query: str, anchor_name: str) -> bool:
    """检查查询内容与锚点名称是否有字符级关联。

    用于过滤 L1 级别的降级误匹配：当 LLM 无法找到精确锚点时会选
    顶层宽泛节点（L1），此时锚点名与用户意图往往毫无关联。
    策略：提取锚点名中的中文 2-gram 和长度≥3 的英文词，
    若均不出现在 query 中则判定为无关。
    """
    if not anchor_name:
        return False
    zh_segs = re.findall(r'[\u4e00-\u9fff]+', anchor_name)
    for seg in zh_segs:
        for i in range(len(seg) - 1):
            if seg[i:i + 2] in query:
                return True
    en_words = re.findall(r'[A-Za-z0-9]+', anchor_name)
    for word in en_words:
        if len(word) >= 3 and word.lower() in query.lower():
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 基础工具
# ─────────────────────────────────────────────────────────────────────────────

async def _read_skill_file(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
    path = args.get("path", "")
    if not path:
        yield {"result": SkillResult(False, "缺少 path 参数")}
        return
    try:
        content = tool_ctx.registry.load_full_content(path)
        if not content and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        if content:
            yield {"result": SkillResult(True, f"已读取 {path}", data={"content": content})}
        else:
            yield {"result": SkillResult(False, f"未找到文件: {path}")}
    except Exception as e:
        logger.error(f"read_skill_file 失败 path={path}: {e}", exc_info=True)
        yield {"result": SkillResult(False, f"读取失败: {e}")}


async def _get_session_status(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
    sid = tool_ctx.session_id
    try:
        has_outline = await tool_ctx.chat_history.has_outline(sid)
        outline_state = await tool_ctx.chat_history.get_outline_state(sid) if has_outline else None
        # 检查 logic-persist 设计态缓存
        cached_ctx_key = ""
        try:
            cached = await tool_ctx.session_service.redis.get(f"logic_persist_ctx:{sid}")
            if cached:
                cached_ctx_key = sid
        except Exception as e:
            logger.debug(f"Redis 缓存检查失败（可忽略）: {e}")
        pending = await tool_ctx.session_service.get_pending_confirm(sid)

        status = {
            "has_outline": has_outline,
            "has_cached_context": bool(cached_ctx_key),
            "cached_context_key": cached_ctx_key or "",
            "has_pending_confirm": bool(pending),
            "outline_summary": "",
        }
        if outline_state and outline_state.get("outline_json"):
            tree = outline_state["outline_json"]
            names = []
            def _collect(node, depth=0):
                if depth > 2: return
                if node.get("name"): names.append(node["name"])
                for c in node.get("children", []): _collect(c, depth + 1)
            _collect(tree)
            status["outline_summary"] = "、".join(names[:5])

        yield {"result": SkillResult(True, "已获取会话状态", data=status)}
    except Exception as e:
        logger.error(f"get_session_status 失败 session={tool_ctx.session_id}: {e}", exc_info=True)
        yield {"result": SkillResult(False, f"获取状态失败: {e}")}


# ─────────────────────────────────────────────────────────────────────────────
# 运行态工具
# ─────────────────────────────────────────────────────────────────────────────

async def _search_skill(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
    query = args.get("query", "")
    if not query:
        yield {"result": SkillResult(False, "缺少 query 参数")}
        return

    executor = tool_ctx.loader.get_executor("customer-analysis")
    if not executor:
        logger.error("search_skill: customer-analysis 执行器未加载")
        yield {"result": SkillResult(False, "customer-analysis 执行器未加载")}
        return

    logger.info(f"search_skill: query={query!r} session={tool_ctx.session_id}")
    skill_ctx = SkillContext(
        session_id=tool_ctx.session_id,
        user_message=query,
        params={"query": query},
        trace_callback=tool_ctx.trace_callback,
    )
    result = None
    async for item in executor.execute(skill_ctx):
        if isinstance(item, SkillResult):
            result = item
            if result.success:
                subtree = result.data.get("subtree")
                if subtree:
                    anchor = result.data.get("anchor") or {}
                    anchor_level = anchor.get("level", -1)
                    anchor_name = anchor.get("name", "")

                    if anchor_level <= 1 and not _query_matches_anchor(query, anchor_name):
                        logger.warning(
                            f"search_skill: L1 降级误匹配，拒绝 anchor={anchor_name!r} query={query!r}"
                        )
                        result = SkillResult(
                            False,
                            f"未在知识库中找到与「{query}」直接匹配的分析场景。"
                            f"请尝试更具体的描述，例如：「帮我分析fgOTN部署机会」"
                            f"或「分析政企OTN站点容量」。",
                        )
                    else:
                        tool_ctx.current_outline = subtree
                        tool_ctx.has_outline = True
                        if anchor:
                            await tool_ctx.chat_history.save_outline_state(
                                tool_ctx.session_id,
                                subtree,
                                anchor,
                            )

                if result.success and (result.data or {}).get("type") == "confirm_required":
                    yield {"sse": json.dumps({
                        "type": "confirm_required",
                        "indicator_name": result.data.get("indicator_name", ""),
                        "full_path": result.data.get("full_path", ""),
                        "ancestors": result.data.get("ancestors", []),
                    }, ensure_ascii=False)}
        elif isinstance(item, str):
            yield {"sse": item}

    final = result or SkillResult(False, "大纲搜索失败")
    logger.info(f"search_skill: 完成 success={final.success} summary={final.summary!r}")
    yield {"result": final}


async def _skill_router(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
    query = args.get("query", "")
    if not query:
        yield {"result": SkillResult(False, "缺少 query 参数")}
        return

    executor = tool_ctx.loader.get_executor("template-router")
    if not executor:
        logger.warning("skill_router: template-router 执行器未加载，跳过路由")
        yield {"result": SkillResult(True, "template_router 未加载，跳过", data={"candidates": []})}
        return

    ctx = SkillContext(
        session_id=tool_ctx.session_id,
        user_message=query,
        params={"query": query},
        trace_callback=tool_ctx.trace_callback,
    )
    result = None
    async for item in executor.execute(ctx):
        if isinstance(item, SkillResult):
            result = item
        elif isinstance(item, str):
            yield {"sse": item}

    yield {"result": result or SkillResult(False, "template_router 执行失败")}


async def _get_current_outline(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
    try:
        state = await tool_ctx.chat_history.get_outline_state(tool_ctx.session_id)
        if state and state.get("outline_json"):
            yield {"result": SkillResult(True, "已获取大纲", data={
                "outline_json": state["outline_json"],
                "anchor_info": state.get("anchor_info"),
            })}
        else:
            yield {"result": SkillResult(False, "当前会话没有大纲")}
    except Exception as e:
        logger.error(f"get_current_outline 失败 session={tool_ctx.session_id}: {e}", exc_info=True)
        yield {"result": SkillResult(False, f"获取大纲失败: {e}")}


async def _clip_outline(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
    instruction = args.get("instruction", "")
    if not instruction:
        yield {"result": SkillResult(False, "缺少 instruction 参数")}
        return

    executor = tool_ctx.loader.get_executor("outline-clip")
    if not executor:
        yield {"result": SkillResult(False, "outline-clip 执行器未加载")}
        return

    state = await tool_ctx.chat_history.get_outline_state(tool_ctx.session_id)
    current_outline = state.get("outline_json") if state else tool_ctx.current_outline

    skill_ctx = SkillContext(
        session_id=tool_ctx.session_id,
        user_message=instruction,
        params={"instructions": instruction},
        current_outline=current_outline,
        trace_callback=tool_ctx.trace_callback,
    )
    result = None
    async for item in executor.execute(skill_ctx):
        if isinstance(item, SkillResult):
            result = item
            if result.success and result.data.get("updated_outline"):
                tool_ctx.current_outline = result.data["updated_outline"]
                await tool_ctx.chat_history.save_outline_state(
                    tool_ctx.session_id, result.data["updated_outline"]
                )
        elif isinstance(item, str):
            yield {"sse": item}

    yield {"result": result or SkillResult(False, "大纲裁剪失败")}


async def _inject_params(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
    param_key = args.get("param_key", "")
    param_value = args.get("param_value", "")
    node_id = args.get("node_id", "")
    operator = args.get("operator", "eq")
    if not param_key:
        yield {"result": SkillResult(False, "缺少 param_key 参数")}
        return

    executor = tool_ctx.loader.get_executor("param-inject")
    if not executor:
        yield {"result": SkillResult(False, "param-inject 执行器未加载")}
        return

    skill_ctx = SkillContext(
        session_id=tool_ctx.session_id,
        user_message="",
        params={"param_key": param_key, "param_value": param_value,
                "node_id": node_id, "operator": operator},
        trace_callback=tool_ctx.trace_callback,
    )
    result = None
    async for item in executor.execute(skill_ctx):
        if isinstance(item, SkillResult):
            result = item
        elif isinstance(item, str):
            yield {"sse": item}

    yield {"result": result or SkillResult(False, "参数注入失败")}


# ─────────────────────────────────────────────────────────────────────────────
# 设计态工具
# ─────────────────────────────────────────────────────────────────────────────

async def _understand_intent(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
    """启动完整设计态流程（四步：意图理解→结构提取→大纲设计→数据绑定）。
    完成后系统发送沉淀确认提示，等待用户决策。
    """
    expert_input = args.get("expert_input", "")
    if not expert_input:
        yield {"result": SkillResult(False, "缺少 expert_input 参数")}
        return

    executor = tool_ctx.loader.get_executor("logic-persist")
    if not executor:
        logger.error("understand_intent: logic-persist 执行器未加载")
        yield {"result": SkillResult(False, "logic-persist 执行器未加载")}
        return

    logger.info(f"understand_intent: 启动设计流程 session={tool_ctx.session_id}")
    skill_ctx = SkillContext(
        session_id=tool_ctx.session_id,
        user_message=expert_input,
        params={"expert_input": expert_input},
        current_outline=tool_ctx.current_outline,
        trace_callback=tool_ctx.trace_callback,
    )
    result = None
    async for item in executor.execute(skill_ctx):
        if isinstance(item, SkillResult):
            result = item
        elif isinstance(item, str):
            yield {"sse": item}

    final = result or SkillResult(False, "设计态流程失败")
    logger.info(f"understand_intent: 完成 success={final.success}")

    if final.success and final.data.get("outline_json"):
        try:
            await tool_ctx.chat_history.save_outline_state(
                tool_ctx.session_id,
                final.data["outline_json"],
            )
            tool_ctx.has_outline = True
            tool_ctx.current_outline = final.data["outline_json"]
        except Exception as e:
            logger.warning(f"understand_intent: 大纲持久化失败（可忽略）: {e}")

    yield {"result": final}


async def _persist_outline(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
    """将设计态成果沉淀为大纲模板（outline_template.json）并写入 templates/ 目录。"""
    context_key = args.get("context_key", tool_ctx.session_id)

    executor = tool_ctx.loader.get_executor("logic-persist")
    if not executor:
        yield {"result": SkillResult(False, "logic-persist 执行器未加载")}
        return

    skill_ctx = SkillContext(
        session_id=tool_ctx.session_id,
        user_message="",
        params={"context_key": context_key},
        trace_callback=tool_ctx.trace_callback,
    )
    result = None
    async for item in executor.execute(skill_ctx):
        if isinstance(item, SkillResult):
            result = item
        elif isinstance(item, str):
            yield {"sse": item}

    yield {"result": result or SkillResult(False, "大纲模板沉淀失败")}


# ─────────────────────────────────────────────────────────────────────────────
# 注册入口
# ─────────────────────────────────────────────────────────────────────────────

def register_all_tools(registry: ToolRegistry):
    """注册所有工具到 ToolRegistry。"""

    # ── 基础工具 ──
    registry.register(
        name="read_skill_file",
        description=(
            "读取指定路径的 SKILL.md 文件全文。执行复杂的看网分析任务前，"
            "先调用此工具了解该技能的完整工作流和步骤要求。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "技能名称（如 logic-persist）或 SKILL.md 文件路径"},
            },
            "required": ["path"],
        },
        fn=_read_skill_file,
    )

    registry.register(
        name="get_session_status",
        description=(
            "获取当前会话状态：是否已有大纲、是否有待沉淀的设计态缓存、"
            "是否有待用户确认的层级选择。在规划下一步操作前调用。"
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        fn=_get_session_status,
    )

    # ── 运行态工具 ──
    registry.register(
        name="skill_router",
        description=(
            "检索已沉淀的大纲模板并通过 LLM 精排，返回候选列表供用户选择。"
            "用户提出看网分析需求时优先调用此工具（在 search_skill 之前）。"
            "若有候选结果会推送 skill_candidates 事件，等待用户选择后再继续。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户的分析需求描述"},
            },
            "required": ["query"],
        },
        fn=_skill_router,
    )

    registry.register(
        name="search_skill",
        description=(
            "通过 GraphRAG 在知识库中搜索匹配的分析场景，生成大纲 JSON。"
            "用户提出网络分析需求且 skill_router 无匹配时调用。"
            "若返回 success=false 且提示未找到匹配场景，应直接告知用户，不得重试。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户的分析需求描述"},
            },
            "required": ["query"],
        },
        fn=_search_skill,
    )

    registry.register(
        name="get_current_outline",
        description="获取当前会话的完整大纲 JSON。在执行裁剪或参数注入前调用以了解现有大纲结构。",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        fn=_get_current_outline,
    )

    registry.register(
        name="clip_outline",
        description=(
            "按裁剪指令删除、筛选或保留大纲节点。"
            "用户说'不看XX/删除XX/去掉XX'时调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "instruction": {"type": "string", "description": "裁剪指令，如'删除低阶交叉节点'"},
            },
            "required": ["instruction"],
        },
        fn=_clip_outline,
    )

    registry.register(
        name="inject_params",
        description=(
            "注入运行时过滤参数（行业筛选/阈值修改等）。"
            "用户说'只看XX行业/阈值改为XX/改成XX%'时调用。"
            "调用前必须先执行 get_current_outline 获取大纲 JSON，从中精确找到目标 L5 节点的 node_id，"
            "不得凭记忆猜测 node_id。同一参数涉及多个节点时，需对每个节点分别调用本工具。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "目标 L5 节点的 id（从大纲 JSON 中获取），为空则全局注入",
                },
                "param_key": {"type": "string", "description": "参数名，如 bandwidth_threshold / industry"},
                "param_value": {"type": "string", "description": "参数值，如 '100' / '金融'"},
                "operator": {
                    "type": "string",
                    "description": "比较运算符：lt/lte/gt/gte/eq，默认 eq",
                    "enum": ["lt", "lte", "gt", "gte", "eq"],
                },
            },
            "required": ["param_key", "param_value"],
        },
        fn=_inject_params,
    )

    # ── 设计态工具 ──
    registry.register(
        name="understand_intent",
        description=(
            "执行完整看网逻辑沉淀设计流程（四步：意图理解→结构提取→大纲设计→数据绑定）。"
            "用户输入超过 80 字的看网逻辑描述时调用此工具。"
            "流程完成后系统自动向用户发送沉淀确认提示，等待用户决定是否保存为可复用模板。"
            "【重要】此工具调用完成后，不得再调用 persist_outline，必须等待用户明确回复。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "expert_input": {"type": "string", "description": "专家输入的看网逻辑文本（完整原文）"},
            },
            "required": ["expert_input"],
        },
        fn=_understand_intent,
    )

    registry.register(
        name="persist_outline",
        description=(
            "将设计态成果沉淀为可复用大纲模板，写入 templates/ 目录。"
            "【重要】仅在用户明确说'保存/沉淀/确认保存'时才调用此工具。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "context_key": {"type": "string", "description": "缓存 key（通常为 session_id，可从 get_session_status 获取）"},
            },
            "required": [],
        },
        fn=_persist_outline,
    )

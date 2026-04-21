import logging
from typing import AsyncGenerator

from agent.context import SkillResult
from agent.tool_registry import ToolContext

logger = logging.getLogger(__name__)

NAME = "get_session_status"
DESCRIPTION = (
    "获取当前会话状态：是否已有大纲、是否有待沉淀的设计态缓存、"
    "是否有待用户确认的层级选择。在规划下一步操作前调用。"
)
PARAMETERS = {
    "type": "object",
    "properties": {},
    "required": [],
}


async def execute(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
    sid = tool_ctx.session_id
    try:
        has_outline = await tool_ctx.chat_history.has_outline(sid)
        outline_state = await tool_ctx.chat_history.get_outline_state(sid) if has_outline else None
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
        logger.error(f"get_session_status 失败 session={sid}: {e}", exc_info=True)
        yield {"result": SkillResult(False, f"获取状态失败: {e}")}

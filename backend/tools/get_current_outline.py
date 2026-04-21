import logging
from typing import AsyncGenerator

from agent.context import SkillResult
from tools.tool_registry import ToolContext

logger = logging.getLogger(__name__)

NAME = "get_current_outline"
DESCRIPTION = "获取当前会话的完整大纲 JSON。在执行裁剪或参数注入前调用以了解现有大纲结构。"
PARAMETERS = {
    "type": "object",
    "properties": {},
    "required": [],
}


async def execute(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
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

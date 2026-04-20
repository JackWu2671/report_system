import logging
from typing import AsyncGenerator

from agent.context import SkillContext, SkillResult
from agent.tool_registry import ToolContext

logger = logging.getLogger(__name__)

NAME = "persist_outline"
DESCRIPTION = (
    "将设计态成果沉淀为可复用大纲模板，写入 templates/ 目录。"
    "【重要】仅在用户明确说'保存/沉淀/确认保存'时才调用此工具。"
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "context_key": {"type": "string", "description": "缓存 key（通常为 session_id，可从 get_session_status 获取）"},
    },
    "required": [],
}


async def execute(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
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

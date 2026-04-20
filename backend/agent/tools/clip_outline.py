import logging
from typing import AsyncGenerator

from agent.context import SkillContext, SkillResult
from agent.tool_registry import ToolContext

logger = logging.getLogger(__name__)

NAME = "clip_outline"
DESCRIPTION = (
    "按裁剪指令删除、筛选或保留大纲节点。"
    "用户说'不看XX/删除XX/去掉XX'时调用。"
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string", "description": "裁剪指令，如'删除低阶交叉节点'"},
    },
    "required": ["instruction"],
}


async def execute(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
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

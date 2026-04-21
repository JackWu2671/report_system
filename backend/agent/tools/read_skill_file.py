import logging
import os
from typing import AsyncGenerator

from agent.context import SkillResult
from agent.tool_registry import ToolContext

logger = logging.getLogger(__name__)

NAME = "read_skill_file"
DESCRIPTION = (
    "读取指定路径的 SKILL.md 文件全文。执行复杂的看网分析任务前，"
    "先调用此工具了解该技能的完整工作流和步骤要求。"
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "技能名称（如 logic-persist）或 SKILL.md 文件路径"},
    },
    "required": ["path"],
}


async def execute(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
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

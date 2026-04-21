import logging
from typing import AsyncGenerator

from agent.context import SkillContext, SkillResult
from tools.tool_registry import ToolContext

logger = logging.getLogger(__name__)

NAME = "skill_router"
DESCRIPTION = (
    "检索已沉淀的大纲模板并通过 LLM 精排，返回候选列表供用户选择。"
    "用户提出看网分析需求时优先调用此工具（在 search_skill 之前）。"
    "若有候选结果会推送 skill_candidates 事件，等待用户选择后再继续。"
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "用户的分析需求描述"},
    },
    "required": ["query"],
}


async def execute(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
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

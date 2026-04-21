import json
import logging
from typing import AsyncGenerator

from agent.context import SkillContext, SkillResult
from agent.tool_registry import ToolContext
from agent.tools._helpers import query_matches_anchor

logger = logging.getLogger(__name__)

NAME = "search_skill"
DESCRIPTION = (
    "通过 GraphRAG 在知识库中搜索匹配的分析场景，生成大纲 JSON。"
    "用户提出网络分析需求且 skill_router 无匹配时调用。"
    "若返回 success=false 且提示未找到匹配场景，应直接告知用户，不得重试。"
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
                    if anchor_level <= 1 and not query_matches_anchor(query, anchor_name):
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
                                tool_ctx.session_id, subtree, anchor,
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

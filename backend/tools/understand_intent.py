import logging
from typing import AsyncGenerator

from agent.context import SkillContext, SkillResult
from tools.tool_registry import ToolContext

logger = logging.getLogger(__name__)

NAME = "understand_intent"
DESCRIPTION = (
    "执行完整看网逻辑沉淀设计流程（四步：意图理解→结构提取→大纲设计→数据绑定）。"
    "用户输入超过 80 字的看网逻辑描述时调用此工具。"
    "流程完成后系统自动向用户发送沉淀确认提示，等待用户决定是否保存为可复用模板。"
    "【重要】此工具调用完成后，不得再调用 persist_outline，必须等待用户明确回复。"
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "expert_input": {"type": "string", "description": "专家输入的看网逻辑文本（完整原文）"},
    },
    "required": ["expert_input"],
}


async def execute(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
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

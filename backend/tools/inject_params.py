import logging
from typing import AsyncGenerator

from agent.context import SkillContext, SkillResult
from tools.tool_registry import ToolContext

logger = logging.getLogger(__name__)

NAME = "inject_params"
DESCRIPTION = (
    "注入运行时过滤参数（行业筛选/阈值修改等）。"
    "用户说'只看XX行业/阈值改为XX/改成XX%'时调用。"
    "调用前必须先执行 get_current_outline 获取大纲 JSON，从中精确找到目标 L5 节点的 node_id，"
    "不得凭记忆猜测 node_id。同一参数涉及多个节点时，需对每个节点分别调用本工具。"
)
PARAMETERS = {
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
}


async def execute(args: dict, tool_ctx: ToolContext) -> AsyncGenerator[dict, None]:
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

"""ToolRegistry —— ReAct 工具注册与执行中心。

每个工具是一个 async generator：
  yield {"sse": "..."}        → 透传 SSE 事件给前端
  yield {"result": SkillResult}  → 工具执行结果（最后一个 yield）

ToolContext 携带执行工具所需的所有服务引用。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """工具执行时的共享上下文，由 LeadAgent 组装后传入。"""
    session_id: str
    loader: Any           # SkillLoader
    registry: Any         # SkillRegistry
    chat_history: Any     # ChatHistoryService
    session_service: Any  # SessionService
    container: Any        # ServiceContainer
    trace_callback: Optional[Callable] = None
    # 当前会话状态（随执行动态更新）
    current_outline: Optional[dict] = None
    has_outline: bool = False
    llm_service: Optional[Any] = None
    step_results: dict = field(default_factory=dict)


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict          # JSON Schema object
    fn: Callable              # async generator function


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, name: str, description: str, parameters: dict, fn: Callable):
        self._tools[name] = ToolDef(name=name, description=description, parameters=parameters, fn=fn)
        logger.debug(f"ToolRegistry: registered '{name}'")

    def get_openai_tools(self) -> list[dict]:
        """返回 OpenAI function calling 格式的 tools 数组。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    async def execute(
        self, tool_call: dict, tool_ctx: ToolContext
    ) -> AsyncGenerator[dict, None]:
        """执行工具调用，yield {"sse": ...} 或 {"result": SkillResult}。"""
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})
        tool_def = self._tools.get(name)
        if not tool_def:
            from agent.context import SkillResult
            yield {"result": SkillResult(False, f"未知工具: {name}")}
            return
        try:
            async for item in tool_def.fn(args, tool_ctx):
                yield item
        except Exception as e:
            logger.error(f"Tool '{name}' execution error: {e}", exc_info=True)
            from agent.context import SkillResult
            yield {"result": SkillResult(False, f"工具 '{name}' 执行失败: {e}")}

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())

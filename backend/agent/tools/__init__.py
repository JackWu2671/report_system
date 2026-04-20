from agent.tool_registry import ToolRegistry
from agent.tools import (
    clip_outline,
    get_current_outline,
    get_session_status,
    inject_params,
    persist_outline,
    read_skill_file,
    search_skill,
    skill_router,
    understand_intent,
)

_TOOLS = [
    read_skill_file,
    get_session_status,
    skill_router,
    search_skill,
    get_current_outline,
    clip_outline,
    inject_params,
    understand_intent,
    persist_outline,
]


def register_all_tools(registry: ToolRegistry):
    for tool in _TOOLS:
        registry.register(
            name=tool.NAME,
            description=tool.DESCRIPTION,
            parameters=tool.PARAMETERS,
            fn=tool.execute,
        )

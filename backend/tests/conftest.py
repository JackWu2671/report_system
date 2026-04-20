"""共享 pytest fixtures。

包含：
  - FakeLLMService：不发真实 aiohttp 请求的假 LLM 服务
  - mock_tool_ctx：最小 ToolContext mock
  - fake_llm_factory：参数化创建 FakeLLMService 的工厂 fixture
"""

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.context import SkillResult
from agent.tool_registry import ToolContext


# ─── FakeLLMService ───────────────────────────────────────────────────────────

class FakeLLMService:
    """离线 LLM 服务。

    rounds: list[list[dict]]
      每次调用 complete_stream 消费 rounds 中下一个列表，
      按顺序 yield 其中的每个 chunk dict。

    示例：
      FakeLLMService([
          [{"tool_calls": [{"id": "c1", "name": "search_skill", "arguments": {"q": "test"}}]}],
          [{"content": "分析完成"}],
      ])
      第1次 complete_stream 调用 → yield tool_calls chunk
      第2次 complete_stream 调用 → yield content chunk
    """

    def __init__(self, rounds: list[list[dict]]):
        self._rounds = iter(rounds)

    async def complete_stream(
        self, messages, config=None, tools=None
    ) -> AsyncGenerator[dict, None]:
        chunks = next(self._rounds, [{"content": "（FakeLLM: 已无预设轮次）"}])
        for chunk in chunks:
            yield chunk

    async def complete(self, messages, config=None) -> str:
        chunks = next(self._rounds, [{"content": ""}])
        return "".join(c.get("content", "") for c in chunks)

    @property
    def default_model(self) -> str:
        return "fake-model"


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_tool_ctx():
    """最小 ToolContext，所有服务引用均为 MagicMock。"""
    ctx = MagicMock(spec=ToolContext)
    ctx.session_id = "test_session"
    ctx.chat_history = MagicMock()
    ctx.chat_history.get_outline_state = AsyncMock(return_value=None)
    ctx.chat_history.save_outline_state = AsyncMock()
    ctx.loader = MagicMock()
    ctx.loader.get_executor = MagicMock(return_value=None)
    ctx.registry = MagicMock()
    ctx.session_service = MagicMock()
    ctx.container = MagicMock()
    ctx.trace_callback = None
    ctx.has_outline = False
    ctx.current_outline = None
    ctx.step_results = {}
    return ctx


@pytest.fixture
def fake_skill_result_ok():
    return SkillResult(success=True, summary="执行成功", data={"key": "value"})


@pytest.fixture
def fake_skill_result_fail():
    return SkillResult(success=False, summary="执行失败", user_prompt="请重试")

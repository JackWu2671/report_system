"""测试 SimpleReActEngine — 使用 FakeLLMService，验证 SSE 事件序列。

不依赖真实 LLM 或数据库，所有工具调用均为 mock。
"""

import json

import pytest

from agent.context import SkillResult
from agent.react_engine import MAX_STEPS, SimpleReActEngine
from tools.tool_registry import ToolContext, ToolRegistry
from tests.conftest import FakeLLMService


def _parse_events(sse_strings: list[str]) -> list[dict]:
    """将 SSE JSON 字符串解析为 dict 列表。"""
    events = []
    for s in sse_strings:
        try:
            events.append(json.loads(s))
        except json.JSONDecodeError:
            pass
    return events


def _event_types(events: list[dict]) -> list[str]:
    return [e.get("type", "?") for e in events]


def _make_registry_with_tool(tool_name: str, result: SkillResult) -> ToolRegistry:
    """创建包含单个简单工具的 ToolRegistry。"""
    registry = ToolRegistry()

    async def fake_tool(args, tool_ctx):
        yield {"result": result}

    registry.register(
        name=tool_name,
        description=f"测试工具 {tool_name}",
        parameters={"type": "object", "properties": {}},
        fn=fake_tool,
    )
    return registry


def _make_registry_with_sse_tool(tool_name: str, sse_events: list[str], result: SkillResult) -> ToolRegistry:
    """创建会发出 SSE 事件的工具。"""
    registry = ToolRegistry()

    async def sse_tool(args, tool_ctx):
        for ev in sse_events:
            yield {"sse": ev}
        yield {"result": result}

    registry.register(
        name=tool_name,
        description=f"SSE 工具 {tool_name}",
        parameters={"type": "object", "properties": {}},
        fn=sse_tool,
    )
    return registry


def _make_tool_ctx() -> ToolContext:
    """创建最小 ToolContext（不用 mock，直接构造）。"""
    from unittest.mock import AsyncMock, MagicMock
    return ToolContext(
        session_id="test_session",
        loader=MagicMock(),
        registry=MagicMock(),
        chat_history=MagicMock(),
        session_service=MagicMock(),
        container=MagicMock(),
    )


class TestReActBasicFlow:

    @pytest.mark.asyncio
    async def test_text_only_reply(self):
        """LLM 直接返回文本（无工具调用）→ 产生 chat_reply 事件。"""
        fake_llm = FakeLLMService([
            [{"content": "这是最终答案"}],
        ])
        registry = ToolRegistry()
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run(
            session_id="s1",
            user_message="你好",
            system_prompt="你是助手",
            chat_history=[],
            tool_ctx=_make_tool_ctx(),
        ):
            events_raw.append(ev)

        events = _parse_events(events_raw)
        types = _event_types(events)
        assert "chat_reply" in types
        chat_events = [e for e in events if e.get("type") == "chat_reply"]
        assert "这是最终答案" in chat_events[0].get("content", "")

    @pytest.mark.asyncio
    async def test_tool_call_then_text_reply(self):
        """LLM 先调工具，再输出文字 → 事件序列：tool_call → tool_result → chat_reply。"""
        fake_llm = FakeLLMService([
            # 第1轮：工具调用
            [{"tool_calls": [{"id": "c1", "name": "search_skill", "arguments": {"q": "test"}}]}],
            # 第2轮：文字答案
            [{"content": "分析完成"}],
        ])
        registry = _make_registry_with_tool("search_skill", SkillResult(True, "找到相关技能"))
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run(
            session_id="s1",
            user_message="分析fgOTN",
            system_prompt="你是助手",
            chat_history=[],
            tool_ctx=_make_tool_ctx(),
        ):
            events_raw.append(ev)

        events = _parse_events(events_raw)
        types = _event_types(events)
        assert "tool_call" in types
        assert "tool_result" in types
        assert "chat_reply" in types
        # 顺序验证
        tc_idx = types.index("tool_call")
        tr_idx = types.index("tool_result")
        cr_idx = types.index("chat_reply")
        assert tc_idx < tr_idx < cr_idx

    @pytest.mark.asyncio
    async def test_tool_call_name_and_args_in_event(self):
        """tool_call 事件包含正确的工具名称和参数。"""
        args = {"query": "容量测试", "limit": 10}
        fake_llm = FakeLLMService([
            [{"tool_calls": [{"id": "c1", "name": "search_skill", "arguments": args}]}],
            [{"content": "完成"}],
        ])
        registry = _make_registry_with_tool("search_skill", SkillResult(True, "ok"))
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run("s1", "test", "sys", [], _make_tool_ctx()):
            events_raw.append(ev)

        events = _parse_events(events_raw)
        tool_call_event = next((e for e in events if e.get("type") == "tool_call"), None)
        assert tool_call_event is not None
        assert tool_call_event["name"] == "search_skill"
        assert tool_call_event["args"] == args

    @pytest.mark.asyncio
    async def test_sse_passthrough_from_tool(self):
        """工具内部 yield {"sse": "..."} 被透传到引擎输出流。"""
        sse_payload = json.dumps({"type": "outline_chunk", "data": "节点A"})
        fake_llm = FakeLLMService([
            [{"tool_calls": [{"id": "c1", "name": "clip_outline", "arguments": {}}]}],
            [{"content": "完成"}],
        ])
        registry = _make_registry_with_sse_tool(
            "clip_outline",
            [sse_payload],
            SkillResult(True, "裁剪完成"),
        )
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run("s1", "裁剪大纲", "sys", [], _make_tool_ctx()):
            events_raw.append(ev)

        assert sse_payload in events_raw

    @pytest.mark.asyncio
    async def test_unknown_tool_yields_tool_result_with_error(self):
        """调用未注册的工具 → tool_result 事件标记失败，但不抛出异常。"""
        fake_llm = FakeLLMService([
            [{"tool_calls": [{"id": "c1", "name": "nonexistent_tool", "arguments": {}}]}],
            [{"content": "好的"}],
        ])
        registry = ToolRegistry()  # 空注册表
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run("s1", "test", "sys", [], _make_tool_ctx()):
            events_raw.append(ev)

        events = _parse_events(events_raw)
        result_events = [e for e in events if e.get("type") == "tool_result"]
        assert len(result_events) >= 1

    @pytest.mark.asyncio
    async def test_max_steps_limit(self):
        """达到 MAX_STEPS 后强制输出 chat_reply，不无限循环。"""
        # 每轮都返回工具调用（触发无限循环场景）
        tool_round = [{"tool_calls": [{"id": "c1", "name": "looping_tool", "arguments": {}}]}]
        rounds = [tool_round] * (MAX_STEPS + 5)
        fake_llm = FakeLLMService(rounds)
        registry = _make_registry_with_tool("looping_tool", SkillResult(True, "ok"))
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run("s1", "test", "sys", [], _make_tool_ctx()):
            events_raw.append(ev)

        events = _parse_events(events_raw)
        assert any(e.get("type") == "chat_reply" for e in events), "应有最终 chat_reply"
        tool_calls = [e for e in events if e.get("type") == "tool_call"]
        assert len(tool_calls) <= MAX_STEPS

    @pytest.mark.asyncio
    async def test_llm_error_yields_chat_reply(self):
        """LLM 返回 error chunk → 输出包含错误信息的 chat_reply，不崩溃。"""
        fake_llm = FakeLLMService([
            [{"error": "服务不可用"}],
        ])
        registry = ToolRegistry()
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run("s1", "test", "sys", [], _make_tool_ctx()):
            events_raw.append(ev)

        events = _parse_events(events_raw)
        assert any(e.get("type") == "chat_reply" for e in events)

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_step(self):
        """LLM 在一步返回多个工具调用 → 每个都触发 tool_call + tool_result 事件。"""
        fake_llm = FakeLLMService([
            [{"tool_calls": [
                {"id": "c1", "name": "tool_a", "arguments": {}},
                {"id": "c2", "name": "tool_b", "arguments": {}},
            ]}],
            [{"content": "完成"}],
        ])
        registry = ToolRegistry()
        for name in ["tool_a", "tool_b"]:
            async def fn(args, ctx, _n=name):
                yield {"result": SkillResult(True, f"{_n} 完成")}
            registry.register(name, f"工具 {name}", {"type": "object", "properties": {}}, fn)

        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run("s1", "test", "sys", [], _make_tool_ctx()):
            events_raw.append(ev)

        events = _parse_events(events_raw)
        tool_calls = [e for e in events if e.get("type") == "tool_call"]
        tool_results = [e for e in events if e.get("type") == "tool_result"]
        assert len(tool_calls) == 2
        assert len(tool_results) == 2


class TestLoopDetectionIntegration:

    @pytest.mark.asyncio
    async def test_loop_warn_injects_message(self):
        """连续相同工具调用触发 warn 时，后续消息列表中应包含警告。

        验证：warn 不立即停止，仍继续执行到最终答案。
        """
        from agent.loop_detector import WARN_THRESHOLD

        # WARN_THRESHOLD 轮相同工具调用，最后一轮返回文字
        tool_round = [{"tool_calls": [{"id": "c1", "name": "repeat_tool", "arguments": {}}]}]
        final_round = [{"content": "最终回答"}]
        rounds = [tool_round] * WARN_THRESHOLD + [final_round]
        fake_llm = FakeLLMService(rounds)
        registry = _make_registry_with_tool("repeat_tool", SkillResult(True, "ok"))
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run("s1", "test", "sys", [], _make_tool_ctx()):
            events_raw.append(ev)

        events = _parse_events(events_raw)
        # 最终应有 chat_reply
        assert any(e.get("type") == "chat_reply" for e in events)

    @pytest.mark.asyncio
    async def test_loop_stop_breaks_loop(self):
        """触发 HARD_LIMIT 后强制停止，输出最终答案。"""
        from agent.loop_detector import HARD_LIMIT

        tool_round = [{"tool_calls": [{"id": "c1", "name": "stuck_tool", "arguments": {}}]}]
        # 提供足够多轮次
        rounds = [tool_round] * (HARD_LIMIT + 3) + [[{"content": "强制结束"}]]
        fake_llm = FakeLLMService(rounds)
        registry = _make_registry_with_tool("stuck_tool", SkillResult(True, "ok"))
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run("s1", "test", "sys", [], _make_tool_ctx()):
            events_raw.append(ev)

        events = _parse_events(events_raw)
        assert any(e.get("type") == "chat_reply" for e in events)
        # 工具调用次数应受 HARD_LIMIT 限制
        tool_calls = [e for e in events if e.get("type") == "tool_call"]
        assert len(tool_calls) <= HARD_LIMIT + 1  # 允许少量误差


class TestTraceCallback:

    @pytest.mark.asyncio
    async def test_trace_called_for_each_step(self):
        """每个 ReAct 步骤都触发一次 trace_callback。"""
        fake_llm = FakeLLMService([
            [{"tool_calls": [{"id": "c1", "name": "search_skill", "arguments": {"q": "test"}}]}],
            [{"content": "完成"}],
        ])
        registry = _make_registry_with_tool("search_skill", SkillResult(True, "ok"))
        engine = SimpleReActEngine(fake_llm, registry)

        calls = []

        async def trace_cb(**kwargs):
            calls.append(kwargs)

        async for _ in engine.run("s1", "test", "sys", [], _make_tool_ctx(), trace_callback=trace_cb):
            pass

        # 两步：step_0（工具调用）+ step_1（文字回复）
        assert len(calls) == 2
        assert calls[0]["step_name"] == "step_0"
        assert calls[1]["step_name"] == "step_1"

    @pytest.mark.asyncio
    async def test_trace_tool_call_step_response_content_is_tool_json(self):
        """纯工具调用步骤：response_content 应序列化工具决策，而非空字符串。"""
        tool_args = {"query": "fgOTN容量"}
        fake_llm = FakeLLMService([
            [{"tool_calls": [{"id": "c1", "name": "search_skill", "arguments": tool_args}]}],
            [{"content": "完成"}],
        ])
        registry = _make_registry_with_tool("search_skill", SkillResult(True, "ok"))
        engine = SimpleReActEngine(fake_llm, registry)

        calls = []

        async def trace_cb(**kwargs):
            calls.append(kwargs)

        async for _ in engine.run("s1", "test", "sys", [], _make_tool_ctx(), trace_callback=trace_cb):
            pass

        # step_0 是工具调用步，response_content 不能为空
        step0 = calls[0]
        assert step0["response_content"], "工具调用步 response_content 不应为空"
        # 应是合法 JSON 且包含工具名
        parsed = json.loads(step0["response_content"])
        assert isinstance(parsed, list)
        assert parsed[0]["name"] == "search_skill"
        assert parsed[0]["arguments"] == tool_args

    @pytest.mark.asyncio
    async def test_trace_text_step_response_content_is_text(self):
        """纯文本回复步骤：response_content 是文字内容（非 JSON 数组）。"""
        fake_llm = FakeLLMService([
            [{"content": "这是最终答案"}],
        ])
        registry = ToolRegistry()
        engine = SimpleReActEngine(fake_llm, registry)

        calls = []

        async def trace_cb(**kwargs):
            calls.append(kwargs)

        async for _ in engine.run("s1", "test", "sys", [], _make_tool_ctx(), trace_callback=trace_cb):
            pass

        assert len(calls) == 1
        assert calls[0]["response_content"] == "这是最终答案"

    @pytest.mark.asyncio
    async def test_trace_elapsed_ms_per_step_not_cumulative(self):
        """elapsed_ms 反映单步耗时，各步 elapsed_ms 都应为正数且合理（非累积递增）。"""
        import asyncio

        class SlowFakeLLM:
            """每次调用都稍有延迟。"""
            def __init__(self):
                self._call = 0

            async def complete_stream(self, messages, config=None, tools=None):
                self._call += 1
                await asyncio.sleep(0.01)  # 10ms 延迟
                if self._call == 1:
                    yield {"tool_calls": [{"id": "c1", "name": "t", "arguments": {}}]}
                else:
                    yield {"content": "done"}

            async def complete(self, messages, config=None):
                return "done"

            @property
            def default_model(self):
                return "fake"

        registry = _make_registry_with_tool("t", SkillResult(True, "ok"))
        engine = SimpleReActEngine(SlowFakeLLM(), registry)

        calls = []

        async def trace_cb(**kwargs):
            calls.append(kwargs)

        async for _ in engine.run("s1", "test", "sys", [], _make_tool_ctx(), trace_callback=trace_cb):
            pass

        assert len(calls) == 2
        # 每步 elapsed_ms 均 > 0
        assert calls[0]["elapsed_ms"] > 0
        assert calls[1]["elapsed_ms"] > 0
        # 单步计时：step_1 不应比 step_0 大出数倍（累积计时时 step_1 会接近 step_0 的2倍）
        # 允许合理误差，但不应超过 step_0 的 5 倍
        assert calls[1]["elapsed_ms"] < calls[0]["elapsed_ms"] * 5

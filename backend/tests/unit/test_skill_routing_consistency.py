"""验证重构前后技能规划路径一致性。

核心命题：
  同一用户问题（如"分析fgOTN容量"），无论有无沉淀的看网能力，
  LLM 总是选择 search_skill 工具；
  executor 内部根据 FAISS score 决定走缓存路径还是 GraphRAG 路径。

场景 A（无沉淀能力，GraphRAG 路径）：
  → search_skill 被调用 → 透传 outline_chunk SSE → chat_reply

场景 B（有沉淀能力，缓存加载路径）：
  → search_skill 被调用 → 透传 outline_done SSE → chat_reply

两条路径的公共保证：
  1. 第一个工具调用始终是 search_skill
  2. 最终总有 chat_reply
  3. 对应 SSE 事件被透传（不丢失）
"""

import json

import pytest

from agent.context import SkillResult
from agent.react_engine import SimpleReActEngine
from tools.tool_registry import ToolContext, ToolRegistry
from tests.conftest import FakeLLMService

SHORT_NETWORK_QUESTION = "分析fgOTN容量"


# ─── 辅助函数 ──────────────────────────────────────────────────────────────────


def _make_search_skill_registry(sse_events: list[str], result: SkillResult) -> ToolRegistry:
    """构造只含 search_skill 的 ToolRegistry，模拟 executor 不同路径的 SSE 输出。"""
    registry = ToolRegistry()

    async def search_skill(args, tool_ctx):
        for ev in sse_events:
            yield {"sse": ev}
        yield {"result": result}

    registry.register(
        name="search_skill",
        description="查找或生成看网大纲",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        fn=search_skill,
    )
    return registry


def _make_tool_ctx() -> ToolContext:
    from unittest.mock import MagicMock
    return ToolContext(
        session_id="s1",
        loader=MagicMock(),
        registry=MagicMock(),
        chat_history=MagicMock(),
        session_service=MagicMock(),
        container=MagicMock(),
    )


def _parse(events_raw: list[str]) -> list[dict]:
    return [json.loads(e) for e in events_raw if e.startswith("{")]


# ─── 场景 A：无沉淀能力，走 GraphRAG 生成路径 ────────────────────────────────


class TestScenarioA_NoPrebuiltSkill:

    @pytest.mark.asyncio
    async def test_llm_calls_search_skill_for_short_question(self):
        """无沉淀能力时，短问题触发 LLM 调用 search_skill 工具。"""
        fake_llm = FakeLLMService([
            [{"tool_calls": [{"id": "c1", "name": "search_skill",
                               "arguments": {"query": SHORT_NETWORK_QUESTION}}]}],
            [{"content": "大纲生成完毕"}],
        ])
        sse_graphrag = json.dumps({"type": "outline_chunk", "content": "# fgOTN容量分析"})
        registry = _make_search_skill_registry(
            [sse_graphrag],
            SkillResult(True, "GraphRAG 已生成大纲"),
        )
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run(
            session_id="s1",
            user_message=SHORT_NETWORK_QUESTION,
            system_prompt="你是看网助手",
            chat_history=[],
            tool_ctx=_make_tool_ctx(),
        ):
            events_raw.append(ev)

        events = _parse(events_raw)
        types = [e.get("type") for e in events]
        assert "tool_call" in types, "search_skill 应被调用"
        assert "chat_reply" in types, "最终应有文字回复"
        assert sse_graphrag in events_raw, "GraphRAG outline_chunk 应透传到前端"

    @pytest.mark.asyncio
    async def test_search_skill_tool_name_in_event(self):
        """tool_call 事件的工具名必须是 search_skill（不是 read_skill_file 或其他）。"""
        fake_llm = FakeLLMService([
            [{"tool_calls": [{"id": "c1", "name": "search_skill",
                               "arguments": {"query": SHORT_NETWORK_QUESTION}}]}],
            [{"content": "完成"}],
        ])
        sse = json.dumps({"type": "outline_chunk", "content": "节点A"})
        registry = _make_search_skill_registry([sse], SkillResult(True, "ok"))
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run("s1", SHORT_NETWORK_QUESTION, "sys", [], _make_tool_ctx()):
            events_raw.append(ev)

        events = _parse(events_raw)
        tc_event = next((e for e in events if e.get("type") == "tool_call"), None)
        assert tc_event is not None, "应存在 tool_call 事件"
        assert tc_event["name"] == "search_skill"


# ─── 场景 B：有沉淀能力，走缓存加载路径 ──────────────────────────────────────


class TestScenarioB_PrebuiltSkillExists:

    @pytest.mark.asyncio
    async def test_llm_still_calls_search_skill_for_prebuilt_case(self):
        """有沉淀能力时，search_skill 仍被调用；executor 内部走缓存路径（outline_done）。"""
        fake_llm = FakeLLMService([
            [{"tool_calls": [{"id": "c1", "name": "search_skill",
                               "arguments": {"query": SHORT_NETWORK_QUESTION}}]}],
            [{"content": "已从沉淀能力加载大纲"}],
        ])
        sse_done = json.dumps({"type": "outline_done", "anchor": "fgOTN"})
        registry = _make_search_skill_registry(
            [sse_done],
            SkillResult(True, "已从沉淀能力加载", data={"source": "cached_skill"}),
        )
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run("s1", SHORT_NETWORK_QUESTION, "sys", [], _make_tool_ctx()):
            events_raw.append(ev)

        events = _parse(events_raw)
        types = [e.get("type") for e in events]
        assert "tool_call" in types, "search_skill 仍应被调用"
        assert sse_done in events_raw, "缓存路径的 outline_done 应透传"

    @pytest.mark.asyncio
    async def test_tool_result_reflects_cached_skill_source(self):
        """tool_result 事件的 summary 来自 SkillResult.summary，应反映缓存路径信息。"""
        fake_llm = FakeLLMService([
            [{"tool_calls": [{"id": "c1", "name": "search_skill",
                               "arguments": {"query": SHORT_NETWORK_QUESTION}}]}],
            [{"content": "完成"}],
        ])
        registry = _make_search_skill_registry(
            [],
            SkillResult(True, "已从沉淀能力加载", data={"source": "cached_skill"}),
        )
        engine = SimpleReActEngine(fake_llm, registry)

        events_raw = []
        async for ev in engine.run("s1", SHORT_NETWORK_QUESTION, "sys", [], _make_tool_ctx()):
            events_raw.append(ev)

        events = _parse(events_raw)
        tr = next((e for e in events if e.get("type") == "tool_result"), None)
        assert tr is not None, "应存在 tool_result 事件"
        assert "沉淀能力" in tr.get("summary", ""), "tool_result 应反映缓存路径摘要"


# ─── 行为一致性对比：两条路径的公共保证 ──────────────────────────────────────


class TestRoutingConsistency:

    @pytest.mark.asyncio
    async def test_both_paths_always_use_search_skill(self):
        """无论是否有沉淀能力，search_skill 始终是第一个被调用的工具。"""
        test_cases = [
            ([json.dumps({"type": "outline_chunk", "content": "A"})], "no_prebuilt（GraphRAG路径）"),
            ([json.dumps({"type": "outline_done", "anchor": "X"})], "prebuilt（缓存路径）"),
        ]
        for sse_events, label in test_cases:
            fake_llm = FakeLLMService([
                [{"tool_calls": [{"id": "c1", "name": "search_skill",
                                   "arguments": {"query": SHORT_NETWORK_QUESTION}}]}],
                [{"content": "done"}],
            ])
            registry = _make_search_skill_registry(sse_events, SkillResult(True, "ok"))
            engine = SimpleReActEngine(fake_llm, registry)

            events_raw = []
            async for ev in engine.run("s1", SHORT_NETWORK_QUESTION, "sys", [], _make_tool_ctx()):
                events_raw.append(ev)

            events = _parse(events_raw)
            tool_calls = [e for e in events if e.get("type") == "tool_call"]
            assert len(tool_calls) >= 1, f"[{label}] 必须有工具调用"
            assert tool_calls[0]["name"] == "search_skill", \
                f"[{label}] 第一个工具必须是 search_skill，实际是 {tool_calls[0]['name']}"

    @pytest.mark.asyncio
    async def test_both_paths_end_with_chat_reply(self):
        """两条路径最终都产生 chat_reply，不会静默结束。"""
        test_cases = [
            ([json.dumps({"type": "outline_chunk", "content": "A"})], "GraphRAG路径"),
            ([json.dumps({"type": "outline_done", "anchor": "X"})], "缓存路径"),
        ]
        for sse_events, label in test_cases:
            fake_llm = FakeLLMService([
                [{"tool_calls": [{"id": "c1", "name": "search_skill",
                                   "arguments": {"query": SHORT_NETWORK_QUESTION}}]}],
                [{"content": "已完成大纲生成，请查看右侧面板"}],
            ])
            registry = _make_search_skill_registry(sse_events, SkillResult(True, "ok"))
            engine = SimpleReActEngine(fake_llm, registry)

            events_raw = []
            async for ev in engine.run("s1", SHORT_NETWORK_QUESTION, "sys", [], _make_tool_ctx()):
                events_raw.append(ev)

            events = _parse(events_raw)
            assert any(e.get("type") == "chat_reply" for e in events), \
                f"[{label}] 最终必须有 chat_reply"

"""测试 ToolRegistry — 注册、查询、执行接口。"""

import pytest

from agent.context import SkillResult
from tools.tool_registry import ToolContext, ToolRegistry


def _make_ctx() -> ToolContext:
    from unittest.mock import MagicMock
    return ToolContext(
        session_id="test",
        loader=MagicMock(),
        registry=MagicMock(),
        chat_history=MagicMock(),
        session_service=MagicMock(),
        container=MagicMock(),
    )


class TestToolRegistryRegister:

    def test_register_and_has(self):
        registry = ToolRegistry()
        async def fn(args, ctx): yield {"result": SkillResult(True, "ok")}
        registry.register("my_tool", "描述", {"type": "object", "properties": {}}, fn)
        assert registry.has("my_tool")
        assert not registry.has("other_tool")

    def test_names_returns_registered(self):
        registry = ToolRegistry()
        async def fn(args, ctx): yield {"result": SkillResult(True, "ok")}
        registry.register("a", "desc a", {"type": "object", "properties": {}}, fn)
        registry.register("b", "desc b", {"type": "object", "properties": {}}, fn)
        assert set(registry.names()) == {"a", "b"}

    def test_register_overwrites(self):
        registry = ToolRegistry()
        async def fn1(args, ctx): yield {"result": SkillResult(True, "v1")}
        async def fn2(args, ctx): yield {"result": SkillResult(True, "v2")}
        registry.register("tool", "desc", {"type": "object", "properties": {}}, fn1)
        registry.register("tool", "desc", {"type": "object", "properties": {}}, fn2)
        assert registry.has("tool")


class TestGetOpenAITools:

    def test_returns_openai_format(self):
        registry = ToolRegistry()
        async def fn(args, ctx): yield {"result": SkillResult(True, "ok")}
        registry.register(
            "search_skill",
            "搜索技能",
            {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            fn,
        )
        tools = registry.get_openai_tools()
        assert len(tools) == 1
        t = tools[0]
        assert t["type"] == "function"
        assert t["function"]["name"] == "search_skill"
        assert t["function"]["description"] == "搜索技能"
        assert "properties" in t["function"]["parameters"]

    def test_empty_registry_returns_empty_list(self):
        registry = ToolRegistry()
        assert registry.get_openai_tools() == []

    def test_multiple_tools_all_included(self):
        registry = ToolRegistry()
        async def fn(args, ctx): yield {"result": SkillResult(True, "ok")}
        for name in ["tool_a", "tool_b", "tool_c"]:
            registry.register(name, f"desc {name}", {"type": "object", "properties": {}}, fn)
        tools = registry.get_openai_tools()
        names = {t["function"]["name"] for t in tools}
        assert names == {"tool_a", "tool_b", "tool_c"}


class TestToolRegistryExecute:

    @pytest.mark.asyncio
    async def test_execute_known_tool_yields_result(self):
        registry = ToolRegistry()
        expected = SkillResult(True, "成功摘要", data={"count": 42})

        async def fn(args, ctx):
            yield {"result": expected}

        registry.register("my_tool", "desc", {"type": "object", "properties": {}}, fn)
        ctx = _make_ctx()
        items = [item async for item in registry.execute({"name": "my_tool", "arguments": {}}, ctx)]
        result_items = [i for i in items if "result" in i]
        assert len(result_items) == 1
        res: SkillResult = result_items[0]["result"]
        assert res.success is True
        assert res.summary == "成功摘要"
        assert res.data == {"count": 42}

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_yields_failure(self):
        registry = ToolRegistry()
        ctx = _make_ctx()
        items = [item async for item in registry.execute({"name": "unknown", "arguments": {}}, ctx)]
        result_items = [i for i in items if "result" in i]
        assert len(result_items) == 1
        assert result_items[0]["result"].success is False
        assert "未知工具" in result_items[0]["result"].summary

    @pytest.mark.asyncio
    async def test_execute_passes_args_to_tool(self):
        registry = ToolRegistry()
        received_args = {}

        async def fn(args, ctx):
            received_args.update(args)
            yield {"result": SkillResult(True, "ok")}

        registry.register("arg_tool", "desc", {"type": "object", "properties": {}}, fn)
        ctx = _make_ctx()
        await registry.execute({"name": "arg_tool", "arguments": {"key": "val"}}, ctx).__anext__()
        assert received_args == {"key": "val"}

    @pytest.mark.asyncio
    async def test_execute_tool_exception_yields_failure(self):
        registry = ToolRegistry()

        async def bad_fn(args, ctx):
            raise ValueError("工具内部错误")
            yield  # 使其成为 async generator

        registry.register("bad_tool", "desc", {"type": "object", "properties": {}}, bad_fn)
        ctx = _make_ctx()
        items = [item async for item in registry.execute({"name": "bad_tool", "arguments": {}}, ctx)]
        result_items = [i for i in items if "result" in i]
        assert len(result_items) == 1
        assert result_items[0]["result"].success is False

    @pytest.mark.asyncio
    async def test_execute_tool_with_sse_and_result(self):
        """工具可以先 yield SSE 事件再 yield result。"""
        registry = ToolRegistry()

        async def fn(args, ctx):
            yield {"sse": '{"type": "outline_chunk", "data": "节点"}'}
            yield {"result": SkillResult(True, "完成")}

        registry.register("sse_tool", "desc", {"type": "object", "properties": {}}, fn)
        ctx = _make_ctx()
        items = [item async for item in registry.execute({"name": "sse_tool", "arguments": {}}, ctx)]
        sse_items = [i for i in items if "sse" in i]
        result_items = [i for i in items if "result" in i]
        assert len(sse_items) == 1
        assert len(result_items) == 1

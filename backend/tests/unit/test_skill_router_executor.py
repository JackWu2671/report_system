"""TemplateRouterExecutor 单元测试。

验证：
  1. _llm_select_templates 正确解析新格式（带 description 的 dict 列表）
  2. _llm_select_templates 兼容旧格式（纯字符串 template_id 列表）
  3. execute() 生成的 candidate 包含 description 字段
  4. 单候选和多候选场景下 description 均正确透传
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.context import SkillContext, SkillResult
from tools.executors.template_router_executor import TemplateRouterExecutor


# ─── 辅助 ──────────────────────────────────────────────────────────────────────

def _make_executor(llm_service=None) -> TemplateRouterExecutor:
    return TemplateRouterExecutor(
        llm_service=llm_service or MagicMock(),
        session_service=MagicMock(),
    )


SAMPLE_META_A = {
    "template_id": "fgotn-deploy-a",
    "template_dir": "templates/fgotn-deploy-a",
    "display_name": "细颗粒板卡部署分析",
    "scene_intro": "分析细颗粒板卡部署，结合商业与网络洞察",
    "keywords": ["fgOTN", "细颗粒", "时延"],
    "query_variants": ["fgOTN部署"],
}

SAMPLE_META_B = {
    "template_id": "fgotn-deploy-b",
    "template_dir": "templates/fgotn-deploy-b",
    "display_name": "OTN站点价值覆盖评估",
    "scene_intro": "基于价值企业覆盖评估OTN站点",
    "keywords": ["fgOTN", "价值覆盖", "站点排序"],
    "query_variants": ["OTN站点评估"],
}


# ─── _llm_select_templates 新格式解析 ─────────────────────────────────────────

class TestLLMSelectTemplatesNewFormat:

    @pytest.mark.asyncio
    async def test_parse_new_format_with_description(self):
        """LLM 返回新格式 [{"template_id": ..., "description": ...}] 时正确解析。"""
        llm_response = {
            "matches": [
                {"template_id": "fgotn-deploy-a", "description": "侧重商业与网络时延综合评估"},
                {"template_id": "fgotn-deploy-b", "description": "侧重价值企业覆盖与站点排序"},
            ]
        }

        executor = _make_executor()
        with patch("llm.agent_llm.AgentLLM") as MockAgent:
            instance = MockAgent.return_value
            instance.chat_json = AsyncMock(return_value=llm_response)

            result = await executor._llm_select_templates("fgOTN部署", [SAMPLE_META_A, SAMPLE_META_B])

        assert len(result) == 2
        assert result[0] == {"template_id": "fgotn-deploy-a", "description": "侧重商业与网络时延综合评估"}
        assert result[1] == {"template_id": "fgotn-deploy-b", "description": "侧重价值企业覆盖与站点排序"}

    @pytest.mark.asyncio
    async def test_parse_new_format_missing_description(self):
        """LLM 返回新格式但 description 缺失时，默认为空字符串。"""
        llm_response = {
            "matches": [
                {"template_id": "fgotn-deploy-a"},
            ]
        }

        executor = _make_executor()
        with patch("llm.agent_llm.AgentLLM") as MockAgent:
            instance = MockAgent.return_value
            instance.chat_json = AsyncMock(return_value=llm_response)

            result = await executor._llm_select_templates("fgOTN部署", [SAMPLE_META_A])

        assert len(result) == 1
        assert result[0] == {"template_id": "fgotn-deploy-a", "description": ""}


# ─── _llm_select_templates 旧格式兼容 ─────────────────────────────────────────

class TestLLMSelectTemplatesOldFormat:

    @pytest.mark.asyncio
    async def test_compat_old_string_format(self):
        """LLM 返回旧格式 ["template_id1", "template_id2"] 时自动转为新格式。"""
        llm_response = {
            "matches": ["fgotn-deploy-a", "fgotn-deploy-b"]
        }

        executor = _make_executor()
        with patch("llm.agent_llm.AgentLLM") as MockAgent:
            instance = MockAgent.return_value
            instance.chat_json = AsyncMock(return_value=llm_response)

            result = await executor._llm_select_templates("fgOTN部署", [SAMPLE_META_A, SAMPLE_META_B])

        assert len(result) == 2
        assert result[0] == {"template_id": "fgotn-deploy-a", "description": ""}
        assert result[1] == {"template_id": "fgotn-deploy-b", "description": ""}

    @pytest.mark.asyncio
    async def test_compat_empty_matches(self):
        """LLM 返回空匹配时返回空列表。"""
        llm_response = {"matches": []}

        executor = _make_executor()
        with patch("llm.agent_llm.AgentLLM") as MockAgent:
            instance = MockAgent.return_value
            instance.chat_json = AsyncMock(return_value=llm_response)

            result = await executor._llm_select_templates("无关查询", [SAMPLE_META_A])

        assert result == []


# ─── execute() 候选列表包含 description ──────────────────────────────────────

class TestExecuteCandidateDescription:

    @pytest.mark.asyncio
    async def test_candidates_include_description(self):
        """execute() 生成的 skill_candidates 事件中每个候选包含 description 和 skill_id 字段。"""
        executor = _make_executor()
        executor._ss = MagicMock()
        executor._ss.set_skill_candidates = AsyncMock()

        executor._llm_select_templates = AsyncMock(return_value=[
            {"template_id": "fgotn-deploy-a", "description": "侧重商业评估"},
            {"template_id": "fgotn-deploy-b", "description": "侧重站点排序"},
        ])

        with patch("os.path.isdir", return_value=True), \
             patch("os.listdir", return_value=["fgotn-deploy-a", "fgotn-deploy-b"]), \
             patch.object(TemplateRouterExecutor, "_load_template_meta",
                          side_effect=[SAMPLE_META_A, SAMPLE_META_B]):

            ctx = SkillContext(
                session_id="test-session",
                user_message="fgOTN部署机会点",
                params={"query": "fgOTN部署机会点"},
            )

            events = []
            async for item in executor.execute(ctx):
                if isinstance(item, SkillResult):
                    events.append(("result", item))
                else:
                    events.append(("sse", item))

        cand_events = []
        for tag, val in events:
            if tag == "sse":
                try:
                    parsed = json.loads(val)
                    if parsed.get("type") == "skill_candidates":
                        cand_events.append(parsed)
                except (json.JSONDecodeError, TypeError):
                    pass

        assert len(cand_events) == 1, "应有一个 skill_candidates SSE 事件"
        candidates = cand_events[0]["candidates"]
        assert len(candidates) == 2

        assert candidates[0]["label"] == "A"
        assert candidates[0]["skill_id"] == "fgotn-deploy-a"
        assert candidates[0]["template_id"] == "fgotn-deploy-a"
        assert candidates[0]["description"] == "侧重商业评估"

        assert candidates[1]["label"] == "B"
        assert candidates[1]["skill_id"] == "fgotn-deploy-b"
        assert candidates[1]["description"] == "侧重站点排序"

    @pytest.mark.asyncio
    async def test_candidate_description_empty_when_old_format(self):
        """当 _llm_select_templates 返回 description 为空时，candidate 的 description 为空字符串。"""
        executor = _make_executor()
        executor._ss = MagicMock()
        executor._ss.set_skill_candidates = AsyncMock()

        executor._llm_select_templates = AsyncMock(return_value=[
            {"template_id": "fgotn-deploy-a", "description": ""},
        ])

        with patch("os.path.isdir", return_value=True), \
             patch("os.listdir", return_value=["fgotn-deploy-a"]), \
             patch.object(TemplateRouterExecutor, "_load_template_meta",
                          side_effect=[SAMPLE_META_A]):

            ctx = SkillContext(
                session_id="test-session",
                user_message="fgOTN",
                params={"query": "fgOTN"},
            )

            events = []
            async for item in executor.execute(ctx):
                if isinstance(item, SkillResult):
                    events.append(("result", item))
                else:
                    events.append(("sse", item))

        cand_events = [json.loads(v) for _, v in events
                       if _ == "sse" and "skill_candidates" in v]
        assert len(cand_events) == 1
        assert cand_events[0]["candidates"][0]["description"] == ""

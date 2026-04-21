"""logic-persist 元技能执行器。

纯编排器——SubSkill 独立模块实现，executor 只负责执行编排。
两种触发路径：
  - 默认：专家输入 → 四步设计流程 → 缓存 + 提示保存
  - persist_current：从当前会话大纲直接沉淀
"""
import json, logging, os, sys
from typing import AsyncGenerator, Union

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from agent.context import SkillContext, SkillResult
from context import SkillFactoryContext, ServiceBundle, sse_event, design_step
from sub_skills import IntentUnderstand, StructExtract, OutlineDesign, DataBinding
from sub_skills.data_binding import _collect_l5_bindings
from sub_skills.outline_design import outline_to_md

logger = logging.getLogger(__name__)


class SkillFactoryExecutor:
    """logic-persist 执行器——纯编排。"""

    def __init__(self, llm_service, embedding_service, faiss_retriever,
                 neo4j_retriever, session_service, kb_store,
                 indicator_resolver=None):
        self._svc = ServiceBundle(llm_service, embedding_service, faiss_retriever,
                                   neo4j_retriever, session_service, kb_store,
                                   indicator_resolver)

    async def execute(self, ctx: SkillContext) -> AsyncGenerator[Union[str, SkillResult], None]:
        expert_input = ctx.params.get("expert_input", ctx.user_message)
        context_key = ctx.params.get("context_key", "")

        # 从缓存恢复并沉淀（用户确认"保存"后触发）
        if context_key:
            async for event in self._persist_from_cache(ctx, context_key):
                yield event
            return

        # 从当前会话大纲直接沉淀
        if ctx.params.get("mode") == "persist_current":
            async for event in self._persist_current(ctx, expert_input):
                yield event
            return

        # 四步设计流程
        fc = SkillFactoryContext(raw_input=expert_input)
        for StepClass in [IntentUnderstand, StructExtract, OutlineDesign, DataBinding]:
            step = StepClass(self._svc)
            async for event in step.run(fc, ctx):
                yield event

        # 缓存上下文 + 提示用户确认保存
        cache_key = ctx.session_id
        await self._svc.session.redis.setex(
            f"logic_persist_ctx:{cache_key}", 3600,
            json.dumps(fc.to_cache_dict(), ensure_ascii=False)
        )
        yield sse_event("persist_prompt", {
            "message": "大纲模板已设计完成，是否保存为可复用模板？请回复「保存」或「不保存」。",
            "context_key": cache_key,
        })
        yield SkillResult(True, "大纲模板已设计完成",
                          data={"context_key": cache_key,
                                "outline_md": fc.outline_md,
                                "outline_json": fc.outline_json,
                                "template_name": fc.template_name})

    async def _persist_from_cache(self, ctx, context_key):
        cached = await self._svc.session.redis.get(f"logic_persist_ctx:{context_key}")
        if not cached:
            yield SkillResult(False, "缓存已过期，请重新执行设计流程")
            return
        fc = SkillFactoryContext.from_cache_dict(json.loads(cached))
        from sub_skills.skill_persist import SkillPersist
        persist = SkillPersist(self._svc)
        async for event in persist.run(fc, ctx):
            yield event
        yield SkillResult(True, f"大纲模板「{fc.template_name}」已沉淀",
                          data={"template_dir": fc.template_dir,
                                "template_name": fc.template_name})

    async def _persist_current(self, ctx, expert_input):
        if not ctx.current_outline:
            yield SkillResult(False, "当前没有大纲，无法保存")
            return

        yield design_step("skill_persist", "running")
        fc = SkillFactoryContext(raw_input=expert_input)
        fc.outline_json = ctx.current_outline
        fc.outline_md = outline_to_md(ctx.current_outline)

        from llm.agent_llm import AgentLLM
        from llm.config import SKILL_FACTORY_JSON_CONFIG
        try:
            agent = AgentLLM(self._svc.llm, "", SKILL_FACTORY_JSON_CONFIG,
                             trace_callback=ctx.trace_callback, llm_type="logic_persist", step_name="quick_extract")
            r = await agent.chat_json(
                f"从以下大纲提取：template_name(英文下划线), scene_intro(50字), keywords(3-5个), query_variants(3种问法)。\n\n"
                f"大纲：\n{fc.outline_md[:500]}\n\n用 ```json ``` 代码块包裹输出。"
            )
            fc.template_name = r.get("template_name", "unnamed_template")
            fc.scene_intro = r.get("scene_intro", "")
            fc.keywords = r.get("keywords", [])
            fc.query_variants = r.get("query_variants", [])[:5]
        except Exception as e:
            logger.warning(f"快速提取失败: {e}")
            fc.template_name = "unnamed_template"

        bindings = []
        _collect_l5_bindings(fc.outline_json, bindings)
        fc.bindings = bindings

        from sub_skills.skill_persist import SkillPersist
        persist = SkillPersist(self._svc)
        async for event in persist.run(fc, ctx):
            yield event
        yield SkillResult(True, f"大纲模板「{fc.template_name}」已沉淀",
                          data={"template_dir": fc.template_dir,
                                "template_name": fc.template_name,
                                "outline_json": fc.outline_json})

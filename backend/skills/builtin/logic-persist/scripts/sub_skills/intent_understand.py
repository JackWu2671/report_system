"""Sub-Step 1：意图理解与泛化。"""
from typing import AsyncGenerator
from llm.agent_llm import AgentLLM
from llm.config import SKILL_FACTORY_JSON_CONFIG
from sub_skills.base import SubSkillBase
from context import SkillFactoryContext
from agent.context import SkillContext
import logging

logger = logging.getLogger(__name__)


class IntentUnderstand(SubSkillBase):
    name = "intent_understand"

    async def execute(self, fc: SkillFactoryContext, ctx: SkillContext) -> AsyncGenerator[str, None]:
        prompt = f"""你是看网逻辑分析专家。分析看网逻辑文本，提取结构化信息。

## 输出格式
```json
{{"scene_intro":"50字以内","keywords":["3-5个关键词"],"query_variants":["3种用户问法"],"skill_name":"英文下划线"}}
```

## 示例
输入: "从传送网络容量角度分析fgOTN部署机会"
```json
{{"scene_intro":"分析fgOTN传送网络容量，评估部署机会","keywords":["fgOTN","传送网络","容量分析","部署"],"query_variants":["帮我分析fgOTN网络容量","fgOTN部署机会分析","传送网络容量评估"],"skill_name":"fgOTN_Capacity_Analysis"}}
```

## 看网逻辑
{fc.raw_input}

用 ```json ``` 代码块包裹输出。"""

        agent = AgentLLM(self._svc.llm, "", SKILL_FACTORY_JSON_CONFIG,
                         trace_callback=ctx.trace_callback, llm_type="skill_factory", step_name="intent_understand")
        try:
            result = await agent.chat_json(prompt)
            fc.scene_intro = result.get("scene_intro", "")
            fc.keywords = result.get("keywords", [])
            fc.query_variants = result.get("query_variants", [])[:5]
            fc.skill_name = result.get("skill_name", "unnamed_skill")
        except Exception as e:
            logger.warning(f"Step 1 LLM 失败: {e}")
            fc.skill_name = "unnamed_skill"
        # 不需要 yield 任何事件（base.run 已处理 running/done）
        return
        yield  # make it a generator

"""TemplateRouterExecutor —— 大纲模板路由。

扫描已沉淀的大纲模板（templates/ 目录），通过 LLM 精排返回候选列表，
推送 skill_candidates SSE 事件并保存 Redis 供 LeadAgent 拦截。
"""
import json
import logging
import os
from typing import AsyncGenerator, Union

from agent.context import SkillContext, SkillResult
from config import settings

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """\
你是看网模板路由专家。根据用户的分析需求，从已沉淀的大纲模板列表中找出最相关的候选项，并为每个候选生成简短的差异化描述帮助用户选择。

## 输出格式
用 ```json ``` 代码块包裹，格式：
```json
{"matches": [{"template_id": "id1", "description": "该模板侧重于..."}, {"template_id": "id2", "description": "该模板侧重于..."}]}
```
- matches 为匹配的候选列表，最多 5 个，按相关度从高到低排列
- 每个候选包含 template_id 和 description 两个字段
- description 规则：
  - 多个候选时：突出该模板与其他候选的差异点（如分析维度、侧重场景的不同），30-50字
  - 仅一个候选时：简要说明该模板适合的场景，30字以内
- 若无匹配则返回空列表：{"matches": []}
- 只输出 JSON，不要加解释文字
"""


class TemplateRouterExecutor:

    def __init__(self, llm_service, session_service):
        self._llm = llm_service
        self._ss = session_service

    async def execute(self, ctx: SkillContext) -> AsyncGenerator[Union[str, SkillResult], None]:
        query = ctx.params.get("query", ctx.user_message)

        if not getattr(settings, "SKILL_ROUTER_ENABLED", True):
            yield SkillResult(True, "template_router 已禁用", data={"candidates": []})
            return

        # Step 1: 扫描 templates/ 目录
        yield json.dumps({"type": "thinking_step", "step": "template_router",
                          "status": "running", "detail": "正在扫描已沉淀的大纲模板..."}, ensure_ascii=False)

        templates_dir = "templates"
        template_metas = []
        if os.path.isdir(templates_dir):
            for entry in sorted(os.listdir(templates_dir)):
                tdir = os.path.join(templates_dir, entry)
                if not os.path.isdir(tdir):
                    continue
                meta = self._load_template_meta(tdir)
                if meta:
                    template_metas.append(meta)

        if not template_metas:
            yield json.dumps({"type": "thinking_step", "step": "template_router",
                              "status": "done", "detail": "暂无已沉淀的大纲模板"}, ensure_ascii=False)
            yield SkillResult(True, "无已沉淀模板", data={"candidates": []})
            return

        yield json.dumps({"type": "thinking_step", "step": "template_router",
                          "status": "running",
                          "detail": f"找到 {len(template_metas)} 个模板，正在 LLM 精排..."}, ensure_ascii=False)

        # Step 2: LLM 精排
        matched = await self._llm_select_templates(
            query, template_metas, trace_callback=ctx.trace_callback
        )

        # Step 3: 构建候选列表
        meta_map = {m["template_id"]: m for m in template_metas}
        labels = ["A", "B", "C", "D", "E"]
        candidates = []
        for i, match_item in enumerate(matched[:5]):
            tid = match_item.get("template_id", "") if isinstance(match_item, dict) else match_item
            m = meta_map.get(tid)
            if not m:
                continue
            desc = match_item.get("description", "") if isinstance(match_item, dict) else ""
            candidates.append({
                "label": labels[i],
                "skill_id": m["template_id"],      # 保持 skill_id 字段名供 LeadAgent 兼容
                "template_id": m["template_id"],
                "template_dir": m["template_dir"],
                "display_name": m.get("display_name", m["template_id"]),
                "scene_intro": m.get("scene_intro", ""),
                "keywords": m.get("keywords", []),
                "description": desc,
            })

        yield json.dumps({"type": "thinking_step", "step": "template_router",
                          "status": "done",
                          "detail": f"精排完成，{len(candidates)} 个候选"}, ensure_ascii=False)

        # Step 4: 推送 SSE + 保存 Redis
        if candidates:
            await self._ss.set_skill_candidates(ctx.session_id, {
                "candidates": candidates,
                "query": query,
            })
            yield json.dumps({
                "type": "skill_candidates",
                "candidates": candidates,
                "query": query,
            }, ensure_ascii=False)

        yield SkillResult(True, f"路由完成，{len(candidates)} 个候选",
                          data={"candidates": candidates, "query": query})

    async def _llm_select_templates(self, query: str, template_metas: list,
                                     trace_callback=None) -> list:
        from llm.config import SKILL_ROUTER_CONFIG
        from llm.agent_llm import AgentLLM

        lines = []
        for m in template_metas:
            kw_str = "、".join(m.get("keywords", [])[:5])
            qv_str = "、".join(m.get("query_variants", [])[:3])
            lines.append(
                f'template_id={m["template_id"]}\n'
                f'  名称: {m.get("display_name", m["template_id"])}\n'
                f'  场景: {m.get("scene_intro", "")}\n'
                f'  关键词: {kw_str}\n'
                f'  触发问法示例: {qv_str}'
            )

        template_list_text = "\n\n".join(lines)
        user_msg = f"## 用户需求\n{query}\n\n## 已沉淀大纲模板列表\n{template_list_text}"

        try:
            agent = AgentLLM(
                self._llm,
                system_prompt=ROUTER_SYSTEM_PROMPT,
                config=SKILL_ROUTER_CONFIG,
                trace_callback=trace_callback,
                llm_type="template_router",
                step_name="template_router_select",
            )
            data = await agent.chat_json(user_msg)
            raw_matches = data.get("matches", [])
            result = []
            for item in raw_matches:
                if isinstance(item, str):
                    result.append({"template_id": item, "description": ""})
                elif isinstance(item, dict) and "template_id" in item:
                    result.append({"template_id": item["template_id"],
                                   "description": item.get("description", "")})
            return result
        except Exception as e:
            logger.warning(f"template_router LLM 精排失败，返回空候选: {e}")
            return []

    @staticmethod
    def _load_template_meta(template_dir: str) -> dict:
        """从 outline_template.json 读取模板元数据。"""
        json_path = os.path.join(template_dir, "outline_template.json")
        if not os.path.isfile(json_path):
            return {}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            meta = data.get("meta", {})
            template_name = meta.get("template_name", os.path.basename(template_dir))

            # query_variants 优先读 txt 文件，fallback 到 meta 字段
            qv_path = os.path.join(template_dir, "query_variants.txt")
            query_variants = meta.get("query_variants", [])
            if os.path.isfile(qv_path):
                with open(qv_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if lines:
                    query_variants = lines

            return {
                "template_id": template_name,
                "template_dir": template_dir,
                "display_name": meta.get("display_name", template_name),
                "scene_intro": meta.get("scene_intro", ""),
                "keywords": meta.get("keywords", []),
                "query_variants": query_variants,
            }
        except Exception as e:
            logger.warning(f"加载模板元数据失败: {template_dir}, {e}")
            return {}

"""Sub-Step 5：大纲模板沉淀——写文件系统 + Neo4j。"""
import json, logging, os
from typing import AsyncGenerator
from sub_skills.base import SubSkillBase
from context import SkillFactoryContext, sse_event
from agent.context import SkillContext

logger = logging.getLogger(__name__)


class SkillPersist(SubSkillBase):
    name = "skill_persist"

    async def execute(self, fc: SkillFactoryContext, ctx: SkillContext) -> AsyncGenerator[str, None]:
        base_dir = f"templates/{fc.template_name}"
        template_dir, version, is_new_version = _resolve_versioned_dir(base_dir)
        os.makedirs(template_dir, exist_ok=True)

        original_name = fc.template_name
        if is_new_version:
            fc.template_name = f"{original_name}_v{version}"
            fc.version = version
            logger.info(f"SkillPersist: 检测到重名模板 '{original_name}'，创建新版本 '{fc.template_name}' → {template_dir}")

        # outline_template.json：meta + outline 合并为一个文件
        outline_template = {
            "meta": {
                "template_name": fc.template_name,
                "display_name": fc.scene_intro or fc.template_name,
                "scene_intro": fc.scene_intro,
                "keywords": fc.keywords,
                "query_variants": fc.query_variants,
                "source": fc.source,
                "version": fc.version,
            },
            "outline": fc.outline_json or {},
        }
        with open(f"{template_dir}/outline_template.json", "w", encoding="utf-8") as f:
            json.dump(outline_template, f, ensure_ascii=False, indent=2)

        with open(f"{template_dir}/query_variants.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(fc.query_variants))

        fc.template_dir = template_dir

        # Neo4j：挂载 template_path
        if fc.anchor_node_id:
            await self._svc.neo4j.set_skill_path(fc.anchor_node_id, template_dir)
        if fc.new_nodes:
            await self._svc.neo4j.create_nodes_and_relations(fc.new_nodes)

        yield sse_event("skill_persisted", {
            "template_name": fc.template_name,
            "template_dir": template_dir,
            "version": fc.version,
            "is_new_version": is_new_version,
            "original_name": original_name,
        })


def _resolve_versioned_dir(base_dir: str) -> tuple:
    if not os.path.exists(base_dir):
        return base_dir, 1, False
    v = 2
    while os.path.exists(f"{base_dir}_v{v}"):
        v += 1
    return f"{base_dir}_v{v}", v, True

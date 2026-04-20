"""Sub-Step 2：知识库节点匹配 + 后验校验。

LLM 只输出匹配到的节点名列表，unmatched 由后处理计算。
"""
import json, logging
from typing import AsyncGenerator
from llm.agent_llm import AgentLLM
from llm.config import SKILL_FACTORY_JSON_CONFIG
from sub_skills.base import SubSkillBase
from context import SkillFactoryContext, sse_event
from agent.context import SkillContext

logger = logging.getLogger(__name__)


class StructExtract(SubSkillBase):
    name = "struct_extract"

    async def execute(self, fc: SkillFactoryContext, ctx: SkillContext) -> AsyncGenerator[str, None]:
        # 1. FAISS 检索 + Neo4j 子树获取
        query = (fc.scene_intro + " " + " ".join(fc.keywords)).strip() or fc.raw_input[:200]
        qe = await self._svc.embedding.get_embedding(query)
        cands = self._svc.faiss.search(qe, top_k=20, threshold=0.3)

        kb_nodes = []
        if cands:
            ancestor_paths = await self._svc.neo4j.get_ancestor_paths([c.neo4j_id for c in cands])
            subtree_roots = set()
            for n in (ancestor_paths or []):
                if n["level"] in (2, 3):
                    subtree_roots.add(n["id"])
            all_nodes = {}
            for root_id in subtree_roots:
                subtree = await self._svc.neo4j.get_subtree(root_id)
                if subtree:
                    _flatten_tree(subtree, all_nodes)
            if not all_nodes and ancestor_paths:
                for n in ancestor_paths:
                    all_nodes[n["id"]] = n
            kb_nodes = sorted(all_nodes.values(), key=lambda x: (x.get("level", 0), x.get("name", "")))

        fc.kb_nodes = kb_nodes

        kb_node_list = "\n".join(
            f"- {n['name']} (L{n['level']})" for n in kb_nodes
        ) if kb_nodes else "无匹配节点"

        # 2. LLM 只输出匹配到的节点名列表
        prompt = f"""你是知识库映射专家。从知识库节点清单中，选出与用户看网逻辑相关的节点。

## 知识库节点清单
{kb_node_list}

## 用户看网逻辑
{fc.raw_input}

## 选取规则（分层意图匹配）
1. **优先选 L2（子场景）**：判断用户意图属于哪个子场景，选出对应 L2 节点——这是最重要的锚点
2. **同时选相关 L3（评估维度）**：在已确定子场景内，用户关注了哪些具体维度，选出对应 L3 节点
3. **不选 L4/L5**：粒度过细，由后续步骤处理
4. 只从清单中选取，不虚构节点名

用 ```json ``` 代码块包裹输出，格式：
```json
{{"matched_names": ["节点名1", "节点名2"]}}
```"""

        agent = AgentLLM(self._svc.llm, "", SKILL_FACTORY_JSON_CONFIG,
                         trace_callback=ctx.trace_callback, llm_type="skill_factory", step_name="struct_extract")

        matched_names = []
        try:
            result = await agent.chat_json(prompt)
            matched_names = result.get("matched_names", [])
        except Exception as e:
            logger.warning(f"Step 2 LLM 失败: {e}")

        # 3. 后处理：用 kb_nodes 全集校验 + 构建 dimension_hints + 计算 unmatched
        kb_name_map = {n["name"].strip(): n for n in kb_nodes}
        validated_hints, phantom = [], []

        for name in matched_names:
            name = name.strip()
            if name in kb_name_map:
                node = kb_name_map[name]
                validated_hints.append({"id": node["id"], "name": name, "level": node["level"]})
            else:
                phantom.append(name)

        if phantom:
            logger.warning(f"Step 2 后验校验：LLM 输出了知识库外节点，已移除: {phantom}")
            yield sse_event("kb_validation", {
                "removed": phantom,
                "message": f"以下节点不在知识库中，已自动过滤：{', '.join(phantom)}"
            })

        fc.dimension_hints = validated_hints

        # 计算 unmatched（用户逻辑涉及但知识库中不存在的概念）
        # 由 LLM 匹配到的都在 kb 里了，未匹配的就是 kb 全集之外的——这里不再需要 LLM 输出
        logger.info(f"Step 2: 匹配 {len(validated_hints)} 个节点, 移除幽灵 {len(phantom)} 个")

        return
        yield  # make it a generator


def _flatten_tree(node: dict, result: dict):
    """将树形结构展平为 {id: node_info} 字典。"""
    if not node:
        return
    nid = node.get("id", node.get("name", ""))
    if nid and nid not in result:
        result[nid] = {"id": node.get("id", ""), "name": node.get("name", ""), "level": node.get("level", 0)}
    for child in node.get("children", []):
        _flatten_tree(child, result)

"""GraphRAG 大纲生成执行器。"""
import json, logging
from typing import AsyncGenerator, Union
from agent.context import SkillContext, SkillResult
from llm.agent_llm import AgentLLM
from llm.config import ANCHOR_SELECT_CONFIG
from llm.service import LLMService
from pipeline.faiss_retriever import FAISSRetriever
from pipeline.neo4j_retriever import Neo4jRetriever
from pipeline.outline_renderer import OutlineRenderer
from services.embedding_service import EmbeddingService
from services.session_service import SessionService
from utils.trace_logger import TraceLogger

logger = logging.getLogger(__name__)

ANCHOR_PROMPT = """你是知识库节点选择专家。从候选中选出最符合用户意图的唯一节点。
判断原则: 宽泛→高层级，具体→低层级，父子关系时按粒度判断。
用 ```json ``` 代码块包裹输出，不要加解释文字。格式:
```json
{"selected_id":"","selected_name":"","selected_path":"","level":0,"reason":""}
```"""

def _ts(step, status, detail, data=None):
    p = {"type":"thinking_step","step":step,"status":status,"detail":detail}
    if data: p["data"] = data
    return json.dumps(p, ensure_ascii=False)


class GraphRAGExecutor:
    def __init__(self, llm_service, embedding_service, faiss_retriever, neo4j_retriever,
                 outline_renderer, session_service, indicator_resolver=None,
                 top_k=10, score_threshold=0.5):
        self._llm = llm_service; self._emb = embedding_service; self._faiss = faiss_retriever
        self._neo4j = neo4j_retriever; self._render = outline_renderer; self._session = session_service
        self._indicator_resolver = indicator_resolver
        self._top_k = top_k; self._threshold = score_threshold

    async def execute(self, ctx: SkillContext) -> AsyncGenerator[Union[str, SkillResult], None]:
        query = ctx.params.get("query", ctx.user_message)
        sid = ctx.session_id; trace = TraceLogger(session_id=sid).child("skill.outline_generate")

        # Step 1: Embedding
        yield _ts("embedding","running","正在转换语义向量..."); trace.start_timer("s1")
        qe = await self._emb.get_embedding(query)
        trace.log_timed("s1","s1")
        yield _ts("embedding","done","语义向量完成")

        # Step 2: FAISS
        yield _ts("knowledge_search","running","正在检索知识库..."); trace.start_timer("s2")
        cands = self._faiss.search(qe, self._top_k, self._threshold); trace.log_timed("s2","s2")
        if not cands:
            yield _ts("knowledge_search","done","未找到相关知识")
            yield SkillResult(False, f"未找到与「{query}」相关的知识"); return
        yield _ts("knowledge_search","done",f"找到 {len(cands)} 个节点",
                   data={"top_matches":[{"name":c.name,"score":f"{c.score:.2f}"} for c in cands[:5]]})

        # Step 3: Neo4j 祖先路径
        yield _ts("path_analysis","running","正在分析知识路径..."); trace.start_timer("s3")
        nodes = await self._neo4j.get_ancestor_paths([c.neo4j_id for c in cands]); trace.log_timed("s3","s3")
        if not nodes:
            yield _ts("path_analysis","done","路径异常"); yield SkillResult(False,"路径查询异常"); return
        yield _ts("path_analysis","done",f"{len(nodes)} 条路径")

        # Step 4: LLM 选锚
        yield _ts("anchor_select","running","正在选择起始节点..."); trace.start_timer("s4")
        anchor = await self._select_anchor(query, nodes, trace_callback=ctx.trace_callback); trace.log_timed("s4","s4")
        yield _ts("anchor_select","done",f'选择「{anchor["selected_name"]}」(L{anchor["level"]})',
                   data={"reason":anchor.get("reason","")})

        # Step 5: L5 层级判断（简单按层级触发确认）
        if anchor["level"] == 5:
            yield _ts("level_check","done","叶子节点，需确认层级")
            ancs = await self._neo4j.get_ancestor_chain(anchor["selected_id"])
            await self._session.set_pending_confirm(sid, ancs, 300)
            trace.log("s5_l5_confirm", data={"ancestors": ancs})
            yield SkillResult(True, f"指标「{anchor['selected_name']}」需确认层级", need_user_input=True,
                data={"type":"confirm_required","indicator_name":anchor["selected_name"],
                      "full_path":anchor["selected_path"],"ancestors":ancs},
                user_prompt=f'找到「{anchor["selected_name"]}」，请选择起始层级。')
            return

        # Step 6: 子树遍历
        yield _ts("subtree_fetch","running","正在获取子树..."); trace.start_timer("s6")
        subtree = await self._neo4j.get_subtree(anchor["selected_id"]); trace.log_timed("s6","s6")
        if not subtree: yield _ts("subtree_fetch","done","子树为空"); yield SkillResult(False,"子树为空"); return
        yield _ts("subtree_fetch","done","子树获取完成")

        # Step 6.5: 基于查询意图自动裁剪（与 clip_outline 工具相同的结构化操作模式）
        top_sections = [c for c in subtree.get("children", []) if c.get("level", 0) <= 4]
        if len(top_sections) >= 2:
            yield _ts("query_filter", "running", "正在分析查询意图，检查是否需要章节过滤...")
            trace.start_timer("s6_5")
            subtree, was_filtered = await self._clip_by_query(query, subtree, ctx.trace_callback)
            trace.log_timed("s6_5", "s6_5")
            if was_filtered:
                yield _ts("query_filter", "done", f"已按查询意图过滤，保留 {self._count_children(subtree)} 个节点")
            else:
                yield _ts("query_filter", "done", "查询覆盖完整子树，无需过滤")

        # Step 6.8: 合并 paragraph 到 L5 节点
        self.merge_paragraph(subtree, skill_dir="")

        # Step 7: 渲染大纲
        yield _ts("outline_render","running","正在生成大纲..."); trace.start_timer("s7"); chunks=[]
        async for c in self._render.render_stream(subtree, anchor):
            chunks.append(c); yield json.dumps({"type":"outline_chunk","content":c},ensure_ascii=False)
        md = "".join(chunks)
        ai = {"id":anchor["selected_id"],"name":anchor["selected_name"],"level":anchor["level"]}
        yield json.dumps({"type":"outline_done","anchor":ai},ensure_ascii=False)
        trace.log_timed("s7","s7",data={"chunks":len(chunks)})
        yield _ts("outline_render","done",f"大纲完成，{len(chunks)} 章节")
        yield SkillResult(True, f"已生成「{anchor['selected_name']}」的大纲",
                          data={"subtree":subtree,"anchor":ai,"outline_md":md})

    async def execute_from_node(self, ctx: SkillContext, node_id: str) -> AsyncGenerator[Union[str, SkillResult], None]:
        ni = await self._neo4j.get_node_by_id(node_id)
        if not ni: yield SkillResult(False,"节点不存在"); return
        yield _ts("subtree_fetch","running","获取子树...")
        subtree = await self._neo4j.get_subtree(node_id)
        if not subtree: yield _ts("subtree_fetch","done","空"); yield SkillResult(False,"子树为空"); return
        yield _ts("subtree_fetch","done","完成")
        self.merge_paragraph(subtree, skill_dir="")
        anchor = {"selected_id":ni["id"],"selected_name":ni["name"],"level":ni["level"]}
        yield _ts("outline_render","running","生成大纲..."); chunks=[]
        async for c in self._render.render_stream(subtree, anchor):
            chunks.append(c); yield json.dumps({"type":"outline_chunk","content":c},ensure_ascii=False)
        ai = {"id":ni["id"],"name":ni["name"],"level":ni["level"]}
        yield json.dumps({"type":"outline_done","anchor":ai},ensure_ascii=False)
        yield _ts("outline_render","done",f"{len(chunks)} 章节")
        yield SkillResult(True, f"已生成「{ni['name']}」大纲",
                          data={"subtree":subtree,"anchor":ai,"outline_md":"".join(chunks)})

    async def _select_anchor(self, query, nodes, trace_callback=None):
        sa = AgentLLM(self._llm, ANCHOR_PROMPT, ANCHOR_SELECT_CONFIG,
                      trace_callback=trace_callback, llm_type="anchor_select", step_name="anchor_select")
        cs = "\n".join(f'- id={n["id"]} name={n["name"]} level={n["level"]} path={n["path"]}' for n in nodes)
        try: return await sa.chat_json(f"## 候选\n{cs}\n\n## 问题\n{query}")
        except:
            f=nodes[0]; return {"selected_id":f["id"],"selected_name":f["name"],"selected_path":f["path"],"level":f["level"],"reason":"fallback"}

    # ─── Step 6.5 辅助：与 OutlineClipExecutor 共用相同的操作模式 ─────────────

    QUERY_CLIP_PROMPT = """\
你是大纲裁剪专家。根据用户查询判断是否需要裁剪大纲，若需要则生成裁剪操作列表。

## 用户查询
{query}

## 当前大纲节点
{nodes_text}

## 判断规则
- 若查询是泛化需求（如"分析fgOTN部署"、"帮我看传送网"），输出 {{"instructions": []}}，不裁剪
- 若查询含明确章节限定词（如"只要覆盖分析"/"只看容量"/"不看低阶交叉"），生成对应操作
- 不处理参数/阈值类约束（如"只看金融行业"/"阈值改为80%"），这类由 inject_params 工具处理
- 允许语义模糊匹配（"覆盖分析"≈"覆盖率分析"≈"网络覆盖"）

## 输出格式
用 ```json ``` 代码块包裹，格式：
```json
{{"instructions": [
    {{"type": "delete_node", "target_name": "节点名"}},
    {{"type": "keep_only", "target_names": ["节点名1", "节点名2"]}}
]}}
```
无需裁剪时输出：{{"instructions": []}}"""

    async def _clip_by_query(self, query: str, subtree: dict, trace_callback=None) -> tuple[dict, bool]:
        """根据用户查询自动裁剪子树，返回 (subtree, was_filtered)。
        使用与 OutlineClipExecutor 相同的结构化操作格式。
        """
        import copy
        from llm.config import LLMConfig

        nodes_text = self._collect_nodes_text(subtree)
        prompt = self.QUERY_CLIP_PROMPT.format(query=query, nodes_text=nodes_text)

        try:
            agent = AgentLLM(
                self._llm, "",
                LLMConfig(temperature=0.1, max_tokens=512),
                trace_callback=trace_callback,
                llm_type="query_filter",
                step_name="clip_by_query",
            )
            result = await agent.chat_json(prompt)
            instructions = result.get("instructions", [])
        except Exception as e:
            logger.warning(f"clip_by_query LLM 失败，保留完整子树: {e}")
            return subtree, False

        if not instructions:
            return subtree, False

        node = copy.deepcopy(subtree)
        changed = False
        for inst in instructions:
            t = inst.get("type")
            if t == "delete_node":
                target = inst.get("target_name", "")
                if target:
                    node = self._delete_node(node, target)
                    changed = True
            elif t == "keep_only":
                targets = inst.get("target_names", [])
                if targets:
                    node = self._keep_only(node, set(targets))
                    changed = True

        logger.info(f"clip_by_query: {len(instructions)} 条操作，changed={changed}")
        return node, changed

    @staticmethod
    def _collect_nodes_text(node, depth=0) -> str:
        """收集大纲节点文本，供 LLM 理解结构（与 OutlineClipExecutor 相同逻辑）。"""
        lines = []
        name = node.get("name", "")
        level = node.get("level", 0)
        if name and level != 5:
            lines.append(f"{'  ' * depth}- {name} (L{level})")
        if depth < 3:
            for child in node.get("children", []):
                lines.append(GraphRAGExecutor._collect_nodes_text(child, depth + 1))
        return "\n".join(lines)

    @staticmethod
    def _delete_node(node: dict, target_name: str) -> dict:
        if not node.get("children"):
            return node
        node["children"] = [
            GraphRAGExecutor._delete_node(c, target_name)
            for c in node["children"]
            if c.get("name") != target_name
        ]
        return node

    @staticmethod
    def _keep_only(node: dict, target_names: set) -> dict:
        if not node.get("children"):
            return node
        node["children"] = [
            GraphRAGExecutor._keep_only(c, target_names)
            for c in node["children"]
            if c.get("name") in target_names or GraphRAGExecutor._has_descendant(c, target_names)
        ]
        return node

    @staticmethod
    def _has_descendant(node: dict, names: set) -> bool:
        if node.get("name") in names:
            return True
        return any(GraphRAGExecutor._has_descendant(c, names) for c in node.get("children", []))

    @staticmethod
    def _count_children(node: dict) -> int:
        count = 1
        for c in node.get("children", []):
            count += GraphRAGExecutor._count_children(c)
        return count

    def merge_paragraph(self, node: dict, skill_dir: str = "") -> None:
        """遍历大纲所有 L5 节点，从 IndicatorResolver 读取 paragraph 并写入节点。"""
        if not self._indicator_resolver:
            return
        self.merge_paragraph_node(node, skill_dir)

    def merge_paragraph_node(self, node: dict, skill_dir: str) -> None:
        if node.get("level") == 5:
            if "paragraph" not in node:
                node["paragraph"] = self._indicator_resolver.resolve(
                    node_id=node.get("id", ""),
                    node_name=node.get("name", ""),
                    skill_dir=skill_dir,
                )
        for child in node.get("children", []):
            self.merge_paragraph_node(child, skill_dir)

    @staticmethod
    def load_skill_outline(skill_dir: str):
        """从文件系统加载已沉淀 Skill 的大纲。返回 (outline_json, outline_md) 或 None。"""
        import os
        outline_path = os.path.join(skill_dir, "references", "outline.json")
        logger.info(f"尝试加载 Skill 大纲: {outline_path} (exists={os.path.isfile(outline_path)})")
        if not os.path.isfile(outline_path):
            logger.warning(f"Skill 大纲文件不存在: {outline_path}")
            return None
        try:
            with open(outline_path, "r", encoding="utf-8") as f:
                outline_json = json.load(f)
            # 验证大纲不为空
            if not outline_json or not outline_json.get("children"):
                logger.warning(f"Skill 大纲为空: {outline_path}")
                return None
            md = GraphRAGExecutor._outline_json_to_md(outline_json)
            logger.info(f"Skill 大纲加载成功: {outline_path}, {len(outline_json.get('children',[]))} 个子节点")
            return outline_json, md
        except Exception as e:
            logger.warning(f"加载 Skill 大纲失败: {outline_path}, {e}")
            return None

    @staticmethod
    def _outline_json_to_md(node: dict, depth: int = 0, numbering: str = "") -> str:
        """渲染大纲 JSON 为 Markdown，带编号，跳过 L5。"""
        if not node:
            return ""
        md = ""
        name = node.get("name", "")
        level = node.get("level", 0)

        if level == 5:
            return ""

        if name:
            if depth == 0:
                md += f"# {name}\n\n"
            else:
                prefix = "#" * min(depth + 1, 6)
                num_str = f"{numbering} " if numbering else ""
                md += f"{prefix} {num_str}{name}\n\n"

        children = node.get("children", [])
        visible = [c for c in children if c.get("level", 0) != 5]
        for i, child in enumerate(visible, 1):
            child_num = f"{numbering}{i}" if numbering else str(i)
            md += GraphRAGExecutor._outline_json_to_md(child, depth + 1, f"{child_num}.")
        return md

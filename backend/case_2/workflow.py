"""case_2 单轮工作流：用户问题 → 分析框架大纲（Markdown）。

4 步流水线：
  S1  FAISS 向量检索       → 候选节点（Top-K）
  S2  路径召回             → 从候选节点向上找 L1 根的完整路径
  S3  LLM 选锚节点         → 确定子树展开起点
  S4  子树展开+裁剪+渲染   → 动作式裁剪 → Markdown
"""
import logging

from case_2.kb_store import KBStore
from case_2.faiss_index import FaissRetriever, ensure_index, get_retriever
from case_2.steps import s1_retrieve, s2_path_recall, s3_anchor, s4_clip_render

logger = logging.getLogger(__name__)

_kb: KBStore | None = None


def _get_kb() -> KBStore:
    global _kb
    if _kb is None:
        _kb = KBStore.load()
        logger.info(f"[workflow] KBStore 已加载：{len(_kb.nodes)} 个节点")
    return _kb


async def run(query: str, llm_fn) -> dict:
    """
    执行完整工作流。

    Args:
        query:   用户输入的自然语言问题
        llm_fn:  async callable(prompt: str) -> str

    Returns:
        {
            "query": str,
            "candidates": [(id, score, node), ...],
            "recalled_paths": [{"hit_node", "score", "path", "path_str"}, ...],
            "anchors": [{"id", "name", "level", "reason"}, ...],
            "subtree": dict,
            "markdown": str,
        }
    """
    kb = _get_kb()

    # 启动时确保 FAISS 索引存在（首次运行会调用 embedding API 构建）
    await ensure_index(list(kb.nodes.values()))
    retriever = get_retriever()

    logger.info(f"\n{'='*60}")
    logger.info(f"[workflow] 开始处理: {query!r}")
    logger.info(f"{'='*60}")

    # S1 FAISS 向量检索
    logger.info("\n── S1 FAISS 向量检索 ──")
    candidates = await s1_retrieve.run(query, kb, retriever)

    # S2 路径召回
    logger.info("\n── S2 路径召回 ──")
    recalled_paths = s2_path_recall.run(candidates, kb)

    if not recalled_paths:
        logger.warning("[workflow] 路径召回为空，无法生成大纲")
        return {
            "query": query, "candidates": candidates,
            "recalled_paths": [], "anchors": [], "subtree": None,
            "markdown": "（未能匹配到相关知识节点，请换一种提问方式）",
        }

    # S3 LLM 选锚节点
    logger.info("\n── S3 LLM 选锚节点 ──")
    anchors = await s3_anchor.run(query, recalled_paths, llm_fn)

    # S4 子树展开 + 裁剪 + 渲染
    logger.info("\n── S4 子树展开 + 大纲裁剪 + 渲染 ──")
    subtree = s4_clip_render.build_subtree(anchors, kb)

    if subtree is None:
        return {
            "query": query, "candidates": candidates,
            "recalled_paths": recalled_paths, "anchors": anchors,
            "subtree": None, "markdown": "（锚点节点不存在，请检查知识库）",
        }

    clipped = await s4_clip_render.clip(query, subtree, llm_fn)
    markdown = s4_clip_render.render(clipped)

    logger.info(f"\n{'='*60}")
    logger.info(f"[workflow] 流程完成")
    logger.info(f"{'='*60}")

    return {
        "query": query,
        "candidates": candidates,
        "recalled_paths": recalled_paths,
        "anchors": anchors,
        "subtree": subtree,
        "markdown": markdown,
    }

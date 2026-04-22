"""Step 1: FAISS 向量检索 —— 找出与 query 最相关的候选节点。

输入：用户问题 query（原始字符串）
输出：[(node_id, score, node_dict), ...] 按 score 降序
"""
import logging

logger = logging.getLogger(__name__)

async def run(
    query: str,
    kb,
    retriever,
    top_k: int = 8,
    score_threshold: float = 0.5,
) -> list[tuple[str, float, dict]]:
    """
    kb:        KBStore 实例
    retriever: FaissRetriever 实例

    返回 [(node_id, score, node), ...]，仅保留 score >= score_threshold 的结果
    """
    logger.info(f"[S1-FAISS检索] query: {query!r}  top_k={top_k}  threshold={score_threshold}")

    hits = await retriever.search(query, top_k=top_k)
    hits = [(nid, s) for nid, s in hits if s >= score_threshold]

    results = []
    for node_id, score in hits:
        node = kb.get_node(node_id)
        if node:
            results.append((node_id, score, node))

    if results:
        logger.info(f"[S1-FAISS检索] 命中 {len(results)} 个候选节点:")
        for nid, score, node in results:
            logger.info(f"  [{score:.3f}] {nid} L{node['level']} {node['name']}")
    else:
        logger.warning("[S1-FAISS检索] 未命中任何节点")

    return results

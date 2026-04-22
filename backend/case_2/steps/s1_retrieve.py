"""S1: get_nodes —— 给定 query，从 FAISS 索引检索相关节点。

自包含函数：内部管理索引加载和节点数据，外部只需传 query + embedding 服务 + 超参。

硬编码路径（相对 case_2/）：
  data/faiss.index        FAISS 二进制索引
  data/faiss_id_map.json  node_id 映射表
  knowledge_base/nodes.json     节点详情
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from case_2.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# ─── 硬编码路径 ─────────────────────────────────────────────────────
_CASE2_DIR   = Path(__file__).parent.parent        # steps/ -> case_2/
_FAISS_INDEX = _CASE2_DIR / "data" / "faiss.index"
_FAISS_IDMAP = _CASE2_DIR / "data" / "faiss_id_map.json"
_NODES_FILE  = _CASE2_DIR / "knowledge_base" / "nodes.json"

# ─── 模块级缓存（进程内只加载一次）────────────────────────────────────
_index = None                        # faiss.Index
_id_map: list[str] = []              # index位置 → node_id
_node_map: dict[str, dict] = {}      # node_id → node dict


def _load_index() -> None:
    global _index, _id_map, _node_map
    import faiss

    _index = faiss.read_index(str(_FAISS_INDEX))
    _id_map = json.loads(_FAISS_IDMAP.read_text(encoding="utf-8"))["node_ids"]
    logger.info(f"[S1] FAISS 索引已加载：{_index.ntotal} 条向量")

    nodes = json.loads(_NODES_FILE.read_text(encoding="utf-8"))
    _node_map = {n["id"]: n for n in nodes}
    logger.info(f"[S1] 节点表已加载：{len(_node_map)} 个节点")


async def get_nodes(
    query: str,
    embedding_svc: EmbeddingService,
    top_k: int = 8,
    score_threshold: float = 0.5,
) -> list[dict]:
    """
    检索与 query 最相关的知识节点。

    Args:
        query:            用户输入的自然语言问题
        embedding_svc:    EmbeddingService 实例（提供 embedding API 的 url/model/dim）
        top_k:            最多返回候选数量
        score_threshold:  cosine 相似度阈值，低于此值的结果被过滤

    Returns:
        list[dict]，每项包含节点原始字段 + score，按 score 降序：
        [
            {
                "id": "L3_002",
                "level": 3,
                "name": "传送网络容量分析",
                "keywords": [...],
                "description": "...",
                "score": 0.87,
            },
            ...
        ]
    """
    global _index

    if _index is None:
        _load_index()

    logger.info(f"[S1] query={query!r}  top_k={top_k}  threshold={score_threshold}")

    qvec = await embedding_svc.get_embedding(query)   # shape=(1, dim)
    scores, indices = _index.search(qvec, top_k)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or float(score) < score_threshold:
            continue
        node_id = _id_map[idx]
        node = _node_map.get(node_id)
        if node:
            results.append({**node, "score": round(float(score), 4)})

    logger.info(f"[S1] 命中 {len(results)} 个节点（过滤前 top_k={top_k}）:")
    for n in results:
        logger.info(f"  [{n['score']:.3f}] {n['id']:8s} L{n['level']} {n['name']}")

    return results

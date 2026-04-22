"""FAISS 向量索引 —— 对 nodes.json 中所有节点建立 embedding 索引。

索引文件缓存在 knowledge_base/index.faiss + index.meta.json，
节点数据变化时（nodes.json mtime 改变）自动重建。

向量化文本 = "节点名称。关键词：k1 k2 k3。描述：xxx"
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from case_2.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

_KB_DIR = Path(__file__).parent / "knowledge_base"
_INDEX_FILE = _KB_DIR / "index.faiss"
_META_FILE = _KB_DIR / "index.meta.json"
_NODES_FILE = _KB_DIR / "nodes.json"


def _node_text(node: dict) -> str:
    """将节点信息拼为待 embedding 的文本。"""
    parts = [node.get("name", "")]
    kws = node.get("keywords", [])
    if kws:
        parts.append("关键词：" + " ".join(kws))
    desc = node.get("description", "")
    if desc:
        parts.append("描述：" + desc)
    return "。".join(parts)


async def build_index(nodes: list[dict], embedding_svc: EmbeddingService) -> None:
    """构建并保存 FAISS 索引（IndexFlatIP = 内积 ≈ cosine）。"""
    import faiss

    texts = [_node_text(n) for n in nodes]
    logger.info(f"[faiss_index] 开始 embedding {len(texts)} 个节点...")
    t0 = time.perf_counter()
    vectors = await embedding_svc.get_embeddings_batch(texts)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[faiss_index] embedding 完成，shape={vectors.shape}，耗时 {elapsed:.0f}ms")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(_INDEX_FILE))

    meta = {
        "node_ids": [n["id"] for n in nodes],
        "nodes_mtime": _NODES_FILE.stat().st_mtime,
        "dim": dim,
    }
    _META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[faiss_index] 索引已保存：{_INDEX_FILE}（{index.ntotal} 条向量）")


def _index_is_fresh() -> bool:
    if not _INDEX_FILE.exists() or not _META_FILE.exists():
        return False
    meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
    return abs(meta.get("nodes_mtime", 0) - _NODES_FILE.stat().st_mtime) < 1.0


class FaissRetriever:
    """运行时 FAISS 检索器，懒加载索引。"""

    def __init__(self, embedding_svc: EmbeddingService):
        self._embedding_svc = embedding_svc
        self._index = None
        self._node_ids: list[str] = []

    def _load(self) -> None:
        import faiss
        self._index = faiss.read_index(str(_INDEX_FILE))
        meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
        self._node_ids = meta["node_ids"]
        logger.info(f"[faiss_index] 索引已加载：{self._index.ntotal} 条向量")

    async def search(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        """返回 [(node_id, score), ...] 按 score 降序。"""
        if self._index is None:
            self._load()

        qvec = await self._embedding_svc.get_embedding(query)  # shape=(1, dim)
        scores, indices = self._index.search(qvec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self._node_ids[idx], float(score)))
        return results


# ─── 全局单例（由 workflow 初始化时传入 embedding_svc）─────────────

_retriever: Optional[FaissRetriever] = None


def get_retriever(embedding_svc: Optional[EmbeddingService] = None) -> FaissRetriever:
    global _retriever
    if _retriever is None:
        if embedding_svc is None:
            raise RuntimeError("首次调用 get_retriever() 必须传入 embedding_svc")
        _retriever = FaissRetriever(embedding_svc)
    return _retriever


async def ensure_index(nodes: list[dict], embedding_svc: EmbeddingService) -> None:
    """启动时调用：若索引不存在或过期则重建。"""
    if _index_is_fresh():
        logger.info("[faiss_index] 索引已是最新，跳过重建")
    else:
        logger.info("[faiss_index] 索引不存在或已过期，开始重建")
        await build_index(nodes, embedding_svc)

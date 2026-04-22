"""构建 FAISS 向量索引脚本。

读取 knowledge_base/nodes.json，对每个节点生成 embedding，
输出 index.faiss 和 index.meta.json 到 .env 配置的路径。

使用方法：
    cd backend
    python case_2/build_index.py

依赖环境变量（从 case_2/.env 自动加载）：
    EMBEDDING_BASE_URL   Embedding API 地址
    EMBEDDING_MODEL      模型名称（默认 bge-m3）
    EMBEDDING_DIM        向量维度（默认 1024）
    FAISS_INDEX_PATH     索引输出路径
    FAISS_ID_MAP_PATH    id_map 输出路径
"""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# 将 backend/ 加入 sys.path
_CASE2_DIR = Path(__file__).parent
_BACKEND = _CASE2_DIR.parent
sys.path.insert(0, str(_BACKEND))

# 优先加载 case_2/.env
try:
    from dotenv import load_dotenv
    load_dotenv(_CASE2_DIR / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")

import faiss
import numpy as np

from case_2.embedding_service import EmbeddingService

_KB_DIR = _CASE2_DIR / "knowledge_base"

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIM", "1024"))
_DATA_DIR = _CASE2_DIR / "data"
FAISS_INDEX_PATH  = Path(os.getenv("FAISS_INDEX_PATH",  str(_DATA_DIR / "faiss.index")))
FAISS_ID_MAP_PATH = Path(os.getenv("FAISS_ID_MAP_PATH", str(_DATA_DIR / "faiss_id_map.json")))


def _node_text(node: dict) -> str:
    """节点文本 = 名称 + 关键词 + 描述，与 faiss_index.py 保持一致。"""
    parts = [node.get("name", "")]
    kws = node.get("keywords", [])
    if kws:
        parts.append("关键词：" + " ".join(kws))
    desc = node.get("description", "")
    if desc:
        parts.append("描述：" + desc)
    return "。".join(parts)


async def build():
    # 1. 读取 nodes.json
    nodes_file = _KB_DIR / "nodes.json"
    with open(nodes_file, encoding="utf-8") as f:
        nodes = json.load(f)
    logger.info(f"读取节点：{len(nodes)} 个（来自 {nodes_file}）")

    # 2. 准备文本
    texts = [_node_text(n) for n in nodes]
    for i, (n, t) in enumerate(zip(nodes, texts)):
        logger.info(f"  [{i:02d}] {n['id']:8s} {n['name']}")
        logger.info(f"         → {t[:80]}")

    # 3. 调用 Embedding API
    svc = EmbeddingService(EMBEDDING_BASE_URL, model=EMBEDDING_MODEL, dim=EMBEDDING_DIM)
    logger.info(f"\n开始生成 Embedding（model={EMBEDDING_MODEL}, url={EMBEDDING_BASE_URL}）...")
    t0 = time.perf_counter()
    vectors = await svc.get_embeddings_batch(texts, batch_size=32)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"Embedding 完成：shape={vectors.shape}，耗时 {elapsed:.0f}ms")

    # 4. 构建 FAISS IndexFlatIP（L2 归一化后内积 = cosine）
    dim = vectors.shape[1]
    if dim != EMBEDDING_DIM:
        logger.warning(f"实际维度 {dim} 与配置 EMBEDDING_DIM={EMBEDDING_DIM} 不一致，以实际为准")

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    logger.info(f"FAISS 索引构建完成：{index.ntotal} 条向量，dim={dim}")

    # 5. 保存索引文件
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info(f"已保存：{FAISS_INDEX_PATH}")

    # 6. 保存 id_map（与向量下标一一对应）
    meta = {
        "node_ids": [n["id"] for n in nodes],
        "nodes_mtime": nodes_file.stat().st_mtime,
        "dim": dim,
        "model": EMBEDDING_MODEL,
        "total": len(nodes),
    }
    FAISS_ID_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAISS_ID_MAP_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"已保存：{FAISS_ID_MAP_PATH}")

    logger.info(f"\n✅ 完成！共 {len(nodes)} 个节点已写入向量索引。")
    logger.info(f"   运行 workflow 时将直接加载，无需重复构建（除非修改 nodes.json）。")


if __name__ == "__main__":
    asyncio.run(build())

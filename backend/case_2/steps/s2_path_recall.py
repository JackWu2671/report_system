"""Step 2: 路径召回 —— 对候选节点向上 BFS，还原从 L1 根到命中节点的完整路径。

无 LLM 调用，纯内存图遍历（relations.json 邻接表）。

输出结构：
  [
    {
      "hit_node": {id, name, level, ...},
      "score": 0.92,
      "path": [root_node, ..., hit_node],   # L1 → ... → 命中节点
      "path_str": "政企OTN升级 > fgOTN部署 > 传送网络容量分析",
    },
    ...
  ]
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def _find_paths_to_root(node_id: str, kb) -> list[list[str]]:
    """
    BFS 向上找所有从 L1 根到 node_id 的路径（节点 id 列表，从根到叶）。
    返回路径列表（一个节点可能有多条路径，因为有共享节点）。
    """
    # DFS 向上，收集所有路径
    def dfs(nid: str, path: list[str]) -> list[list[str]]:
        parents = kb.get_parent_ids(nid)
        if not parents:
            # 已到达根节点
            return [list(reversed(path + [nid]))]
        paths = []
        for pid in parents:
            paths.extend(dfs(pid, path + [nid]))
        return paths

    return dfs(node_id, [])


def run(candidates: list[tuple[str, float, dict]], kb) -> list[dict]:
    """
    candidates: [(node_id, score, node), ...]
    返回路径列表，每条路径附带命中节点和得分。
    """
    if not candidates:
        logger.warning("[S2-路径召回] 候选节点为空，跳过")
        return []

    recalled: list[dict] = []
    seen_paths: set[str] = set()  # 去重：相同路径只保留得分最高的

    for node_id, score, node in candidates:
        paths = _find_paths_to_root(node_id, kb)
        for path_ids in paths:
            path_key = " > ".join(path_ids)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)

            path_nodes = [kb.get_node(nid) for nid in path_ids if kb.get_node(nid)]
            path_str = " > ".join(n["name"] for n in path_nodes)

            recalled.append({
                "hit_node": node,
                "score": score,
                "path": path_nodes,
                "path_str": path_str,
            })

    # 按得分降序
    recalled.sort(key=lambda x: -x["score"])

    logger.info(f"[S2-路径召回] 召回 {len(recalled)} 条路径:")
    for item in recalled:
        logger.info(f"  [{item['score']:.3f}] {item['path_str']}")

    return recalled

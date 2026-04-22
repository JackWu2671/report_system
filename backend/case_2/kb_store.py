"""KBStore —— 基于内存邻接表的知识图谱，替代 Neo4j。

数据结构：
  nodes: {id → node_dict}
  children: {parent_id → [child_id, ...]}（按 order 排序）
  parents:  {child_id  → [parent_id, ...]}

支持共享节点（同一节点可有多个父节点），与图数据库行为一致。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

_KB_DIR = Path(__file__).parent / "knowledge_base"


class KBStore:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.children: dict[str, list[str]] = {}
        self.parents: dict[str, list[str]] = {}

    @classmethod
    def load(cls) -> "KBStore":
        store = cls()
        with open(_KB_DIR / "nodes.json", encoding="utf-8") as f:
            for node in json.load(f):
                store.nodes[node["id"]] = node

        with open(_KB_DIR / "relations.json", encoding="utf-8") as f:
            edges: list[dict] = json.load(f)

        # 按 parent 分组后排序，保证子节点顺序与 order 一致
        from collections import defaultdict
        grouped: dict[str, list[dict]] = defaultdict(list)
        for e in edges:
            grouped[e["parent"]].append(e)

        for parent_id, edge_list in grouped.items():
            edge_list.sort(key=lambda e: e.get("order", 0))
            store.children[parent_id] = [e["child"] for e in edge_list]

        for e in edges:
            store.parents.setdefault(e["child"], [])
            if e["parent"] not in store.parents[e["child"]]:
                store.parents[e["child"]].append(e["parent"])

        return store

    def get_node(self, node_id: str) -> Optional[dict]:
        return self.nodes.get(node_id)

    def get_children_ids(self, node_id: str) -> list[str]:
        return self.children.get(node_id, [])

    def get_parent_ids(self, node_id: str) -> list[str]:
        return self.parents.get(node_id, [])

    def get_ancestors(self, node_id: str) -> list[str]:
        """BFS 向上遍历，返回所有祖先 id（不含自身）。"""
        visited, queue = set(), [node_id]
        ancestors = []
        while queue:
            nid = queue.pop(0)
            for pid in self.get_parent_ids(nid):
                if pid not in visited:
                    visited.add(pid)
                    ancestors.append(pid)
                    queue.append(pid)
        return ancestors

    def build_subtree(self, root_id: str, visited: Optional[set] = None) -> dict:
        """
        从 root_id 向下 BFS 构建嵌套树（dict）。
        visited 防止共享节点在同一子树内被重复展开。
        """
        if visited is None:
            visited = set()
        node = dict(self.nodes[root_id])
        child_ids = self.get_children_ids(root_id)
        children = []
        for cid in child_ids:
            if cid not in visited:
                visited.add(cid)
                children.append(self.build_subtree(cid, visited))
        node["children"] = children
        return node

    def all_ids(self) -> list[str]:
        return list(self.nodes.keys())

    def search_by_keywords(self, query_tokens: list[str]) -> list[tuple[str, int]]:
        """
        纯文本关键词匹配打分。
        返回 [(node_id, score), ...] 按 score 降序，score>0 的节点。
        """
        results = []
        query_tokens_lower = [t.lower() for t in query_tokens]
        for nid, node in self.nodes.items():
            score = 0
            keywords = [k.lower() for k in node.get("keywords", [])]
            name = node.get("name", "").lower()
            desc = node.get("description", "").lower()
            for token in query_tokens_lower:
                # 精确关键词命中权重最高
                if token in keywords:
                    score += 3
                # 节点名称命中
                if token in name:
                    score += 2
                # 描述命中
                if token in desc:
                    score += 1
            if score > 0:
                results.append((nid, score))
        results.sort(key=lambda x: -x[1])
        return results

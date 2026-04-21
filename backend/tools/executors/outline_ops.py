"""大纲树结构操作公共工具函数。

OutlineClipExecutor 和 GraphRAGExecutor 共用，避免重复实现。
"""
from __future__ import annotations
from typing import Set


def collect_nodes_text(
    node: dict,
    depth: int = 0,
    skip_l5: bool = False,
    max_depth: int = 5,
) -> str:
    """将大纲树展开为缩进文本，供 LLM 理解结构。

    Args:
        skip_l5: 为 True 时跳过 L5 叶节点（Step 6.5 过滤场景使用）
        max_depth: 递归深度上限，防止超深树撑爆 prompt
    """
    lines = []
    name = node.get("name", "")
    level = node.get("level", 0)
    if name and not (skip_l5 and level == 5):
        lines.append(f"{'  ' * depth}- {name} (L{level})")
    if depth < max_depth:
        for child in node.get("children", []):
            sub = collect_nodes_text(child, depth + 1, skip_l5, max_depth)
            if sub:
                lines.append(sub)
    return "\n".join(lines)


def delete_node(node: dict, target_name: str) -> dict:
    """递归删除所有与 target_name 同名的节点（含其子树）。"""
    if not node.get("children"):
        return node
    node["children"] = [
        delete_node(c, target_name)
        for c in node["children"]
        if c.get("name") != target_name
    ]
    return node


def keep_only(node: dict, target_names: Set[str]) -> dict:
    """仅保留 target_names 中的节点及其祖先/后代路径。"""
    if not node.get("children"):
        return node
    node["children"] = [
        keep_only(c, target_names)
        for c in node["children"]
        if c.get("name") in target_names or has_descendant(c, target_names)
    ]
    return node


def has_descendant(node: dict, names: Set[str]) -> bool:
    """判断节点自身或任意后代的名称是否在 names 中。"""
    if node.get("name") in names:
        return True
    return any(has_descendant(c, names) for c in node.get("children", []))


def count_nodes(node: dict) -> int:
    """统计节点总数（含自身）。"""
    return 1 + sum(count_nodes(c) for c in node.get("children", []))

"""Step 4: 子树展开 + 大纲裁剪（动作式）+ Markdown 渲染。

4a  从锚点向下展开完整子树（内存 BFS）
4b  LLM 输出裁剪动作指令（delete_chapter / keep_only）
4c  本地执行动作（复用 outline_ops）
4d  Markdown 渲染（无 LLM）

裁剪动作集（MVP）：
  delete_chapter(target_name)   删除章节及其子树
  keep_only(target_name)        只保留指定章节（删其余同级节点）

参数类动作（Phase 2，预留字段，本版不执行）：
  modify_threshold(target_name, param, value)
  set_filter(target_name, field, value)
  add_annotation(target_name, text)
"""
from __future__ import annotations

import copy
import json
import logging
import re

from case_2.outline_ops import collect_nodes_text, delete_node, keep_only

logger = logging.getLogger(__name__)

# Markdown 标题层级映射
_HEADING = {1: "#", 2: "##", 3: "###", 4: "####"}

# ─── 4a: 子树展开 ────────────────────────────────────────────────

def build_subtree(anchors: list[dict], kb) -> dict | None:
    """从锚点向下展开子树。多锚点取第一个（通常 LLM 已选好）。"""
    if not anchors:
        return None
    anchor_id = anchors[0]["id"]
    node = kb.get_node(anchor_id)
    if not node:
        logger.warning(f"[S4] 锚点 {anchor_id} 不在知识库中")
        return None
    subtree = kb.build_subtree(anchor_id)
    node_count = _count(subtree)
    logger.info(f"[S4-子树展开] 锚点={anchor_id} {node['name']}，共 {node_count} 个节点")
    return subtree


def _count(node: dict) -> int:
    return 1 + sum(_count(c) for c in node.get("children", []))


# ─── 4b + 4c: LLM 裁剪 ─────────────────────────────────────────

CLIP_PROMPT = """\
你是一个电信网络分析助手。根据用户问题，对以下大纲结构进行裁剪，去除不相关的部分。

## 用户问题
{query}

## 当前大纲节点（跳过 L5 细节）
{nodes_text}

## 可用动作（MVP 阶段）
| 动作 | 说明 | 示例 |
|------|------|------|
| delete_chapter | 删除章节及其全部子节点 | 用户不需要量子加密 |
| keep_only | 只保留此章节，删除同级其他章节 | 用户只关注容量分析 |

注：modify_threshold / set_filter / add_annotation 为预留字段，本次输出中请勿使用。

## 裁剪原则
- 用户问整体框架 → clip_needed=false，保留全部
- 用户聚焦某方向 → 删除无关章节，保留核心路径
- 不要过度裁剪：宁多勿少

## 输出格式（严格 JSON）
```json
{{
  "clip_needed": true,
  "reason": "裁剪原因（15字以内）",
  "instructions": [
    {{"type": "delete_chapter", "target_name": "章节名称"}},
    {{"type": "keep_only",      "target_name": "章节名称"}}
  ]
}}
```
只输出 JSON，不加解释。
"""


async def clip(query: str, subtree: dict, llm_fn) -> dict:
    """LLM 裁剪，返回操作后的子树（deepcopy，不修改原始树）。"""
    working = copy.deepcopy(subtree)

    nodes_text = collect_nodes_text(working, skip_l5=True, max_depth=4)
    prompt = CLIP_PROMPT.format(query=query, nodes_text=nodes_text)
    logger.info(f"[S4-大纲裁剪] prompt ({len(prompt)}ch):\n{prompt}")

    raw = await llm_fn(prompt)
    logger.info(f"[S4-大纲裁剪] LLM 响应 ({len(raw)}ch): {raw[:600]}")

    code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, re.DOTALL)
    if code_block:
        raw = code_block.group(1).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[S4-大纲裁剪] JSON 解析失败，跳过裁剪")
        return working

    if not result.get("clip_needed", False):
        logger.info(f"[S4-大纲裁剪] 无需裁剪: {result.get('reason', '')}")
        return working

    instructions = result.get("instructions", [])
    logger.info(f"[S4-大纲裁剪] {len(instructions)} 条动作: {instructions}")

    for inst in instructions:
        op = inst.get("type")
        target = inst.get("target_name", "").strip()
        if not target:
            continue
        if op == "delete_chapter":
            working = delete_node(working, target)
            logger.info(f"[S4-大纲裁剪] ✂ delete_chapter: {target!r}")
        elif op == "keep_only":
            working = keep_only(working, {target})
            logger.info(f"[S4-大纲裁剪] ✂ keep_only: {target!r}")
        else:
            logger.info(f"[S4-大纲裁剪] 跳过预留动作 {op!r}（Phase 2）")

    return working


# ─── 4d: Markdown 渲染 ─────────────────────────────────────────

def render(node: dict) -> str:
    lines: list[str] = []
    _render_node(node, lines)
    while lines and not lines[-1].strip():
        lines.pop()
    markdown = "\n".join(lines)
    logger.info(f"[S4-渲染] Markdown ({len(markdown)}ch):\n{markdown}")
    return markdown


def _render_node(node: dict, lines: list[str]) -> None:
    level = node.get("level", 0)
    name = node.get("name", "")
    description = node.get("description", "")

    if level in _HEADING:
        prefix = _HEADING[level]
        lines.append(f"{prefix} {name}")
        if description and level >= 3:
            lines.append(f"> {description}")
        lines.append("")
    elif level == 5:
        short = description[:40] + ("..." if len(description) > 40 else "")
        lines.append(f"- **{name}**：{short}")
    else:
        lines.append(f"- {name}")

    for child in node.get("children", []):
        _render_node(child, lines)

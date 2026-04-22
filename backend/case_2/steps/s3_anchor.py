"""Step 3: LLM 选锚节点 —— 从路径召回结果中选出最合适的分析入口。

锚点：子树向下展开的起始节点。
- 用户问整体框架 → 锚点选 L1/L2
- 用户聚焦具体问题 → 锚点选 L3/L4

输出：[{"id": ..., "name": ..., "reason": ...}, ...]
"""
import json
import logging
import re

logger = logging.getLogger(__name__)

ANCHOR_PROMPT = """\
你是一个电信网络分析助手。根据用户问题和知识库路径召回结果，选出最合适的分析入口节点（锚点）。

## 用户问题
{query}

## 知识库路径（按相关度排序）
{paths_text}

## 锚点选择规则
- 锚点是子树展开的起点，决定了分析框架的范围
- 若用户问整体框架/全貌，选 L1 或 L2 层节点（范围大）
- 若用户聚焦某一具体子问题，选 L3 或 L4 层节点（范围小）
- 锚点最多选 2 个，且应来自不同的业务方向
- 优先选择路径中得分最高的节点所在的根分支

## 输出格式（严格 JSON）
```json
{{
  "anchors": [
    {{"id": "节点ID", "name": "节点名称", "level": 2, "reason": "选择原因（15字以内）"}}
  ]
}}
```
只输出 JSON，不加解释。
"""


async def run(query: str, recalled_paths: list[dict], llm_fn) -> list[dict]:
    """
    recalled_paths: S2 输出的路径列表
    返回锚点列表 [{"id", "name", "level", "reason"}, ...]
    """
    if not recalled_paths:
        logger.warning("[S3-锚点选择] 无路径，跳过")
        return []

    # 格式化路径展示给 LLM（最多展示前 6 条）
    lines = []
    for i, item in enumerate(recalled_paths[:6]):
        lines.append(
            f"{i+1}. [{item['score']:.2f}] {item['path_str']}"
            f"（命中：L{item['hit_node']['level']} {item['hit_node']['name']}）"
        )
    paths_text = "\n".join(lines)

    prompt = ANCHOR_PROMPT.format(query=query, paths_text=paths_text)
    logger.info(f"[S3-锚点选择] prompt ({len(prompt)}ch):\n{prompt}")

    raw = await llm_fn(prompt)
    logger.info(f"[S3-锚点选择] LLM 响应 ({len(raw)}ch): {raw[:600]}")

    # 提取 JSON
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, re.DOTALL)
    if code_block:
        raw = code_block.group(1).strip()

    try:
        result = json.loads(raw)
        anchors = result.get("anchors", [])
    except json.JSONDecodeError:
        logger.warning("[S3-锚点选择] JSON 解析失败，降级：取最高分路径的根节点")
        top_path = recalled_paths[0]["path"]
        root = top_path[0]
        anchors = [{"id": root["id"], "name": root["name"],
                    "level": root["level"], "reason": "得分最高路径根节点"}]

    logger.info(f"[S3-锚点选择] 选出 {len(anchors)} 个锚点:")
    for a in anchors:
        logger.info(f"  → {a['id']} L{a.get('level', '?')} {a['name']} ({a.get('reason', '')})")

    return anchors

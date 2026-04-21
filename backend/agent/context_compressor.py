"""上下文压缩器 —— 对标 DeerFlow SummarizationMiddleware。

触发条件（OR）：
  token 估算 > 15000（len(content)/3.3 求和）
  消息条数 > 60

压缩算法：
  保留 [system_prompt] + [摘要消息] + 最近 10 条
  分割点必须在安全边界（不拆分 AI/Tool 消息对）
  摘要注入为 HumanMessage（兼容 Anthropic 不允许中途插入 SystemMessage 的限制）
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

TOKEN_LIMIT = 15_000    # 估算 token 阈值
MSG_LIMIT = 60          # 消息条数阈值
KEEP_RECENT = 10        # 保留最近 N 条不压缩
SUMMARIZE_MAX_TOKENS = 4_000  # 送给 LLM 摘要的最大 token 数

DOMAIN_SUMMARY_PROMPT = """\
你是看网系统的上下文摘要助手。将以下对话历史压缩为精简摘要，重点保留：
1. 当前大纲结构和关键节点（哪些节点已保留/已删除）
2. 用户已施加的约束（行业过滤、阈值修改）
3. 已完成的数据查询结果摘要
4. 设计态流程的当前进度和已完成的步骤
5. 尚未执行的操作

聚焦"当前状态"，避免重复已完成操作的细节。输出控制在 400 字以内。

## 对话历史
{history}
"""


def _estimate_tokens(content: str) -> float:
    return len(content) / 3.3


def _total_tokens(messages: list[dict]) -> float:
    return sum(_estimate_tokens(m.get("content") or "") for m in messages)


def should_compress(messages: list[dict]) -> bool:
    """判断是否需要压缩（不含 system prompt，即 messages[1:]）。"""
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) > MSG_LIMIT:
        return True
    if _total_tokens(non_system) > TOKEN_LIMIT:
        return True
    return False


def _find_safe_split(messages: list[dict], target_end: int) -> int:
    """找到安全分割点：不能落在 ToolMessage 序列中间，向前移到 AssistantMessage 边界。"""
    # 从 target_end 向前查找，直到找到一个 user/assistant（不是 tool）角色
    idx = min(target_end, len(messages) - 1)
    while idx > 0:
        role = messages[idx].get("role", "")
        if role in ("user", "assistant") and messages[idx].get("tool_calls") is None:
            # 确认前面不是紧跟着 tool 消息
            if idx + 1 < len(messages) and messages[idx + 1].get("role") == "tool":
                idx -= 1
                continue
            return idx
        idx -= 1
    return 0


def _trim_to_tokens(messages: list[dict], max_tokens: int) -> list[dict]:
    """从前端截断，保证 token 总量不超过 max_tokens。"""
    result, total = [], 0.0
    for m in messages:
        t = _estimate_tokens(m.get("content") or "")
        if total + t > max_tokens:
            break
        result.append(m)
        total += t
    return result


async def compress(messages: list[dict], llm_service) -> list[dict]:
    """
    压缩对话历史。

    messages 格式：[SystemMessage, ...history..., HumanMessage(current)]
    返回压缩后的 messages，格式相同。
    """
    if len(messages) < 3:
        return messages

    system_msg = messages[0] if messages[0].get("role") == "system" else None
    history_start = 1 if system_msg else 0
    history = messages[history_start:]

    if len(history) <= KEEP_RECENT:
        return messages

    to_keep = history[-KEEP_RECENT:]
    to_summarize_raw = history[:-KEEP_RECENT]

    # 找安全分割点
    split_idx = _find_safe_split(to_summarize_raw, len(to_summarize_raw) - 1)
    to_summarize = to_summarize_raw[:split_idx + 1] if split_idx > 0 else to_summarize_raw

    # trim 防止摘要调用超限
    to_summarize = _trim_to_tokens(to_summarize, SUMMARIZE_MAX_TOKENS)

    if not to_summarize:
        return messages

    # 格式化历史文本
    history_text = "\n".join(
        f"[{m.get('role', '?')}]: {str(m.get('content', ''))[:500]}"
        for m in to_summarize
    )

    prompt = DOMAIN_SUMMARY_PROMPT.format(history=history_text)
    logger.info(f"[context_compressor] 摘要 prompt ({len(prompt)}ch): {prompt[:400]}")
    try:
        from llm.config import LLMConfig
        summary_config = LLMConfig(temperature=0.3, max_tokens=600)
        summary_text = await llm_service.complete(
            [{"role": "user", "content": prompt}], summary_config
        )
        logger.info(f"[context_compressor] 摘要结果 ({len(summary_text)}ch): {summary_text[:400]}")
    except Exception as e:
        logger.warning(f"上下文压缩摘要失败，跳过压缩: {e}")
        return messages

    # 摘要注入为 HumanMessage（Anthropic 兼容）
    summary_msg = {
        "role": "user",
        "content": f"[系统：以下是此前对话的摘要，供你参考]\n\n{summary_text}",
    }

    compressed = []
    if system_msg:
        compressed.append(system_msg)
    compressed.append(summary_msg)
    compressed.extend(to_keep)

    logger.info(
        f"上下文压缩：{len(messages)} 条 → {len(compressed)} 条，"
        f"token 估算 {_total_tokens(messages):.0f} → {_total_tokens(compressed):.0f}"
    )
    return compressed

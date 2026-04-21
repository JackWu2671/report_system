"""Memory 持久化存储 —— JSON 文件，原子写（temp + rename）。

存储格式：
{
  "user": {
    "workContext": {"summary": "..."},
    "topOfMind": {"summary": "..."}
  },
  "history": {
    "recentMonths": {"summary": "..."}
  },
  "facts": [
    {"id": "uuid", "content": "...", "category": "preference|behavior|context", "confidence": 0.9}
  ]
}
"""

import copy
import json
import logging
import os
import tempfile
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

_EMPTY_MEMORY = {
    "user": {
        "workContext": {"summary": ""},
        "topOfMind": {"summary": ""},
    },
    "history": {
        "recentMonths": {"summary": ""},
    },
    "facts": [],
}

_MAX_FACTS = 50   # 最多保留 fact 条数
_MIN_CONFIDENCE = 0.5  # 低于此置信度的 fact 被剔除


class MemoryStore:

    def __init__(self, memory_file: str = "./data/memory.json"):
        self._path = memory_file
        os.makedirs(os.path.dirname(self._path) if os.path.dirname(self._path) else ".", exist_ok=True)

    def load(self) -> dict:
        if not os.path.isfile(self._path):
            return copy.deepcopy(_EMPTY_MEMORY)
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 兼容旧格式：补全缺失字段
            for key, default in _EMPTY_MEMORY.items():
                if key not in data:
                    data[key] = copy.deepcopy(default)
            return data
        except Exception as e:
            logger.warning(f"MemoryStore: 读取失败，返回空记忆: {e}")
            return copy.deepcopy(_EMPTY_MEMORY)

    def save(self, data: dict):
        """原子写：先写临时文件，再 rename。"""
        dir_path = os.path.dirname(self._path) or "."
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=dir_path,
                suffix=".tmp", delete=False
            ) as tf:
                json.dump(data, tf, ensure_ascii=False, indent=2)
                tmp_path = tf.name
            os.replace(tmp_path, self._path)
            logger.debug(f"MemoryStore: 已保存，facts={len(data.get('facts', []))}")
        except Exception as e:
            logger.error(f"MemoryStore: 写入失败: {e}")
            if "tmp_path" in dir() and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def merge_update(self, update: dict):
        """将 LLM 提取的更新合并到现有记忆。"""
        current = self.load()

        # 更新 user 字段
        for key in ("workContext", "topOfMind"):
            if update.get("user", {}).get(key, {}).get("summary"):
                current["user"][key]["summary"] = update["user"][key]["summary"]

        # 更新 history
        if update.get("history", {}).get("recentMonths", {}).get("summary"):
            current["history"]["recentMonths"]["summary"] = update["history"]["recentMonths"]["summary"]

        # 合并 facts（去重 + 限量）
        existing_contents = {f["content"] for f in current.get("facts", [])}
        new_facts = []
        for fact in update.get("facts", []):
            content = fact.get("content", "").strip()
            if not content:
                continue
            if fact.get("confidence", 0) < _MIN_CONFIDENCE:
                continue
            if content not in existing_contents:
                new_facts.append({
                    "id": str(uuid.uuid4()),
                    "content": content,
                    "category": fact.get("category", "context"),
                    "confidence": min(1.0, max(0.0, float(fact.get("confidence", 0.7)))),
                })
                existing_contents.add(content)

        all_facts = current.get("facts", []) + new_facts
        # 按置信度排序，保留最高的 _MAX_FACTS 条
        all_facts.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        current["facts"] = all_facts[:_MAX_FACTS]

        self.save(current)
        return current

    def clear(self):
        self.save(copy.deepcopy(_EMPTY_MEMORY))

    def format_for_injection(self, max_tokens: int = 2000) -> str:
        """格式化记忆内容，注入 system prompt 的 <memory> 块。"""
        data = self.load()
        parts = []

        work = data.get("user", {}).get("workContext", {}).get("summary", "")
        if work:
            parts.append(f"工作上下文：{work}")

        top = data.get("user", {}).get("topOfMind", {}).get("summary", "")
        if top:
            parts.append(f"当前关注：{top}")

        recent = data.get("history", {}).get("recentMonths", {}).get("summary", "")
        if recent:
            parts.append(f"近期历史：{recent}")

        facts = data.get("facts", [])
        if facts:
            facts_text = "\n".join(
                f"- [{f.get('category', 'context')}] {f['content']} (置信度:{f.get('confidence', 0):.1f})"
                for f in facts[:20]  # 最多展示 20 条
            )
            parts.append(f"已记住的偏好和行为：\n{facts_text}")

        result = "\n\n".join(parts)

        # 粗略 token 限制（按字符数估算）
        if len(result) > max_tokens * 3:
            result = result[: max_tokens * 3]

        return result

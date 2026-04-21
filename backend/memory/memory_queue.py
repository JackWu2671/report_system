"""Memory 更新队列 —— 收集对话片段，异步触发记忆提取。"""

import asyncio
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryQueue:
    """轻量级异步队列，存储待处理的对话片段。"""

    def __init__(self):
        self._queue: deque[dict] = deque()
        self._lock = asyncio.Lock()

    async def enqueue(self, session_id: str, user_message: str, assistant_reply: str):
        async with self._lock:
            self._queue.append({
                "session_id": session_id,
                "user": user_message,
                "assistant": assistant_reply,
            })

    async def dequeue_all(self) -> list[dict]:
        async with self._lock:
            items = list(self._queue)
            self._queue.clear()
            return items

    def size(self) -> int:
        return len(self._queue)

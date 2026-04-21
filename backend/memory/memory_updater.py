"""Memory 更新器 —— debounce 30s，LLM 提取事实，原子更新存储。

对标 DeerFlow MemoryMiddleware 的异步写路径。
"""

import asyncio
import json
import logging

from memory.memory_queue import MemoryQueue
from memory.memory_store import MemoryStore

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 30   # 对标 DeerFlow
MAX_BATCH = 10          # 单次处理最多 N 条对话


EXTRACT_PROMPT = """\
你是看网系统的记忆提取助手。分析以下对话片段，提取值得记住的用户信息。

## 提取目标
1. 工作上下文：用户负责什么网络/业务领域
2. 当前关注：用户最近在关注/分析什么问题
3. 近期历史：用户最近几次分析的主要内容
4. 偏好事实：用户明确表达的分析偏好、习惯筛选条件、排除项

## 输出格式（严格 JSON）
```json
{{
  "user": {{
    "workContext": {{"summary": "50字以内，如无变化则为空"}},
    "topOfMind": {{"summary": "50字以内，如无变化则为空"}}
  }},
  "history": {{
    "recentMonths": {{"summary": "100字以内，如无变化则为空"}}
  }},
  "facts": [
    {{"content": "具体事实", "category": "preference|behavior|context", "confidence": 0.8}}
  ]
}}
```

## 规则
- facts 只提取明确、具体、有价值的信息（避免泛化）
- confidence: 0.5-1.0（越明确越高）
- 如果对话中没有新信息，返回空 facts 和空 summary
- 不要推断或猜测，只提取用户明确表达的内容

## 对话片段
{conversations}

输出纯 JSON，不加解释。
"""


class MemoryUpdater:

    def __init__(self, llm_service, memory_store: MemoryStore, memory_queue: MemoryQueue):
        self._llm = llm_service
        self._store = memory_store
        self._queue = memory_queue
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self):
        """在 FastAPI lifespan 中启动后台 worker。"""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._worker())
            logger.info("MemoryUpdater: 后台 worker 已启动")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MemoryUpdater: 已停止")

    async def _worker(self):
        """持续运行的 debounce worker。"""
        while self._running:
            try:
                await asyncio.sleep(DEBOUNCE_SECONDS)
                if self._queue.size() > 0:
                    await self._process_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MemoryUpdater worker 异常: {e}", exc_info=True)

    async def _process_batch(self):
        """批量处理队列中的对话，提取记忆。"""
        items = await self._queue.dequeue_all()
        if not items:
            return

        # 限制批次大小
        batch = items[:MAX_BATCH]
        conversations = "\n\n".join(
            f"[对话 {i+1}]\n用户：{item['user'][:300]}\n助手：{item['assistant'][:300]}"
            for i, item in enumerate(batch)
        )

        prompt = EXTRACT_PROMPT.format(conversations=conversations)
        logger.info(f"[memory_updater] 记忆提取 prompt ({len(prompt)}ch): {prompt[:400]}")
        try:
            from llm.config import LLMConfig
            cfg = LLMConfig(temperature=0.2, max_tokens=1000)
            raw = await self._llm.complete(
                [{"role": "user", "content": prompt}], cfg
            )
            logger.info(f"[memory_updater] 记忆提取结果 ({len(raw)}ch): {raw[:400]}")
            # 解析 JSON
            import re
            code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, re.DOTALL)
            if code_block:
                raw = code_block.group(1).strip()
            update = json.loads(raw)
            self._store.merge_update(update)
            logger.info(f"[memory_updater] 处理了 {len(batch)} 条对话，提取 {len(update.get('facts', []))} 条 facts")
        except Exception as e:
            logger.warning(f"MemoryUpdater: 记忆提取失败（已忽略）: {e}")

    async def enqueue_update(self, session_id: str, user_message: str, assistant_reply: str):
        """在对话结束后调用，将对话加入队列。"""
        await self._queue.enqueue(session_id, user_message, assistant_reply)

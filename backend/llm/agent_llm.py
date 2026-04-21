"""AgentLLM —— 上下文隔离的 LLM 会话 + 轨迹回调。

trace_callback 签名:
  async def callback(
      llm_type: str, step_name: str,
      request_messages: list, response_content: str,
      reasoning_content: str, model: str, temperature: float,
      elapsed_ms: float, success: bool, error: str
  )
"""

import logging
import time
from typing import AsyncGenerator, Callable, Dict, Optional

from llm.config import LLMConfig
from llm.service import LLMService

logger = logging.getLogger(__name__)


class AgentLLM:

    def __init__(
        self,
        llm_service: LLMService,
        system_prompt: str = "",
        config: LLMConfig | None = None,
        trace_callback: Optional[Callable] = None,
        llm_type: str = "",
        step_name: str = "",
    ):
        self._svc = llm_service
        self.system_prompt = system_prompt
        self.config = config or LLMConfig()
        self.history: list[dict] = []
        # 轨迹
        self._trace_cb = trace_callback
        self.llm_type = llm_type
        self.step_name = step_name

    def _msgs(self, user_content: str) -> list[dict]:
        m = []
        if self.system_prompt:
            m.append({"role": "system", "content": self.system_prompt})
        m.extend(self.history)
        m.append({"role": "user", "content": user_content})
        return m

    async def chat(self, content: str) -> str:
        tag = f"[{self.llm_type}/{self.step_name}]"
        logger.info(f"{tag} chat prompt ({len(content)}ch): {content[:600]}")
        msgs = self._msgs(content)
        t0 = time.perf_counter()
        err, result = None, {"content": "", "reasoning_content": ""}
        try:
            result = await self._svc.complete_full(msgs, self.config)
        except Exception as e:
            err = str(e)
            logger.warning(f"{tag} chat 失败: {e}")
            raise
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            await self._record_trace(msgs, result["content"], result["reasoning_content"],
                                      elapsed, err is None, err or "")
        logger.info(f"{tag} chat 结果 ({len(result['content'])}ch): {result['content'][:400]}")
        self.history.extend([{"role": "user", "content": content}, {"role": "assistant", "content": result["content"]}])
        return result["content"]

    async def chat_json(self, content: str) -> dict:
        """非流式 JSON。内部走 complete_full → _parse_json，保留 reasoning。
        若 content 为空但 reasoning_content 非空，自动从 reasoning 中提取（qwen3 think 模式降级）。
        """
        tag = f"[{self.llm_type}/{self.step_name}]"
        logger.info(f"{tag} chat_json prompt ({len(content)}ch): {content[:600]}")
        msgs = self._msgs(content)
        t0 = time.perf_counter()
        err, reasoning_text, resp_str = None, "", ""
        try:
            full = await self._svc.complete_full(msgs, self.config)
            reasoning_text = full["reasoning_content"]
            raw_content = full["content"]

            # qwen3 think 模式：content 为空时降级到 reasoning_content
            if not raw_content.strip() and reasoning_text.strip():
                logger.warning(
                    f"{tag} content 为空（reasoning={len(reasoning_text)}ch），"
                    f"降级从 reasoning 提取 JSON"
                )
                raw_content = reasoning_text

            r = self._svc._parse_json(raw_content)
            resp_str = str(r)
        except Exception as e:
            err = str(e)
            logger.warning(f"{tag} chat_json 失败: {e}")
            raise
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            await self._record_trace(msgs, resp_str, reasoning_text,
                                      elapsed, err is None, err or "")
        logger.info(f"{tag} chat_json 结果: {resp_str[:400]}")
        self.history.extend([{"role": "user", "content": content}, {"role": "assistant", "content": resp_str}])
        return r

    async def chat_stream(self, content: str) -> AsyncGenerator[str, None]:
        """流式，只 yield content 文本。"""
        msgs = self._msgs(content)
        t0 = time.perf_counter()
        content_parts, reasoning_parts = [], []
        err = None
        try:
            async for chunk in self._svc.complete_stream(msgs, self.config):
                if isinstance(chunk, dict):
                    if "content" in chunk:
                        content_parts.append(chunk["content"])
                        yield chunk["content"]
                    if "reasoning_content" in chunk:
                        reasoning_parts.append(chunk["reasoning_content"])
                    if "error" in chunk:
                        err = chunk["error"]; raise RuntimeError(chunk["error"])
        except RuntimeError:
            raise
        except Exception as e:
            err = str(e); raise
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            await self._record_trace(
                msgs, "".join(content_parts), "".join(reasoning_parts),
                elapsed, err is None, err or ""
            )
        self.history.extend([
            {"role": "user", "content": content},
            {"role": "assistant", "content": "".join(content_parts)},
        ])

    async def chat_stream_full(self, content: str) -> AsyncGenerator[Dict, None]:
        """流式，yield 完整 Dict。"""
        msgs = self._msgs(content)
        t0 = time.perf_counter()
        content_parts, reasoning_parts = [], []
        err = None
        try:
            async for chunk in self._svc.complete_stream(msgs, self.config):
                if isinstance(chunk, dict):
                    if "content" in chunk:
                        content_parts.append(chunk["content"])
                    if "reasoning_content" in chunk:
                        reasoning_parts.append(chunk["reasoning_content"])
                yield chunk
        except Exception as e:
            err = str(e); raise
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            await self._record_trace(
                msgs, "".join(content_parts), "".join(reasoning_parts),
                elapsed, err is None, err or ""
            )
        self.history.extend([
            {"role": "user", "content": content},
            {"role": "assistant", "content": "".join(content_parts)},
        ])

    async def _record_trace(self, request_messages, response_content,
                             reasoning_content, elapsed_ms, success, error):
        """通过回调记录轨迹（如果有 trace_callback）。"""
        if not self._trace_cb:
            return
        try:
            await self._trace_cb(
                llm_type=self.llm_type,
                step_name=self.step_name,
                request_messages=request_messages,
                response_content=response_content,
                reasoning_content=reasoning_content,
                model=self.config.model or self._svc.default_model,
                temperature=self.config.temperature,
                elapsed_ms=elapsed_ms,
                success=success,
                error=error,
            )
        except Exception:
            pass  # 轨迹记录失败不影响主流程

    def fork(self, system_prompt=None, config=None, llm_type=None, step_name=None):
        """派生子 Agent，继承 trace_callback。"""
        return AgentLLM(
            llm_service=self._svc,
            system_prompt=system_prompt if system_prompt is not None else self.system_prompt,
            config=config or self.config,
            trace_callback=self._trace_cb,
            llm_type=llm_type or self.llm_type,
            step_name=step_name or self.step_name,
        )

    def reset(self):
        self.history.clear()

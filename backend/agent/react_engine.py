"""SimpleReActEngine —— 自研 ReAct 循环引擎，对标 DeerFlow LangGraph Agent Loop。

执行流程：
  messages = [system] + chat_history[-20:] + [HumanMessage(user_input)]
  LOOP (max MAX_STEPS):
    ① 上下文压缩检查
    ② LLM call (with tool schemas)
    ③ if tool_calls:
         for each tool_call:
           yield SSE("tool_call", {name, args})
           execute tool → yield SSE events + collect SkillResult
           messages.append(ToolMessage)
           yield SSE("tool_result", {name, summary})
         循环检测
         continue
    ④ else (text only → final answer):
         yield SSE("chat_reply", content)
         break
"""

import json
import logging
import time
from typing import AsyncGenerator, Optional

from agent.context_compressor import compress, should_compress
from agent.loop_detector import LoopDetector
from tools.tool_registry import ToolContext, ToolRegistry
from config import settings
from llm.config import LLMConfig, REACT_AGENT_CONFIG
from llm.service import LLMService

logger = logging.getLogger(__name__)

MAX_STEPS: int = settings.REACT_MAX_STEPS
_REACT_CONFIG: LLMConfig = REACT_AGENT_CONFIG


def _ev(t: str, **kw) -> str:
    return json.dumps({"type": t, **kw}, ensure_ascii=False)


class SimpleReActEngine:

    def __init__(self, llm_service: LLMService, tool_registry: ToolRegistry):
        self._llm = llm_service
        self._tools = tool_registry

    async def run(
        self,
        session_id: str,
        user_message: str,
        system_prompt: str,
        chat_history: list[dict],
        tool_ctx: ToolContext,
        config: Optional[LLMConfig] = None,
        trace_callback=None,
    ) -> AsyncGenerator[str, None]:
        """
        执行 ReAct 循环。

        Yields SSE 事件字符串（JSON）：
          {"type": "tool_call", "name": ..., "args": {...}}
          {"type": "tool_result", "name": ..., "summary": ...}
          {"type": "chat_reply", "content": ...}
          业务 SSE 事件（来自工具内部）
        """
        cfg = config or _REACT_CONFIG
        tool_schemas = self._tools.get_openai_tools()
        loop_detector = LoopDetector()

        # 构建初始消息列表
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(chat_history[-20:] if len(chat_history) > 20 else chat_history)
        messages.append({"role": "user", "content": user_message})

        for step in range(MAX_STEPS):
            # ① 上下文压缩检查
            if should_compress(messages):
                try:
                    messages = await compress(messages, self._llm)
                except Exception as e:
                    logger.warning(f"ReAct: 上下文压缩失败（继续）: {e}")

            # ② LLM 调用（带工具 schemas）
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[dict] = []
            llm_error = None
            t0 = time.perf_counter()  # 每步单独计时

            try:
                async for chunk in self._llm.complete_stream(messages, cfg, tools=tool_schemas):
                    if "content" in chunk:
                        content_parts.append(chunk["content"])
                    if "reasoning_content" in chunk:
                        reasoning_parts.append(chunk["reasoning_content"])
                    if "tool_calls" in chunk:
                        tool_calls = chunk["tool_calls"]
                    if "error" in chunk:
                        llm_error = chunk["error"]
            except Exception as e:
                llm_error = str(e)

            if llm_error:
                logger.error(f"ReAct step {step}: LLM error: {llm_error}")
                yield _ev("chat_reply", content=f"对话服务暂时不可用：{llm_error}")
                break

            content_text = "".join(content_parts)
            reasoning_text = "".join(reasoning_parts)

            # 记录 LLM trace（如有 callback）
            # response_content：有文字用文字；纯工具调用时序列化 tool_calls 决策
            if trace_callback:
                try:
                    elapsed = (time.perf_counter() - t0) * 1000
                    if tool_calls and not content_text:
                        trace_response = json.dumps(
                            [{"name": tc.get("name"), "arguments": tc.get("arguments")}
                             for tc in tool_calls],
                            ensure_ascii=False,
                        )
                    else:
                        trace_response = content_text
                    await trace_callback(
                        llm_type="react_engine",
                        step_name=f"step_{step}",
                        request_messages=messages,
                        response_content=trace_response,
                        reasoning_content=reasoning_text,
                        model=cfg.model or self._llm.default_model,
                        temperature=cfg.temperature,
                        elapsed_ms=elapsed,
                        success=True,
                        error="",
                    )
                except Exception:
                    pass

            # ③ 有工具调用 → 执行工具
            if tool_calls:
                # 将 AI 的工具调用决策加入消息列表
                ai_tool_calls_msg = {
                    "role": "assistant",
                    "content": content_text or None,
                    "tool_calls": [
                        {
                            "id": tc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                            },
                        }
                        for i, tc in enumerate(tool_calls)
                    ],
                }
                messages.append(ai_tool_calls_msg)

                for tool_call in tool_calls:
                    tc_name = tool_call.get("name", "unknown")
                    tc_args = tool_call.get("arguments", {})
                    tc_id = tool_call.get("id", "")

                    # 发送工具调用开始事件
                    yield _ev("tool_call", name=tc_name, args=tc_args)

                    # 执行工具（记录耗时用于 tool_call_traces）
                    tool_result_summary = ""
                    tool_result_data = {}
                    tool_success = False
                    t_tool = time.perf_counter()
                    from agent.context import SkillResult

                    async for item in self._tools.execute(tool_call, tool_ctx):
                        if "sse" in item:
                            yield item["sse"]  # 透传业务 SSE 事件
                        elif "result" in item:
                            res: SkillResult = item["result"]
                            tool_success = res.success
                            tool_result_summary = res.summary
                            tool_result_data = res.data
                            # 如果工具执行失败，也告知 LLM
                            if not res.success and res.user_prompt:
                                yield _ev("chat_reply", content=res.user_prompt)

                    # 工具调用落库（tool_call_traces）
                    tool_elapsed = (time.perf_counter() - t_tool) * 1000
                    if hasattr(tool_ctx, "chat_history") and tool_ctx.chat_history and trace_callback:
                        try:
                            # 从 trace_callback 的闭包里拿 chat_history
                            _ch = tool_ctx.chat_history
                            _tid = getattr(tool_ctx, "_trace_id", session_id)
                            await _ch.save_tool_trace(
                                session_id=session_id, trace_id=_tid, step=step,
                                tool_name=tc_name, tool_id=tc_id, tool_args=tc_args,
                                result_summary=tool_result_summary, result_data=tool_result_data,
                                success=tool_success, elapsed_ms=tool_elapsed,
                            )
                        except Exception as e:
                            logger.debug(f"工具轨迹落库失败（可忽略）: {e}")

                    # 工具结果注入 ToolMessage
                    tool_result_content = json.dumps(
                        {"success": tool_success, "summary": tool_result_summary, "data": tool_result_data},
                        ensure_ascii=False, default=str
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_result_content,
                    })

                    # 发送工具结果事件（携带 id，与 tool_call 事件对应）
                    yield _ev("tool_result", id=tc_id, name=tc_name, summary=tool_result_summary, success=tool_success)

                # 循环检测
                status, loop_msg = loop_detector.check(tool_calls)
                if status == "stop":
                    messages.append({"role": "user", "content": loop_msg})
                    # 强制最后一次 LLM 调用输出文字答案（不带 tools）
                    try:
                        final_result = await self._llm.complete(messages, cfg)
                        yield _ev("chat_reply", content=final_result or "（循环已终止，请查看已生成的结果）")
                    except Exception:
                        yield _ev("chat_reply", content="（循环已终止，请查看已生成的结果）")
                    break
                elif status == "warn":
                    messages.append({"role": "user", "content": loop_msg})

                # 继续循环
                continue

            # ④ 纯文本回复 → 最终答案
            if content_text:
                yield _ev("chat_reply", content=content_text)
            else:
                yield _ev("chat_reply", content="（已处理完毕）")
            break

        else:
            # 达到 MAX_STEPS 上限
            logger.warning(f"ReAct: 达到最大步数 {MAX_STEPS}，强制输出")
            yield _ev("chat_reply", content="已达到最大处理步数，请查看已生成的结果。")

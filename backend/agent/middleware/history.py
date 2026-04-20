from agent.middleware.base import AgentMiddleware
from agent.context import AgentContext
from services.chat_history import ChatHistoryService

class HistoryMiddleware(AgentMiddleware):
    def __init__(self, ch: ChatHistoryService, max_rounds=10): self._ch = ch; self._mr = max_rounds
    async def before_agent(self, ctx: AgentContext):
        msgs = await self._ch.get_messages(ctx.session_id, limit=self._mr*2)
        filtered = []
        for m in msgs:
            meta = m.get("metadata") or {}
            if meta.get("is_thinking"): continue
            content = m["content"]
            if m["role"]=="assistant" and m.get("msg_type")=="skill_result":
                content = f"[摘要]{meta.get('summary','')}" if meta.get("summary") else "[已执行]"
            if m.get("msg_type")=="outline": content = "[已生成大纲]"
            filtered.append({"role":m["role"],"content":content})
        ctx.chat_history = filtered; return ctx

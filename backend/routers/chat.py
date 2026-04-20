import logging, uuid
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from models.chat import ChatRequest
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])

def _agent():
    from main import app_state; return app_state["lead_agent"]
def _ch():
    from main import app_state; return app_state["chat_history"]
def _memory():
    from main import app_state; return app_state.get("memory_store")

@router.post("/chat")
async def chat(req: ChatRequest, agent=Depends(_agent)):
    import json
    async def stream():
        try:
            async for chunk in agent.handle_message(req.session_id, req.message):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error(f"chat stream 异常 session={req.session_id}: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

@router.get("/sessions")
async def list_sessions(limit:int=50, offset:int=0, h=Depends(_ch)):
    return {"sessions": await h.list_sessions(limit, offset)}

@router.post("/sessions")
async def create_session(h=Depends(_ch)):
    return await h.create_session(str(uuid.uuid4()))

@router.get("/sessions/{sid}/messages")
async def get_messages(sid:str, limit:int=100, h=Depends(_ch)):
    return {"messages": await h.get_messages(sid, limit)}

@router.get("/sessions/{sid}/outline")
async def get_outline(sid:str, h=Depends(_ch)):
    s = await h.get_outline_state(sid); return s or {"outline_json":None,"anchor_info":None}

@router.get("/sessions/{sid}/artifacts")
async def get_artifacts(sid:str, h=Depends(_ch)):
    """获取会话产物：大纲 JSON。"""
    outline_state = await h.get_outline_state(sid)
    return {
        "outline_json": outline_state.get("outline_json") if outline_state else None,
        "anchor_info": outline_state.get("anchor_info") if outline_state else None,
    }

@router.delete("/sessions/{sid}")
async def delete_session(sid:str, h=Depends(_ch)):
    await h.delete_session(sid); return {"success":True}

# ─── Memory API ───

@router.get("/memory")
async def get_memory(ms=Depends(_memory)):
    if not ms:
        return {"user": {}, "history": {}, "facts": []}
    return ms.load()

@router.delete("/memory")
async def clear_memory(ms=Depends(_memory)):
    if ms:
        ms.clear()
    return {"success": True}

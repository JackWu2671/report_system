"""FastAPI 应用入口。

启动流程:
  1. 创建各基础服务 → 注册到 ServiceContainer
  2. SkillRegistry 扫描 SKILL.md
  3. SkillLoader 自动加载所有执行器（从 SKILL.md executor 声明 + ServiceContainer 依赖注入）
  4. 组装 Lead Agent
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from llm.service import LLMService
from pipeline.faiss_retriever import FAISSRetriever
from pipeline.neo4j_retriever import Neo4jRetriever
from pipeline.outline_renderer import OutlineRenderer
from services.embedding_service import EmbeddingService
from services.session_service import SessionService
from services.chat_history import ChatHistoryService
from services.kb_content_store import KBContentStore
from agent.lead_agent import LeadAgent
from agent.middleware import MiddlewareChain, HistoryMiddleware, OutlineStateMiddleware, PendingConfirmMiddleware
from agent.skill_registry import SkillRegistry
from agent.skill_loader import SkillLoader
from agent.service_container import ServiceContainer
from routers.chat import router as chat_router
from routers.admin import router as admin_router
from routers.skills import router as skills_router
from agent.memory import MemoryStore, MemoryQueue, MemoryUpdater
from utils.log_setup import setup_logging

setup_logging(log_dir=settings.LOG_DIR, level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("========== 启动 ==========")

    # ─── 1. 创建基础服务 ───
    from llm.config_loader import get_llm_config_loader
    llm_loader = get_llm_config_loader()
    llm_service = LLMService(
        base_url=llm_loader.get_base_url(),
        default_model=llm_loader.get_model_name(),
        api_key=llm_loader.get_api_key() or settings.LLM_API_KEY,  # YAML 优先，.env fallback
        proxy=llm_loader.get_proxy() or settings.LLM_PROXY,
        ssl_verify=llm_loader.get_ssl_verify(),
        timeout_connect=llm_loader.get_timeout_connect(),
        timeout_read=llm_loader.get_timeout_read(),
        timeout_total=llm_loader.get_timeout_total(),
        think_tag_mode=llm_loader.get_think_tag_mode(),
    )
    embedding_service = EmbeddingService(settings.EMBEDDING_BASE_URL, dim=settings.EMBEDDING_DIM)

    faiss_retriever = FAISSRetriever(dim=settings.EMBEDDING_DIM)
    if os.path.exists(settings.FAISS_INDEX_PATH) and os.path.exists(settings.FAISS_ID_MAP_PATH):
        faiss_retriever.load(settings.FAISS_INDEX_PATH, settings.FAISS_ID_MAP_PATH)
    else:
        logger.warning("FAISS 索引不存在，请先运行 scripts/import_kb.py")

    neo4j_retriever = Neo4jRetriever(settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    try:
        await neo4j_retriever.verify_connectivity()
        logger.info(f"Neo4j: {await neo4j_retriever.get_entity_count()} 实体")
    except Exception as e:
        logger.error(f"Neo4j 连接失败: {e}")

    session_service = SessionService(settings.REDIS_URL)

    # 初始化 SQLite 对话历史
    os.makedirs(settings.DB_DIR, exist_ok=True)
    chat_history = ChatHistoryService(db_path=f"{settings.DB_DIR}/chat_history.db")
    await chat_history.init()

    kb_store = KBContentStore(f"{settings.DB_DIR}/kb_contents.db")
    await kb_store.init()

    outline_renderer = OutlineRenderer()

    # ─── 2. 注册所有服务到 ServiceContainer ───
    container = ServiceContainer()
    container.register("llm_service", llm_service)
    container.register("embedding_service", embedding_service)
    container.register("faiss_retriever", faiss_retriever)
    container.register("neo4j_retriever", neo4j_retriever)
    container.register("outline_renderer", outline_renderer)
    container.register("session_service", session_service)
    container.register("chat_history", chat_history)
    container.register("kb_store", kb_store)

    # IndicatorResolver（L5 paragraph 模板库）
    from services.data.indicator_resolver import IndicatorResolver
    _default_indicators_path = os.path.join(os.path.dirname(__file__), "services", "data", "default_indicators.json")
    indicator_resolver = IndicatorResolver(_default_indicators_path)
    container.register("indicator_resolver", indicator_resolver)

    logger.info(f"ServiceContainer: {container.registered_names()}")

    # ─── 3. Skill 体系：扫描 + 自动加载执行器 ───
    registry = SkillRegistry("./skills")
    registry.scan()
    container.register("skill_registry", registry)

    loader = SkillLoader(registry)
    loader.auto_load_all(container)
    logger.info(f"已加载执行器: {loader.loaded_skills()}")

    # ─── 4. Memory 系统 ───
    memory_store = MemoryStore(memory_file=os.path.join(settings.DB_DIR, "memory.json"))
    memory_queue = MemoryQueue()
    memory_updater = MemoryUpdater(llm_service, memory_store, memory_queue)

    # ─── 5. 组装 Agent ───
    mw = MiddlewareChain([
        HistoryMiddleware(chat_history),
        OutlineStateMiddleware(chat_history),
        PendingConfirmMiddleware(session_service),
    ])
    lead_agent = LeadAgent(llm_service, mw, registry, loader, chat_history, session_service, memory_store=memory_store, memory_updater=memory_updater)

    # ─── 6. 全局状态 ───
    app_state.update({
        "lead_agent": lead_agent,
        "chat_history": chat_history,
        "llm_service": llm_service,
        "neo4j_retriever": neo4j_retriever,
        "session_service": session_service,
        "faiss_retriever": faiss_retriever,
        "kb_store": kb_store,
        "container": container,
        "memory_store": memory_store,
        "skill_registry": registry,
    })

    # memory_updater.start()  # 已禁用：跨会话污染，业务场景不适合持续更新
    logger.info("========== 初始化完成 ==========")
    yield

    # ─── 清理 ───
    logger.info("========== 关闭 ==========")
    await memory_updater.stop()
    await neo4j_retriever.close()
    await session_service.close()
    await chat_history.close()
    await kb_store.close()
    await llm_service.close()


app = FastAPI(title="看网分析系统", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(skills_router)


@app.get("/health")
async def health():
    fr = app_state.get("faiss_retriever")
    return {"status": "ok", "version": "3.1.0", "faiss_vectors": fr.total if fr else 0}

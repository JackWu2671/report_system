from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # ─── LLM ───
    LLM_BASE_URL: str = "http://localhost:8000/v1"
    LLM_MODEL_NAME: str = "qwen3.5-27b"
    LLM_TEMPERATURE: float = 0.1
    LLM_TOP_P: float = 1.0
    LLM_API_KEY: str = ""
    LLM_PROXY: str = ""
    LLM_SSL_VERIFY: bool = True
    LLM_TIMEOUT_CONNECT: int = 60
    LLM_TIMEOUT_READ: int = 600
    LLM_TIMEOUT_TOTAL: int = 660
    LLM_THINK_TAG_MODE: str = "qwen3"

    # ─── Embedding ───
    EMBEDDING_BASE_URL: str = "http://localhost:8001/v1"
    EMBEDDING_DIM: int = 1024

    # ─── Neo4j ───
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "your_password"

    # ─── Redis ───
    REDIS_URL: str = "redis://localhost:6379/0"

    # ─── FAISS ───
    FAISS_INDEX_PATH: str = "./data/faiss.index"
    FAISS_ID_MAP_PATH: str = "./data/faiss_id_map.json"
    FAISS_TOP_K: int = 10
    FAISS_SCORE_THRESHOLD: float = 0.5

    # ─── SQLite ───
    DB_DIR: str = "./data"

    # ─── Skill 匹配（运行态） ───
    SKILL_MATCH_THRESHOLD: float = 0.7

    # ─── Skill 路由 ───
    SKILL_ROUTER_ENABLED: bool = True

    # ─── ReAct 引擎 ───
    REACT_MAX_STEPS: int = 15

    # ─── Logging ───
    LOG_DIR: str = "./data/logs"
    LOG_LEVEL: str = "DEBUG"

    # ─── 预留：真实数据源 ───
    # SQL_DB_URL: str = ""
    # EXTERNAL_API_BASE_URL: str = ""
    # EXTERNAL_API_KEY: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()

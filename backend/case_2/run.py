"""CLI 入口：python case_2/run.py "你的问题" [--debug]

环境变量：
  LLM_BASE_URL        OpenAI-compatible LLM API 地址
  LLM_API_KEY         LLM API Key
  LLM_MODEL           模型名称（默认 qwen3-235b-a22b）
  LLM_THINK_TAG_MODE  qwen3 | none（默认 none）
  EMBEDDING_BASE_URL  Embedding API 地址（默认同 LLM_BASE_URL）
  EMBEDDING_API_KEY   Embedding API Key（默认同 LLM_API_KEY）
  EMBEDDING_MODEL     Embedding 模型名称（默认 text-embedding-3-small）
"""
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND))

try:
    from dotenv import load_dotenv
    load_dotenv(_BACKEND / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("case_2.run")

import aiohttp

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8001/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-235b-a22b")
LLM_THINK_TAG_MODE = os.getenv("LLM_THINK_TAG_MODE", "none")

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or LLM_BASE_URL
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


async def _call_llm(prompt: str) -> str:
    """轻量 LLM 调用，直接调用 OpenAI-compatible API。"""
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1500,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=120)) as resp:
            resp.raise_for_status()
            data = await resp.json()

    choice = data["choices"][0]
    content = choice["message"].get("content", "") or ""
    reasoning = choice["message"].get("reasoning_content", "") or ""

    if LLM_THINK_TAG_MODE == "qwen3":
        if not content.strip() and reasoning.strip():
            logger.warning("[llm] content 为空，从 reasoning 提取（qwen3 think 模式）")
            content = reasoning
        elif not content.strip():
            cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            content = cleaned if cleaned else reasoning

    return content


async def main():
    if len(sys.argv) < 2:
        print("用法: python case_2/run.py \"你的问题\" [--debug]")
        print("示例: python case_2/run.py \"我想了解fgOTN的部署情况\"")
        sys.exit(1)

    query = sys.argv[1]

    from case_2.embedding_service import EmbeddingService
    from case_2.workflow import run

    embedding_svc = EmbeddingService(
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
        dim=EMBEDDING_DIM,
    )
    result = await run(query, _call_llm, embedding_svc)

    print("\n" + "=" * 60)
    print("【分析框架大纲】")
    print("=" * 60)
    print(result["markdown"])

    if "--debug" in sys.argv:
        print("\n【选中锚点】")
        print(json.dumps(result["anchors"], ensure_ascii=False, indent=2))
        print("\n【路径召回】")
        for item in result["recalled_paths"]:
            print(f"  [{item['score']:.3f}] {item['path_str']}")


if __name__ == "__main__":
    asyncio.run(main())

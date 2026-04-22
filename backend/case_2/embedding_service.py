import aiohttp
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, base_url, model="bge-m3", dim=1024):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim

    async def get_embedding(self, text):
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": text},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                if r.status != 200:
                    raise RuntimeError(f"Embedding失败:{r.status}")
                d = await r.json()
                vec = np.array(d["data"][0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                return vec.reshape(1, -1)

    async def get_embeddings_batch(self, texts, batch_size=32):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": batch},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as r:
                    if r.status != 200:
                        raise RuntimeError(f"Embedding批量失败:{r.status}")
                    for item in (await r.json())["data"]:
                        vec = np.array(item["embedding"], dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec /= norm
                        all_emb.append(vec)
            logger.info(f"Embedding: {min(i + batch_size, len(texts))}/{len(texts)}")
        return np.array(all_emb, dtype=np.float32)

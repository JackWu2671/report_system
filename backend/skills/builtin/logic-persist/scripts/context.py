"""logic-persist 上下文、服务束和 SSE 工具函数。"""
import json
from dataclasses import dataclass, field
from typing import Optional


def sse_event(event_type: str, data: dict) -> str:
    return json.dumps({"type": event_type, **data}, ensure_ascii=False)


def design_step(step: str, status: str, result: dict = None) -> str:
    d = {"type": "design_step", "step": step, "status": status}
    if result:
        d["result"] = result
    return json.dumps(d, ensure_ascii=False)


@dataclass
class SkillFactoryContext:
    """logic-persist 内部传递的上下文。"""
    raw_input: str = ""
    # Sub-Step 1 产出
    scene_intro: str = ""
    keywords: list[str] = field(default_factory=list)
    query_variants: list[str] = field(default_factory=list)
    # Sub-Step 2 产出
    structured_text: str = ""
    dimension_hints: list[dict] = field(default_factory=list)
    kb_nodes: list[dict] = field(default_factory=list)
    # Sub-Step 3 产出
    outline_json: Optional[dict] = None
    outline_md: str = ""
    anchor_node_id: str = ""
    # Sub-Step 4 产出
    bindings: list[dict] = field(default_factory=list)
    # Sub-Step 5 产出
    template_name: str = ""
    template_dir: str = ""
    source: str = "design"
    version: int = 1
    new_nodes: list[dict] = field(default_factory=list)

    def to_cache_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_cache_dict(cls, d: dict) -> "SkillFactoryContext":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


class ServiceBundle:
    """打包所有服务依赖，SubSkill 按需取用。"""
    __slots__ = ("llm", "embedding", "faiss", "neo4j", "session", "kb", "indicator_resolver")

    def __init__(self, llm_service, embedding_service, faiss_retriever,
                 neo4j_retriever, session_service, kb_store,
                 indicator_resolver=None):
        self.llm = llm_service
        self.embedding = embedding_service
        self.faiss = faiss_retriever
        self.neo4j = neo4j_retriever
        self.session = session_service
        self.kb = kb_store
        self.indicator_resolver = indicator_resolver

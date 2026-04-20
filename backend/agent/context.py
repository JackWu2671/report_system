from dataclasses import dataclass, field
from typing import Optional, Callable, Any

@dataclass
class SkillContext:
    session_id: str; user_message: str; params: dict = field(default_factory=dict)
    chat_history: list[dict] = field(default_factory=list)
    current_outline: Optional[dict] = None; outline_summary: str = ""
    has_pending_confirm: bool = False; pending_confirm_options: Optional[list[dict]] = None
    step_results: dict = field(default_factory=dict)
    trace_callback: Optional[Callable] = None  # LLM 轨迹回调

@dataclass
class SkillResult:
    success: bool; summary: str; data: dict = field(default_factory=dict)
    need_user_input: bool = False; user_prompt: str = ""

@dataclass
class PlanStep:
    skill: str; params: dict = field(default_factory=dict)

@dataclass
class Plan:
    intent: str; steps: list[PlanStep] = field(default_factory=list)
    reply_before: str = ""; raw: dict = field(default_factory=dict)

@dataclass
class ReflectAction:
    action: str; reason: str = ""; retry_params: dict = field(default_factory=dict)
    new_steps: list[PlanStep] = field(default_factory=list); user_question: str = ""

@dataclass
class AgentContext:
    session_id: str; user_message: str; trace_id: str = ""
    chat_history: list[dict] = field(default_factory=list)
    current_outline: Optional[dict] = None; outline_summary: str = ""
    has_pending_confirm: bool = False; pending_confirm_options: Optional[list[dict]] = None
    has_outline: bool = False

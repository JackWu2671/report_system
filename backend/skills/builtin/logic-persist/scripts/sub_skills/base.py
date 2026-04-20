"""SubSkill 基类——统一 running/done 事件、异常处理。"""
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Union

from context import (
    SkillFactoryContext, ServiceBundle, design_step,
)
from agent.context import SkillContext

logger = logging.getLogger(__name__)


class SubSkillBase(ABC):
    """Sub-Skill 基类。"""

    name: str = ""  # 子步骤名称，子类必须覆盖

    def __init__(self, svc: ServiceBundle):
        self._svc = svc

    @abstractmethod
    async def execute(self, fc: SkillFactoryContext, ctx: SkillContext) -> AsyncGenerator[str, None]:
        """子类实现：yield SSE 事件字符串，直接修改 fc 上下文。"""
        yield ""  # pragma: no cover

    async def run(self, fc: SkillFactoryContext, ctx: SkillContext) -> AsyncGenerator[str, None]:
        """统一包装：自动发 running/done + 异常捕获。"""
        yield design_step(self.name, "running")
        try:
            async for event in self.execute(fc, ctx):
                yield event
            yield design_step(self.name, "done")
        except Exception as e:
            logger.error(f"SubSkill {self.name} 异常: {e}", exc_info=True)
            yield design_step(self.name, "done", {"error": str(e)})

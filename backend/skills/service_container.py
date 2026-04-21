"""服务容器 —— 统一管理所有服务实例，供 SkillLoader 自动注入依赖。

main.py 创建服务后注册到这里，SkillLoader 实例化执行器时按名取用。
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ServiceContainer:
    """轻量级服务容器（本质是一个命名字典）。"""

    def __init__(self):
        self._services: dict[str, Any] = {}

    def register(self, name: str, instance: Any):
        """注册服务实例。"""
        self._services[name] = instance
        logger.debug(f"服务注册: {name} → {type(instance).__name__}")

    def get(self, name: str) -> Optional[Any]:
        """按名获取服务实例。"""
        return self._services.get(name)

    def get_required(self, name: str) -> Any:
        """按名获取服务实例，不存在则抛异常。"""
        svc = self._services.get(name)
        if svc is None:
            raise KeyError(f"服务未注册: {name}（已注册: {list(self._services.keys())}）")
        return svc

    def get_many(self, names: list[str]) -> dict[str, Any]:
        """批量获取。返回 {name: instance}，缺失的 key 会抛异常。"""
        result = {}
        missing = []
        for name in names:
            svc = self._services.get(name)
            if svc is None:
                missing.append(name)
            else:
                result[name] = svc
        if missing:
            raise KeyError(f"服务未注册: {missing}（已注册: {list(self._services.keys())}）")
        return result

    def has(self, name: str) -> bool:
        return name in self._services

    def registered_names(self) -> list[str]:
        return list(self._services.keys())

    def __repr__(self):
        return f"ServiceContainer({list(self._services.keys())})"

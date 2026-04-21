"""Skill 加载器 —— 基于 SKILL.md executor 声明自动发现、加载、注入依赖。

核心流程:
  1. 遍历 SkillRegistry 中所有已注册的 Skill
  2. 读取 executor 声明（module、class、deps、config）
  3. 动态导入 scripts/ 下的 Python 模块
  4. 从 ServiceContainer 中取出 deps 声明的服务实例
  5. 实例化执行器，注入依赖
"""

import importlib.util
import logging
import os
import sys
from typing import Any, Optional

from skills.skill_registry import SkillRegistry, SkillMeta
from skills.service_container import ServiceContainer

logger = logging.getLogger(__name__)


class SkillLoader:
    """Skill 执行器加载器。"""

    def __init__(self, registry: SkillRegistry):
        self._registry = registry
        self._executors: dict[str, Any] = {}  # {skill_name: executor_instance}

    # ─── 自动加载 ───

    def auto_load_all(self, container: ServiceContainer):
        """
        自动加载所有已启用 Skill 的执行器。

        读取每个 Skill 的 SKILL.md executor 声明，
        动态导入模块 → 从 container 获取依赖 → 实例化执行器。
        """
        for skill in self._registry.get_enabled():
            if not skill.executor or not skill.executor.module:
                logger.debug(f"Skill {skill.name}: 无 executor 声明，跳过")
                continue

            try:
                executor = self._load_one(skill, container)
                if executor:
                    self._executors[skill.name] = executor
                    logger.info(f"Executor: {skill.name} → {type(executor).__name__}")
            except Exception as e:
                logger.error(f"加载执行器失败: {skill.name}, {e}")
                # 单个 Skill 加载失败不影响其他 Skill

    def _load_one(self, skill: SkillMeta, container: ServiceContainer) -> Optional[Any]:
        """加载单个 Skill 的执行器。"""
        exe = skill.executor
        if not exe:
            return None

        # Step 1: 动态导入模块
        module = self._import_script(skill.skill_dir, exe.module)
        if module is None:
            return None

        # Step 2: 获取执行器类
        cls_name = exe.cls
        if not hasattr(module, cls_name):
            # 回退：找模块里第一个以 Executor 结尾的类
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and attr.endswith("Executor"):
                    cls_name = attr
                    logger.warning(f"Skill {skill.name}: 未找到 {exe.cls}，使用 {cls_name}")
                    break
            else:
                logger.error(f"Skill {skill.name}: 模块 {exe.module} 中无 Executor 类")
                return None

        executor_cls = getattr(module, cls_name)

        # Step 3: 从 container 获取依赖
        kwargs = {}
        for dep_name in exe.deps:
            svc = container.get(dep_name)
            if svc is None:
                logger.error(f"Skill {skill.name}: 依赖 '{dep_name}' 未在 ServiceContainer 中注册")
                return None
            kwargs[dep_name] = svc

        # Step 4: 合并 config 中的额外参数
        if exe.config:
            for k, v in exe.config.items():
                # 尝试转数值
                try:
                    v = int(v)
                except (ValueError, TypeError):
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        pass
                kwargs[k] = v

        # Step 5: 实例化
        try:
            return executor_cls(**kwargs)
        except TypeError as e:
            logger.error(f"Skill {skill.name}: 实例化 {cls_name} 失败（参数不匹配）: {e}")
            logger.error(f"  提供的参数: {list(kwargs.keys())}")
            return None

    # ─── 手动注册（兜底） ───

    def register_executor(self, skill_name: str, executor: Any):
        """手动注册执行器（用于无法自动加载的特殊情况）。"""
        self._executors[skill_name] = executor
        logger.info(f"Executor(手动): {skill_name} → {type(executor).__name__}")

    # ─── 查询 ───

    def get_executor(self, skill_name: str) -> Optional[Any]:
        return self._executors.get(skill_name)

    def has_executor(self, skill_name: str) -> bool:
        return skill_name in self._executors

    def loaded_skills(self) -> list[str]:
        return list(self._executors.keys())

    # ─── 内部 ───

    @staticmethod
    def _import_script(skill_dir: str, module_name: str):
        """动态导入 skill_dir/scripts/{module_name}.py"""
        path = os.path.join(skill_dir, "scripts", f"{module_name}.py")
        if not os.path.isfile(path):
            logger.error(f"脚本不存在: {path}")
            return None
        try:
            spec = importlib.util.spec_from_file_location(f"skills.{module_name}", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            logger.error(f"导入脚本失败: {path}, {e}")
            return None

"""Skill 注册中心 —— 扫描 SKILL.md，解析 frontmatter（含 executor 声明）。"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutorMeta:
    """执行器元数据（从 SKILL.md frontmatter 的 executor 字段解析）。"""
    module: str = ""      # 脚本模块名（如 graph_rag_executor）
    cls: str = ""         # 类名（如 GraphRAGExecutor）
    deps: list[str] = field(default_factory=list)   # 依赖的服务名列表
    config: dict = field(default_factory=dict)       # 额外配置（如 top_k, score_threshold）


@dataclass
class SkillMeta:
    """从 SKILL.md frontmatter 解析的完整 Skill 元数据。"""
    name: str
    display_name: str = ""
    description: str = ""
    enabled: bool = True
    params: dict = field(default_factory=dict)
    executor: Optional[ExecutorMeta] = None
    skill_dir: str = ""
    skill_md_path: str = ""
    source: str = "builtin"


class SkillRegistry:
    """Skill 注册中心。"""

    def __init__(self, skills_root: str = "./skills"):
        self._root = skills_root
        self._skills: dict[str, SkillMeta] = {}

    def scan(self):
        self._skills.clear()
        for src in ["builtin", "custom"]:
            self._scan_source(src)
        logger.info(f"Skills: {len(self._skills)}个, 启用{sum(1 for s in self._skills.values() if s.enabled)}个")

    def reload_custom_skills(self):
        """热重载 custom/ 目录（skill-persist 沉淀后调用，不影响 builtin）。"""
        # 移除旧的 custom skills
        to_remove = [name for name, s in self._skills.items() if s.source == "custom"]
        for name in to_remove:
            del self._skills[name]
        # 重新扫描 custom/
        self._scan_source("custom")
        logger.info(f"Custom skills 热重载完成: {len([s for s in self._skills.values() if s.source=='custom'])}个")

    def _scan_source(self, src: str):
        sd = os.path.join(self._root, src)
        if not os.path.isdir(sd):
            return
        for entry in os.listdir(sd):
            mp = os.path.join(sd, entry, "SKILL.md")
            if not os.path.isfile(mp):
                continue
            try:
                meta = self._parse(mp)
                meta.skill_dir = os.path.join(sd, entry)
                meta.skill_md_path = mp
                meta.source = src
                if not meta.name:
                    meta.name = entry
                # 名称冲突时（同名但不同目录）使用目录名作为 key，避免覆盖
                key = meta.name
                if key in self._skills and self._skills[key].skill_dir != meta.skill_dir:
                    logger.warning(
                        f"Skill 名称冲突: name='{key}' 已被 {self._skills[key].skill_dir} 占用，"
                        f"{entry} 改用目录名注册"
                    )
                    key = entry
                    meta.name = entry
                self._skills[key] = meta
                exe_info = f" → {meta.executor.cls}" if meta.executor and meta.executor.cls else " (无执行器)"
                logger.info(f"Skill: {meta.name} ({'ON' if meta.enabled else 'OFF'}) [{src}]{exe_info}")
            except Exception as e:
                logger.warning(f"SKILL.md 解析失败: {mp}, {e}")

    def get_all(self) -> list[SkillMeta]:
        return list(self._skills.values())

    def get_enabled(self) -> list[SkillMeta]:
        return [s for s in self._skills.values() if s.enabled]

    def get(self, name: str) -> Optional[SkillMeta]:
        return self._skills.get(name)

    def get_skills_prompt_section(self) -> str:
        """生成 Planner Prompt 的 Skill 摘要段落（只包含 builtin，custom 由 outline-generate Step 0 自动匹配）。"""
        enabled = [s for s in self._skills.values() if s.enabled and s.source == "builtin"]
        if not enabled:
            return ""
        items = []
        for s in enabled:
            pd = ""
            if s.params:
                parts = [f"{pn}({pdef.get('type','str')},{'req' if pdef.get('required') else 'opt'}): {pdef.get('description','')}"
                         for pn, pdef in s.params.items()]
                pd = "; ".join(parts)
            items.append(
                f"  <skill>\n    <n>{s.name}</n>\n    <display_name>{s.display_name}</display_name>\n"
                f"    <description>{s.description}</description>\n    <params>{pd}</params>\n  </skill>"
            )
        return "<available_skills>\n" + "\n".join(items) + "\n</available_skills>"

    def load_full_content(self, name: str) -> str:
        m = self._skills.get(name)
        if not m:
            return ""
        try:
            with open(m.skill_md_path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            return ""

    # ─── 解析 ───

    @staticmethod
    def _parse(filepath: str) -> SkillMeta:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        fm_match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
        if not fm_match:
            return SkillMeta(name="")

        fm = fm_match.group(1)
        meta = SkillMeta(name="")

        # 顶层简单字段
        for line in fm.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "name": meta.name = v
                elif k == "display_name": meta.display_name = v
                elif k == "description": meta.description = v
                elif k == "enabled": meta.enabled = v.lower() in ("true", "yes", "1")

        # params 块
        pm = re.search(r'params:\s*\n((?:[ \t]+.*(?:\n|$))*)', fm)
        if pm:
            meta.params = SkillRegistry._parse_nested_block(pm.group(1))

        # executor 块
        em = re.search(r'executor:\s*\n((?:[ \t]+.*(?:\n|$))*)', fm)
        if em:
            meta.executor = SkillRegistry._parse_executor(em.group(1))

        return meta

    @staticmethod
    def _parse_executor(block: str) -> ExecutorMeta:
        """解析 executor YAML 块。"""
        exe = ExecutorMeta()
        in_deps = False
        in_config = False
        for line in block.split("\n"):
            s = line.strip()
            if not s:
                continue
            indent = len(line) - len(line.lstrip())

            # 列表项（deps 子项）
            if s.startswith("- ") and in_deps:
                exe.deps.append(s[2:].strip())
                continue

            if ":" in s:
                k, _, v = s.partition(":")
                k, v = k.strip(), v.strip()

                if k == "module":
                    exe.module = v; in_deps = False; in_config = False
                elif k == "class":
                    exe.cls = v; in_deps = False; in_config = False
                elif k == "deps":
                    in_deps = True; in_config = False
                    if v and v.startswith("[") and v.endswith("]"):
                        exe.deps = [x.strip() for x in v[1:-1].split(",") if x.strip()]
                        in_deps = False
                elif k == "config":
                    in_config = True; in_deps = False
                elif in_config and indent >= 4:
                    # config 子字段
                    exe.config[k] = v
        return exe

    @staticmethod
    def _parse_nested_block(block: str) -> dict:
        """解析嵌套 YAML 块（params 等）。"""
        result, cur = {}, None
        for line in block.split("\n"):
            s = line.strip()
            if not s:
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= 2 and ":" in s:
                k, _, v = s.partition(":")
                k, v = k.strip(), v.strip()
                if not v:
                    cur = k
                    result[cur] = {}
                else:
                    result[k] = v
            elif indent > 2 and cur and ":" in s:
                k, _, v = s.partition(":")
                result[cur][k.strip()] = v.strip()
        return result

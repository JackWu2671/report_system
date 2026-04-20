"""Sub-Step 4：数据源绑定 + paragraph 富化。

从 IndicatorResolver 为每个 L5 节点读取 paragraph 模板，
就地写入 outline_json 节点，同时生成 fc.bindings 供 SkillPersist 沉淀用。
"""
from typing import AsyncGenerator
from sub_skills.base import SubSkillBase
from context import SkillFactoryContext
from agent.context import SkillContext


class DataBinding(SubSkillBase):
    name = "data_binding"

    async def execute(self, fc: SkillFactoryContext, ctx: SkillContext) -> AsyncGenerator[str, None]:
        resolver = self._svc.indicator_resolver
        l5_nodes: list[dict] = []
        _collect_l5_nodes(fc.outline_json, l5_nodes)

        bindings = []
        for node in l5_nodes:
            node_id = node.get("id", "")
            node_name = node.get("name", "")

            if resolver:
                paragraph = resolver.resolve(node_id, node_name, skill_dir="")
            else:
                # 兜底：无 resolver 时返回空段落
                paragraph = {
                    "content": "",
                    "metrics": [],
                    "tables": [],
                    "data_source": "Mock",
                    "params": {},
                }

            # 就地写入大纲节点（outline_json 同步更新）
            node["paragraph"] = paragraph

            bindings.append({
                "node_id": node_id,
                "node_name": node_name,
                "paragraph": paragraph,
            })

        fc.bindings = bindings
        return
        yield


def _collect_l5_nodes(node: dict, result: list):
    """递归收集所有 L5 节点的引用（就地修改用）。"""
    if not node:
        return
    if node.get("level") == 5:
        result.append(node)
        return
    for child in node.get("children", []):
        _collect_l5_nodes(child, result)


def _collect_l5_bindings(node: dict, bindings: list):
    """兼容旧接口：递归收集 L5 节点并生成 bindings 列表（不含 paragraph）。
    供 persist_current 路径使用，该路径大纲可能不含 paragraph。
    """
    if not node:
        return
    if node.get("level") == 5:
        bindings.append({
            "node_id": node.get("id", ""),
            "node_name": node.get("name", ""),
            "paragraph": node.get("paragraph", {
                "content": "", "metrics": [], "tables": [], "data_source": "Mock", "params": {}
            }),
        })
    for child in node.get("children", []):
        _collect_l5_bindings(child, bindings)

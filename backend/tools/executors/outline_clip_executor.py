"""大纲动态裁剪执行器。

替代原 outline-modify + filter_conditions，支持：
- delete_node: 删除指定节点及子树
- filter_param: 修改数据绑定参数
- keep_only: 仅保留指定节点
"""
import json
import logging
from typing import AsyncGenerator, Union

from agent.context import SkillContext, SkillResult
from llm.agent_llm import AgentLLM
from llm.config import LLMConfig
from llm.service import LLMService
from tools.executors.outline_ops import collect_nodes_text, delete_node, keep_only

logger = logging.getLogger(__name__)

CLIP_PROMPT = """你是大纲裁剪专家。根据用户指令，生成裁剪操作列表。

## 当前大纲节点
{nodes_text}

## 用户指令
{user_instruction}

## 输出格式
用 ```json ``` 代码块包裹输出，格式：
```json
{{"instructions": [
    {{"type": "delete_node", "target_name": "节点名", "level": 4}},
    {{"type": "filter_param", "target_name": "节点名", "param_key": "industry", "param_value": "金融企业"}},
    {{"type": "keep_only", "target_names": ["节点名1", "节点名2"]}}
]}}
```"""

def _ts(step, status, detail, data=None):
    p = {"type": "thinking_step", "step": step, "status": status, "detail": detail}
    if data: p["data"] = data
    return json.dumps(p, ensure_ascii=False)


class OutlineClipExecutor:

    def __init__(self, llm_service: LLMService):
        self._llm = llm_service

    async def execute(self, ctx: SkillContext) -> AsyncGenerator[Union[str, SkillResult], None]:
        outline = ctx.current_outline
        if not outline:
            yield SkillResult(False, "当前没有大纲，请先生成大纲")
            return

        instruction = ctx.params.get("instructions", ctx.user_message)
        yield _ts("outline_clip", "running", "正在解析裁剪指令...")

        nodes_text = collect_nodes_text(outline)

        agent = AgentLLM(self._llm, "", LLMConfig(temperature=0.1, max_tokens=1024),
                         trace_callback=ctx.trace_callback, llm_type="outline_clip", step_name="parse_clip")
        prompt = CLIP_PROMPT.format(nodes_text=nodes_text, user_instruction=instruction)
        logger.info(f"[outline_clip] 节点文本:\n{nodes_text}")
        try:
            result = await agent.chat_json(prompt)
            instructions = result.get("instructions", [])
            logger.info(f"[outline_clip] 解析到 {len(instructions)} 条操作: {instructions}")
        except Exception as e:
            logger.warning(f"[outline_clip] 裁剪指令解析失败: {e}")
            yield SkillResult(False, f"无法解析裁剪指令: {e}")
            return

        # 执行裁剪
        deleted_nodes, modified_params = [], []
        for inst in instructions:
            t = inst.get("type")
            if t == "delete_node":
                target = inst.get("target_name", "")
                if target:
                    outline = delete_node(outline, target)
                    deleted_nodes.append(target)
                    logger.info(f"[outline_clip] delete_node: {target!r}")
            elif t == "filter_param":
                target = inst.get("target_name", "")
                pk, pv = inst.get("param_key", ""), inst.get("param_value", "")
                if target and pk:
                    modified_params.append({"node": target, pk: pv})
                    logger.info(f"[outline_clip] filter_param: {target!r} {pk}={pv!r}")
            elif t == "keep_only":
                targets = inst.get("target_names", [])
                if targets:
                    outline = keep_only(outline, set(targets))
                    logger.info(f"[outline_clip] keep_only: {targets}")

        yield _ts("outline_clip", "done",
                   f"裁剪完成: 删除{len(deleted_nodes)}个节点, 修改{len(modified_params)}个参数")

        yield json.dumps({"type": "outline_clipped",
                          "deleted_nodes": deleted_nodes,
                          "modified_params": modified_params}, ensure_ascii=False)
        yield json.dumps({"type": "outline_updated", "outline_json": outline}, ensure_ascii=False)

        yield SkillResult(True, f"大纲裁剪完成",
                          data={"updated_outline": outline,
                                "deleted_nodes": deleted_nodes,
                                "modified_params": modified_params})


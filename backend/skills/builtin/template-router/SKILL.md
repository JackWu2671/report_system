---
name: template-router
display_name: 大纲模板路由
description: 检索已沉淀的大纲模板，通过 LLM 精排返回候选列表供用户选择。用户提出分析需求时优先调用。
enabled: true
params:
  query:
    type: string
    required: true
    description: 用户的分析需求描述
executor:
  module: template_router_executor
  class: TemplateRouterExecutor
  deps:
    - llm_service
    - session_service
  config: {}
---

# 大纲模板路由（Template Router）

## 适用场景
用户提出看网分析需求时，优先检索已沉淀的大纲模板，通过 LLM 精排返回候选列表，
让用户从中选择，或直接进入 GraphRAG 流程（customer-analysis）。

## 工作流：skill_router(query)

**Step 1：扫描模板目录**
遍历 `templates/` 下所有有效模板目录，读取 `outline_template.json` 和元数据。

**Step 2：加载模板元数据**
读取每个模板的 meta 字段（display_name、scene_intro、keywords、query_variants）。

**Step 3：LLM 精排**
将用户 query 和所有模板元数据发给 LLM，返回匹配的模板 ID 列表（最多 5 个）。

**Step 4：构建候选列表**
将匹配模板的展示信息（display_name、scene_intro、keywords、template_dir）组装成候选项。

**Step 5：推送 SSE + 保存 Redis**
发送 `skill_candidates` 事件到前端，同时将候选列表保存到 Redis（TTL=300s）。

## 关键规则
- 若无已沉淀模板或 LLM 精排无匹配，直接返回空候选列表，转入 customer-analysis
- LeadAgent 在下一轮消息中读取 Redis 中的候选列表，拦截用户选择
- 用户选择后由 LeadAgent 直接加载对应模板的 outline_template.json，不再调用 search_skill

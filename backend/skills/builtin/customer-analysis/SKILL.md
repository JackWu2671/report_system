---
name: customer-analysis
display_name: 客户分析
description: 运行态技能，通过 GraphRAG 检索知识图谱生成客户分析大纲 JSON，到大纲为止，不生成报告
enabled: true
params:
  query:
    type: string
    required: true
    description: 用户的分析需求文本
executor:
  module: graph_rag_executor
  class: GraphRAGExecutor
  deps:
    - llm_service
    - embedding_service
    - faiss_retriever
    - neo4j_retriever
    - outline_renderer
    - session_service
    - indicator_resolver
  config:
    top_k: 10
    score_threshold: 0.5
---

# 客户分析（GraphRAG 工作流）

## 适用场景
用户提出网络分析需求（如"分析 fgOTN 容量"、"评估传送网质量"），需要生成分析大纲 JSON。
大纲生成后即完成，不进行数据执行或报告生成。

## 工具调用：search_skill(query)
运行态大纲生成的统一入口，内部自动执行以下 7 步流程：

**Step 1-2：语义检索**
用户查询向量化 → FAISS 搜索知识库（top_k=10, threshold=0.5），返回候选节点。

**Step 3：路径补充**
Neo4j 路径分析，补全候选节点的完整祖先链（L1→L5）。

**Step 4：锚节点选择（LLM）**
LLM 从候选节点中选择最合适的起始锚点（L2-L4 层）。

**Step 5：L5 确认（如触发）**
若锚点为叶节点（L5），向用户提示选择分析层级，等待确认后继续。

**Step 6：子树遍历**
从锚点向下遍历子树，获取完整的子树结构。

**Step 6.5：查询意图过滤（自动）**
将用户查询与子树章节列表对比，用 LLM 判断用户是否只需要其中特定部分。
- 若查询含明确限定词（"只要覆盖分析"/"只看容量"），自动裁掉不相关章节
- 若查询是泛化需求（"分析fgOTN部署"），保留完整子树，跳过此步

**Step 7：大纲渲染**
生成带层级编号的大纲 JSON，流式输出。到此结束。

## 关键规则
- 章节过滤在 Step 6.5 自动完成，无需 LLM Agent 额外调用 clip_outline
- 若用户事后还要进一步删减节点，才追加 clip_outline
- 阈值/参数修改不在此处理，由 inject_params 工具负责
- L5 确认等待用户回复后，由系统自动继续生成
- 大纲 JSON 生成后即为最终产物，不调用 execute_data 或 render_report
- 已沉淀模板由 template-router 工具提前路由，本工具专注于 GraphRAG 检索生成

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
从锚点向下遍历子树，按约束条件过滤节点。

**Step 7：大纲渲染**
生成带层级编号的大纲 JSON，流式输出。到此结束。

## 关键规则
- 若用户附加约束（"不看XX"/"只看XX"），在 search_skill 后追加 clip_outline
- L5 确认等待用户回复后，由系统自动继续生成
- 大纲 JSON 生成后即为最终产物，不调用 execute_data 或 render_report
- 已沉淀模板由 template-router 工具提前路由，本工具专注于 GraphRAG 检索生成

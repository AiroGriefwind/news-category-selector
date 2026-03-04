# IMPLEMENTATION PLAN（执行约束）

## 重要约束
- 在执行任何开发任务前，必须完整阅读 `docs` 目录下以下 5 份文档：
  - `docs/PRD.MD`
  - `docs/TECHSTACK.MD`
  - `docs/STRUCTURE.MD`
  - `docs/IMPLEMENTATION_PLAN.MD`
  - `docs/PROGRESS.MD`
- 若以上文档与代码实现出现冲突，以 `PRD` 和本计划为准，并先更新 `PROGRESS` 记录决策。

## 分阶段实施
1. 建立文档与目录约束。
2. 定义统一数据结构与分类枚举。
3. 实现 WP 输入清洗器。
4. 实现分割策略（标题/副标题/自然段）。
5. 固化 prompts（标题、subtitle chunk、paragraph chunk、总决策）。
6. 实现 OpenAI-compatible LLM 调用与 JSON 容错。
7. 实现聚合器（标题高权重 + 低置信度降权）。
8. 完成 Streamlit demo 页面。
9. 用真实样例进行端到端验证并更新文档。

## 推进规则
- 每完成一个阶段，更新 `docs/PROGRESS.MD`。
- 项目文件变更后，更新 `docs/STRUCTURE.MD`。
- 保持输出可解释：最终结果必须附证据摘要与决策原因。


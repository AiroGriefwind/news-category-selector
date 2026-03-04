# 进度记录

## 2026-03-04

### 已完成
- 初始化项目目录：`docs/`、`core/`。
- 创建文档群：`PRD`、`TECHSTACK`、`STRUCTURE`、`IMPLEMENTATION_PLAN`、`PROGRESS`。
- 固化执行约束：开发前必须完整阅读 docs 文档群。
- 定义统一数据契约：`Category/Confidence` 枚举、`TitleResult/ChunkResult/FinalResult`。
- 实现 WP 输入清洗器：移除 caption/img、HTML 实体解码、提取 `p/h2/h3/h4`。
- 实现分割器：标题锚点 + 副标题优先分块 + 3 段分块兜底。
- 固化 4 套 Prompt：标题、subtitle chunk、paragraph chunk、总决策。
- 实现 OpenAI-compatible LLM 客户端：超时、重试、JSON 解析容错。
- 实现聚合器：标题高权重 + 低置信度降权（不丢弃）。
- 实现 Streamlit 页面：输入、清洗预览、分块预览、分类结果、JSON 下载。
- 补充依赖与环境模板：`requirements.txt`、`env.example`。
- 已完成结构与流程验收文档更新（`PROGRESS`、`STRUCTURE`）。

### 风险与问题
- 终端在中文路径下可能存在编码兼容问题，优先使用文件工具进行改动。
- 当前环境对点文件写入受限，使用 `env.example` 代替 `.env.example`。
- 未执行真实 API 端到端调用（需在本地填写可用 `LLM_API_KEY` 后运行验证）。

### 下一步
- 复制 `env.example` 为 `.env` 并填写真实 LLM 参数。
- 本地运行 `streamlit run app.py`，用真实样例进行端到端验证。


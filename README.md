# 新闻分类 Demo (Streamlit + LLM)

一个可本地运行并可部署到 Streamlit Cloud 的新闻分类 Demo。  
支持粘贴 WordPress 后台“文字模式”内容（HTML/caption/img），自动清洗并进行标题+分块分类，再输出最终聚合结果。

## 功能

- WP 原始 HTML 内容清洗（caption、img、实体解码）
- 单模型可选（`R1` / `R1-Distill`，默认 Distill）
- 分割策略下拉：
  - 全文分割：优先按副标题分块（`h2/h3/h4`，以及 `p` 内纯粗体小标题），无副标题时按每 3 段分块
  - 前三段分割：仅使用标题 + 前三自然段（默认）
- 批量导入：文章可先保存到列表，支持删除，再统一开始分析
- 多文章状态追踪：等待中 / 处理中 / 完成 / 出错
- 每篇文章显示总耗时与阶段耗时（标题 / 分块 / 总决策）
- 下载全部结果为 ZIP（每篇一个 JSON）

## 环境变量

复制 `env.example` 为 `.env` 并填写：

```env
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_API_KEY=your_api_key
LLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
LLM_TEMPERATURE=0.2
LLM_TIMEOUT_SECONDS=90
LLM_MAX_RETRIES=3
```

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Community Cloud

1. 将代码 push 到 GitHub（public repo）
2. Streamlit Cloud 新建 App，Main file 选择 `app.py`
3. 在 App Settings -> Secrets 填入环境变量（与 `.env` 同名）

## 安全提醒

- 不要提交 `.env` 与任何真实 API Key
- 若密钥曾在聊天/截图中暴露，请先重置


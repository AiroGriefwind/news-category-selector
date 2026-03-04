# 新闻分类 Demo (Streamlit + LLM)

一个可本地运行并可部署到 Streamlit Cloud 的新闻分类 Demo。  
支持粘贴 WordPress 后台“文字模式”内容（HTML/caption/img），自动清洗并进行标题+分块分类，再输出最终聚合结果。

## 功能

- WP 原始 HTML 内容清洗（caption、img、实体解码）
- 分割策略：
  - 标题单独分析
  - 优先按副标题分块（`h2/h3/h4`，以及 `p` 内纯粗体小标题）
  - 无副标题时按每 3 段分块
- 双模型对比分析（模型 A / 模型 B）
- 可视化进度条（标题 -> chunk -> 总决策）
- 结果下载为 JSON

## 环境变量

复制 `env.example` 为 `.env` 并填写：

```env
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_API_KEY=your_api_key
LLM_MODEL=Pro/deepseek-ai/DeepSeek-R1
LLM_MODEL_B=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
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


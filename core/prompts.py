from __future__ import annotations

import json

from core.splitter import ChunkInput


def build_title_prompt(title: str) -> str:
    return f"""你是一名新闻分类专家。请分析以下新闻标题，严格输出 JSON（不要输出任何额外文字）。

分类选项：财经、体育、娱乐、时事、科技、其他

标题：{title}

输出格式：
{{
  "primary": "主分类",
  "secondary": ["次分类1", "次分类2"],
  "confidence": "高/中/低",
  "score": 0,
  "reason": "一句话说明关键判断依据"
}}"""


def build_subtitle_chunk_prompt(chunk: ChunkInput) -> str:
    return f"""你是一名新闻分类专家。你将收到一篇新闻的“副标题分块”（一个副标题 + 其对应正文段落）。
请基于“副标题语义 + 正文内容”判断该分块的新闻分类，并严格输出 JSON（不要输出任何额外文字）。

分类选项：财经、体育、娱乐、时事、科技、其他

输入信息：
- 文章标题：{chunk.title_anchor}
- 分块序号：{chunk.chunk_index}
- 副标题：{chunk.subtitle}
- 分块正文：
---
{chunk.text}
---

输出格式：
{{
  "chunk_index": {chunk.chunk_index},
  "chunk_type": "subtitle",
  "primary": "主分类",
  "secondary": ["次分类1", "次分类2"],
  "confidence": "高/中/低",
  "score": 0,
  "key_signals": ["信号词1", "信号词2"],
  "reason": "一句话说明关键判断依据"
}}"""


def build_paragraph_chunk_prompt(chunk: ChunkInput) -> str:
    return f"""你是一名新闻分类专家。以下是一篇文章的第 {chunk.chunk_index} 组段落。
请根据段落内容进行分类，标题仅作为语义锚点，并严格输出 JSON（不要输出任何额外文字）。

分类选项：财经、体育、娱乐、时事、科技、其他

输入信息：
- 文章标题：{chunk.title_anchor}
- 分块序号：{chunk.chunk_index}
- 段落内容：
---
{chunk.text}
---

输出格式：
{{
  "chunk_index": {chunk.chunk_index},
  "chunk_type": "paragraph",
  "primary": "主分类",
  "secondary": [],
  "confidence": "高/中/低",
  "score": 0,
  "key_signals": ["信号词1", "信号词2"],
  "reason": "一句话说明关键判断依据"
}}"""


def build_final_prompt(title_result: dict, chunk_results: list[dict]) -> str:
    return f"""你是新闻分类总评审。请根据标题分析结果与多个正文分块分析结果，给出整篇文章最终分类。

分类选项：财经、体育、娱乐、时事、科技、其他

输入：
- 标题分析结果：{json.dumps(title_result, ensure_ascii=False)}
- 分块分析结果列表：{json.dumps(chunk_results, ensure_ascii=False)}
- 聚合规则提示：
  1) 标题权重更高
  2) 低置信度结果降权，不直接丢弃
  3) 若主分类接近，允许给出 secondary
  4) 若文章明显跨领域，reason 必须说明主次关系

仅输出 JSON：
{{
  "final_primary": "主分类",
  "final_secondary": ["次分类1", "次分类2"],
  "final_confidence": "高/中/低",
  "final_score": 0,
  "evidence_summary": ["证据1", "证据2", "证据3"],
  "decision_reason": "一句话解释最终归类逻辑"
}}"""


from __future__ import annotations

import json
import os

import streamlit as st
from dotenv import load_dotenv

from core.aggregator import aggregate_results
from core.cleaner import clean_wp_content
from core.llm_client import LLMClientError, OpenAICompatibleClient
from core.models import ChunkResult, TitleResult
from core.prompts import (
    build_final_prompt,
    build_paragraph_chunk_prompt,
    build_subtitle_chunk_prompt,
    build_title_prompt,
)
from core.splitter import split_article


def _render_chunk_preview(chunks) -> None:
    st.subheader("分块预览")
    if not chunks:
        st.info("未识别到可分析段落。")
        return

    for chunk in chunks:
        header = f"Chunk {chunk.chunk_index} | 类型: {chunk.chunk_type}"
        if chunk.subtitle:
            header += f" | 副标题: {chunk.subtitle}"
        with st.expander(header, expanded=False):
            st.write(chunk.text)


def _run_model_pipeline(client, model_name: str, title: str, chunks, tab):
    progress_text = tab.empty()
    progress_bar = tab.progress(0)
    total_steps = max(2, len(chunks) + 2)  # 标题 + n个chunk + 总决策
    completed = 0

    def advance(text: str) -> None:
        nonlocal completed
        completed += 1
        progress_text.info(f"{model_name}：{text}")
        progress_bar.progress(min(100, int(completed / total_steps * 100)))

    title_json = client.call_json(build_title_prompt(title.strip()), model=model_name)
    title_result = TitleResult.from_dict(title_json)
    advance("标题分析完成")

    chunk_results: list[ChunkResult] = []
    for chunk in chunks:
        if chunk.chunk_type == "subtitle":
            prompt = build_subtitle_chunk_prompt(chunk)
        else:
            prompt = build_paragraph_chunk_prompt(chunk)
        chunk_json = client.call_json(prompt, model=model_name)
        chunk_results.append(ChunkResult.from_dict(chunk_json))
        advance(f"分块 {chunk.chunk_index}/{len(chunks)} 完成")

    local_final = aggregate_results(title_result=title_result, chunk_results=chunk_results)
    final_prompt_json = client.call_json(
        build_final_prompt(
            title_result={
                "primary": title_result.primary,
                "secondary": title_result.secondary,
                "confidence": title_result.confidence,
                "score": title_result.score,
                "reason": title_result.reason,
            },
            chunk_results=[
                {
                    "chunk_index": item.chunk_index,
                    "chunk_type": item.chunk_type,
                    "primary": item.primary,
                    "secondary": item.secondary,
                    "confidence": item.confidence,
                    "score": item.score,
                    "key_signals": item.key_signals,
                    "reason": item.reason,
                }
                for item in chunk_results
            ],
        ),
        model=model_name,
    )
    advance("总决策完成")
    progress_text.success(f"{model_name}：分析完成")

    return {
        "title_result": title_result,
        "chunk_results": chunk_results,
        "local_final": local_final,
        "llm_final": final_prompt_json,
    }


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="新闻分类 Demo", layout="wide")
    st.title("新闻分类 Demo（Streamlit + LLM）")
    st.caption("支持粘贴 WP 文字模式内容，自动清洗并完成分类。")

    default_model_a = os.getenv("LLM_MODEL", "Pro/deepseek-ai/DeepSeek-R1")
    default_model_b = os.getenv("LLM_MODEL_B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    col_a, col_b = st.columns(2)
    with col_a:
        model_a = st.text_input("模型A", value=default_model_a)
    with col_b:
        model_b = st.text_input("模型B", value=default_model_b)

    title = st.text_input("新闻标题", placeholder="请输入标题")
    raw_body = st.text_area(
        "WP 原始正文（可包含 HTML/caption/img）",
        height=320,
        placeholder="请粘贴 WP 后台“文字”模式内容",
    )
    run_button = st.button("开始分析", type="primary")

    if not run_button:
        return

    if not title.strip() or not raw_body.strip():
        st.warning("请先输入标题和正文内容。")
        return

    article = clean_wp_content(raw_body)
    chunks = split_article(title=title.strip(), article=article)

    pre_tab, model_a_tab, model_b_tab = st.tabs(["预处理", f"模型A：{model_a}", f"模型B：{model_b}"])

    with pre_tab:
        st.subheader("清洗结果")
        st.write(f"段落数量：{len(article.paragraphs)}")
        st.write(f"检测到副标题：{'是' if article.has_subtitles else '否'}")

        with st.expander("清洗后段落", expanded=False):
            for i, paragraph in enumerate(article.paragraphs, start=1):
                st.markdown(f"{i}. {paragraph}")

        _render_chunk_preview(chunks)

    try:
        client = OpenAICompatibleClient.from_env()
        with model_a_tab:
            result_a = _run_model_pipeline(client, model_a, title, chunks, model_a_tab)
            st.subheader("标题分类结果")
            st.json(
                {
                    "primary": result_a["title_result"].primary,
                    "secondary": result_a["title_result"].secondary,
                    "confidence": result_a["title_result"].confidence,
                    "score": result_a["title_result"].score,
                    "reason": result_a["title_result"].reason,
                }
            )
            st.subheader("分块分类结果")
            st.json(
                [
                    {
                        "chunk_index": item.chunk_index,
                        "chunk_type": item.chunk_type,
                        "primary": item.primary,
                        "secondary": item.secondary,
                        "confidence": item.confidence,
                        "score": item.score,
                        "key_signals": item.key_signals,
                        "reason": item.reason,
                    }
                    for item in result_a["chunk_results"]
                ]
            )
            st.subheader("最终分类（本地聚合）")
            st.json(result_a["local_final"].to_dict())
            st.subheader("最终分类（LLM 总决策）")
            st.json(result_a["llm_final"])

        with model_b_tab:
            result_b = _run_model_pipeline(client, model_b, title, chunks, model_b_tab)
            st.subheader("标题分类结果")
            st.json(
                {
                    "primary": result_b["title_result"].primary,
                    "secondary": result_b["title_result"].secondary,
                    "confidence": result_b["title_result"].confidence,
                    "score": result_b["title_result"].score,
                    "reason": result_b["title_result"].reason,
                }
            )
            st.subheader("分块分类结果")
            st.json(
                [
                    {
                        "chunk_index": item.chunk_index,
                        "chunk_type": item.chunk_type,
                        "primary": item.primary,
                        "secondary": item.secondary,
                        "confidence": item.confidence,
                        "score": item.score,
                        "key_signals": item.key_signals,
                        "reason": item.reason,
                    }
                    for item in result_b["chunk_results"]
                ]
            )
            st.subheader("最终分类（本地聚合）")
            st.json(result_b["local_final"].to_dict())
            st.subheader("最终分类（LLM 总决策）")
            st.json(result_b["llm_final"])

        st.download_button(
            "下载双模型结果 JSON",
            data=json.dumps(
                {
                    "model_a": model_a,
                    "result_a": {
                        "title_result": {
                            "primary": result_a["title_result"].primary,
                            "secondary": result_a["title_result"].secondary,
                            "confidence": result_a["title_result"].confidence,
                            "score": result_a["title_result"].score,
                            "reason": result_a["title_result"].reason,
                        },
                        "chunk_results": [
                            {
                                "chunk_index": item.chunk_index,
                                "chunk_type": item.chunk_type,
                                "primary": item.primary,
                                "secondary": item.secondary,
                                "confidence": item.confidence,
                                "score": item.score,
                                "key_signals": item.key_signals,
                                "reason": item.reason,
                            }
                            for item in result_a["chunk_results"]
                        ],
                        "local_final_result": result_a["local_final"].to_dict(),
                        "llm_final_result": result_a["llm_final"],
                    },
                    "model_b": model_b,
                    "result_b": {
                        "title_result": {
                            "primary": result_b["title_result"].primary,
                            "secondary": result_b["title_result"].secondary,
                            "confidence": result_b["title_result"].confidence,
                            "score": result_b["title_result"].score,
                            "reason": result_b["title_result"].reason,
                        },
                        "chunk_results": [
                            {
                                "chunk_index": item.chunk_index,
                                "chunk_type": item.chunk_type,
                                "primary": item.primary,
                                "secondary": item.secondary,
                                "confidence": item.confidence,
                                "score": item.score,
                                "key_signals": item.key_signals,
                                "reason": item.reason,
                            }
                            for item in result_b["chunk_results"]
                        ],
                        "local_final_result": result_b["local_final"].to_dict(),
                        "llm_final_result": result_b["llm_final"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            file_name="news_classification_dual_model_result.json",
            mime="application/json",
        )
    except LLMClientError as exc:
        st.error(f"LLM 调用失败：{exc}")
        st.info("请检查 .env 中的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL。")


if __name__ == "__main__":
    main()


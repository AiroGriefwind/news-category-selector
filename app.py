from __future__ import annotations

import io
import json
import os
import re
import time
import uuid
import zipfile

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

MODEL_OPTIONS = {
    "R1-Distill": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "R1": "Pro/deepseek-ai/DeepSeek-R1",
}

SPLIT_STRATEGY_OPTIONS = {
    "前三段分割": "first3",
    "全文分割": "full",
}

STATUS_LABELS = {
    "waiting": "等待中",
    "processing": "处理中",
    "done": "完成",
    "error": "出错",
}


def _init_session_state() -> None:
    defaults = {
        "draft_title": "",
        "draft_body": "",
        "articles_queue": [],
        "analysis_status": {},
        "analysis_results": {},
        "analysis_metrics": {},
        "analysis_errors": {},
        "analysis_preprocess": {},
        "analysis_model_name": "",
        "analysis_model_label": "",
        "analysis_strategy_label": "",
        "flash_message": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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


def _run_model_pipeline(client, model_name: str, title: str, chunks, container):
    progress_text = container.empty()
    progress_bar = container.progress(0)
    total_steps = max(2, len(chunks) + 2)  # 标题 + n个chunk + 总决策
    completed = 0
    timing = {
        "article_total_seconds": 0.0,
        "title_stage_seconds": 0.0,
        "chunks_stage_seconds": 0.0,
        "final_stage_seconds": 0.0,
    }
    article_start = time.perf_counter()

    def advance(text: str) -> None:
        nonlocal completed
        completed += 1
        progress_text.info(f"{model_name}：{text}")
        progress_bar.progress(min(100, int(completed / total_steps * 100)))

    title_start = time.perf_counter()
    title_json = client.call_json(build_title_prompt(title.strip()), model=model_name)
    timing["title_stage_seconds"] = time.perf_counter() - title_start
    title_result = TitleResult.from_dict(title_json)
    advance("标题分析完成")

    chunks_start = time.perf_counter()
    chunk_results: list[ChunkResult] = []
    for chunk in chunks:
        if chunk.chunk_type == "subtitle":
            prompt = build_subtitle_chunk_prompt(chunk)
        else:
            prompt = build_paragraph_chunk_prompt(chunk)
        chunk_json = client.call_json(prompt, model=model_name)
        chunk_results.append(ChunkResult.from_dict(chunk_json))
        advance(f"分块 {chunk.chunk_index}/{len(chunks)} 完成")
    timing["chunks_stage_seconds"] = time.perf_counter() - chunks_start

    final_start = time.perf_counter()
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
    timing["final_stage_seconds"] = time.perf_counter() - final_start
    timing["article_total_seconds"] = time.perf_counter() - article_start
    advance("总决策完成")
    progress_text.success(f"{model_name}：分析完成")

    return {
        "title_result": title_result,
        "chunk_results": chunk_results,
        "local_final": local_final,
        "llm_final": final_prompt_json,
        "timing": timing,
    }


def _status_text(article_id: str) -> str:
    status = st.session_state["analysis_status"].get(article_id, "waiting")
    if status == "done":
        total_seconds = st.session_state["analysis_metrics"].get(article_id, {}).get("article_total_seconds", 0.0)
        return f"{STATUS_LABELS[status]}（{total_seconds:.2f} 秒）"
    return STATUS_LABELS.get(status, status)


def _remove_article(article_id: str) -> None:
    st.session_state["articles_queue"] = [
        item for item in st.session_state["articles_queue"] if item["id"] != article_id
    ]
    st.session_state["analysis_status"].pop(article_id, None)
    st.session_state["analysis_results"].pop(article_id, None)
    st.session_state["analysis_metrics"].pop(article_id, None)
    st.session_state["analysis_errors"].pop(article_id, None)
    st.session_state["analysis_preprocess"].pop(article_id, None)


def _render_queue() -> None:
    queue = st.session_state["articles_queue"]
    st.subheader("已保存文章")
    if not queue:
        st.info("请先添加文章，然后点击“开始分析”。")
        return

    for idx, item in enumerate(queue, start=1):
        preview = item["raw_body"][:15].replace("\n", " ")
        col_info, col_action = st.columns([10, 1])
        with col_info:
            st.write(f"{idx}. {item['title']} | {preview}")
        with col_action:
            if st.button("删除", key=f"delete_{item['id']}"):
                _remove_article(item["id"])
                st.rerun()


def _to_title_dict(result: TitleResult) -> dict:
    return {
        "primary": result.primary,
        "secondary": result.secondary,
        "confidence": result.confidence,
        "score": result.score,
        "reason": result.reason,
    }


def _to_chunk_dicts(chunk_results: list[ChunkResult]) -> list[dict]:
    return [
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
    ]


def _format_seconds(value: float) -> str:
    return f"{value:.2f} 秒"


def _render_status_list() -> None:
    for item in st.session_state["articles_queue"]:
        article_id = item["id"]
        st.write(f"- {item['title']}：{_status_text(article_id)}")


def _safe_filename(value: str, max_len: int = 32) -> str:
    sanitized = re.sub(r"[\\/:*?\"<>|]+", "_", value).strip()
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.strip("._")
    if not sanitized:
        return "untitled"
    return sanitized[:max_len]


def _build_zip_bytes(model_display_name: str) -> bytes:
    buffer = io.BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for item in st.session_state["articles_queue"]:
            article_id = item["id"]
            if st.session_state["analysis_status"].get(article_id) != "done":
                continue

            payload = st.session_state["analysis_results"][article_id]
            llm_final = payload.get("llm_final_result", {})
            local_final = payload.get("local_final_result", {})
            category = llm_final.get("final_primary") or local_final.get("final_primary") or "其他"
            base_name = (
                f"{_safe_filename(category, 12)}_"
                f"{_safe_filename(item['title'][:8], 16)}_"
                f"{_safe_filename(model_display_name, 24)}.json"
            )
            file_name = base_name
            suffix = 1
            while file_name in used_names:
                file_name = base_name.replace(".json", f"_{suffix}.json")
                suffix += 1
            used_names.add(file_name)
            zip_file.writestr(file_name, json.dumps(payload, ensure_ascii=False, indent=2))
    return buffer.getvalue()


def _render_preprocess_tab() -> None:
    preprocess_data = st.session_state["analysis_preprocess"]
    if not preprocess_data:
        st.info("尚未开始分析。")
        return

    for item in st.session_state["articles_queue"]:
        article_id = item["id"]
        article_data = preprocess_data.get(article_id)
        if not article_data:
            continue
        with st.expander(f"{item['title']} | {_status_text(article_id)}", expanded=False):
            st.write(f"段落数量：{len(article_data['paragraphs'])}")
            st.write(f"检测到副标题：{'是' if article_data['has_subtitles'] else '否'}")
            with st.expander("清洗后段落", expanded=False):
                for i, paragraph in enumerate(article_data["paragraphs"], start=1):
                    st.markdown(f"{i}. {paragraph}")
            st.subheader("分块预览")
            for chunk in article_data["chunks"]:
                header = f"Chunk {chunk['chunk_index']} | 类型: {chunk['chunk_type']}"
                if chunk["subtitle"]:
                    header += f" | 副标题: {chunk['subtitle']}"
                with st.expander(header, expanded=False):
                    st.write(chunk["text"])


def _render_model_tab(model_name: str, strategy_label: str) -> None:
    st.caption(f"当前模型：{model_name} | 分割策略：{strategy_label}")
    _render_status_list()

    for item in st.session_state["articles_queue"]:
        article_id = item["id"]
        with st.expander(f"{item['title']} | {_status_text(article_id)}", expanded=False):
            status = st.session_state["analysis_status"].get(article_id, "waiting")
            if status == "error":
                st.error(st.session_state["analysis_errors"].get(article_id, "未知错误"))
                continue
            if status != "done":
                st.info("尚未完成分析。")
                continue

            metrics = st.session_state["analysis_metrics"].get(article_id, {})
            st.write(
                "耗时："
                f"总计 {_format_seconds(metrics.get('article_total_seconds', 0.0))}，"
                f"标题 {_format_seconds(metrics.get('title_stage_seconds', 0.0))}，"
                f"分块 {_format_seconds(metrics.get('chunks_stage_seconds', 0.0))}，"
                f"总决策 {_format_seconds(metrics.get('final_stage_seconds', 0.0))}"
            )

            result = st.session_state["analysis_results"][article_id]
            st.subheader("标题分类结果")
            st.json(result["title_result"])
            st.subheader("分块分类结果")
            st.json(result["chunk_results"])
            st.subheader("最终分类（本地聚合）")
            st.json(result["local_final_result"])
            st.subheader("最终分类（LLM 总决策）")
            st.json(result["llm_final_result"])

    done_count = sum(1 for s in st.session_state["analysis_status"].values() if s == "done")
    if done_count:
        st.download_button(
            "下载全部结果 ZIP",
            data=_build_zip_bytes(st.session_state["analysis_model_label"] or model_name),
            file_name="news_classification_results.zip",
            mime="application/zip",
        )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="新闻分类 Demo", layout="wide")
    st.title("新闻分类 Demo（Streamlit + LLM）")
    st.caption("支持粘贴 WP 文字模式内容，自动清洗并完成分类。")
    _init_session_state()

    default_model_value = os.getenv("LLM_MODEL", MODEL_OPTIONS["R1-Distill"])
    default_index = 0
    for idx, option in enumerate(MODEL_OPTIONS.keys()):
        if MODEL_OPTIONS[option] == default_model_value:
            default_index = idx
            break

    model_display_name = st.selectbox("模型选择", options=list(MODEL_OPTIONS.keys()), index=default_index)
    model_name = MODEL_OPTIONS[model_display_name]
    strategy_label = st.selectbox("分割策略", options=list(SPLIT_STRATEGY_OPTIONS.keys()), index=0)
    strategy_value = SPLIT_STRATEGY_OPTIONS[strategy_label]

    st.text_input("新闻标题", placeholder="请输入标题", key="draft_title")
    st.text_area(
        "WP 原始正文（可包含 HTML/caption/img）",
        height=320,
        placeholder="请粘贴 WP 后台“文字”模式内容",
        key="draft_body",
    )
    col_save, col_run = st.columns([1, 1])
    with col_save:
        save_button = st.button("保存")
    with col_run:
        run_button = st.button("开始分析", type="primary")

    if save_button:
        title = st.session_state["draft_title"].strip()
        raw_body = st.session_state["draft_body"].strip()
        if not title or not raw_body:
            st.warning("请先输入标题和正文内容。")
        else:
            st.session_state["articles_queue"].append(
                {
                    "id": uuid.uuid4().hex[:8],
                    "title": title,
                    "raw_body": raw_body,
                }
            )
            st.session_state["draft_title"] = ""
            st.session_state["draft_body"] = ""
            st.session_state["flash_message"] = "文章已成功添加。"
            st.rerun()

    if st.session_state["flash_message"]:
        st.success(st.session_state["flash_message"])
        st.session_state["flash_message"] = ""

    _render_queue()

    if run_button:
        queue = st.session_state["articles_queue"]
        if not queue:
            st.warning("请先保存至少一篇文章。")
        else:
            st.session_state["analysis_status"] = {item["id"]: "waiting" for item in queue}
            st.session_state["analysis_results"] = {}
            st.session_state["analysis_metrics"] = {}
            st.session_state["analysis_errors"] = {}
            st.session_state["analysis_preprocess"] = {}
            st.session_state["analysis_model_name"] = model_name
            st.session_state["analysis_model_label"] = model_display_name
            st.session_state["analysis_strategy_label"] = strategy_label

            preprocess_cache = {}
            for item in queue:
                article = clean_wp_content(item["raw_body"])
                chunks = split_article(title=item["title"], article=article, strategy=strategy_value)
                preprocess_cache[item["id"]] = {"article": article, "chunks": chunks}
                st.session_state["analysis_preprocess"][item["id"]] = {
                    "paragraphs": article.paragraphs,
                    "has_subtitles": article.has_subtitles,
                    "chunks": [
                        {
                            "chunk_index": chunk.chunk_index,
                            "chunk_type": chunk.chunk_type,
                            "subtitle": chunk.subtitle,
                            "text": chunk.text,
                        }
                        for chunk in chunks
                    ],
                }

            try:
                client = OpenAICompatibleClient.from_env()
                status_panel = st.empty()
                for item in queue:
                    article_id = item["id"]
                    st.session_state["analysis_status"][article_id] = "processing"
                    with status_panel.container():
                        _render_status_list()

                    try:
                        pipeline_container = st.container()
                        data = preprocess_cache[article_id]
                        result = _run_model_pipeline(
                            client=client,
                            model_name=model_name,
                            title=item["title"],
                            chunks=data["chunks"],
                            container=pipeline_container,
                        )
                        st.session_state["analysis_results"][article_id] = {
                            "article_id": article_id,
                            "title": item["title"],
                            "model_name": model_name,
                            "split_strategy": strategy_label,
                            "title_result": _to_title_dict(result["title_result"]),
                            "chunk_results": _to_chunk_dicts(result["chunk_results"]),
                            "local_final_result": result["local_final"].to_dict(),
                            "llm_final_result": result["llm_final"],
                            "timing": result["timing"],
                        }
                        st.session_state["analysis_metrics"][article_id] = result["timing"]
                        st.session_state["analysis_status"][article_id] = "done"
                    except Exception as exc:  # noqa: BLE001
                        st.session_state["analysis_status"][article_id] = "error"
                        st.session_state["analysis_errors"][article_id] = str(exc)

                with status_panel.container():
                    _render_status_list()
            except LLMClientError as exc:
                st.error(f"LLM 初始化失败：{exc}")
                st.info("请检查 .env 中的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL。")
            except Exception as exc:  # noqa: BLE001
                st.error(f"批量分析失败：{exc}")

    if st.session_state["analysis_preprocess"]:
        st.divider()
        pre_tab, model_tab = st.tabs(
            [
                "预处理",
                f"模型结果：{st.session_state['analysis_model_label'] or model_display_name}",
            ]
        )
        with pre_tab:
            _render_preprocess_tab()
        with model_tab:
            _render_model_tab(
                model_name=st.session_state["analysis_model_name"] or model_name,
                strategy_label=st.session_state["analysis_strategy_label"] or strategy_label,
            )


if __name__ == "__main__":
    main()


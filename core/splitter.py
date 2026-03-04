from __future__ import annotations

from dataclasses import dataclass

from core.cleaner import CleanedArticle


@dataclass
class ChunkInput:
    chunk_index: int
    chunk_type: str
    title_anchor: str
    subtitle: str
    text: str


def _join_paragraphs(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs).strip()


def _build_paragraph_chunks(title: str, paragraphs: list[str], group_size: int = 3) -> list[ChunkInput]:
    if not paragraphs:
        return []

    if len(paragraphs) <= group_size:
        return [
            ChunkInput(
                chunk_index=1,
                chunk_type="paragraph",
                title_anchor=title,
                subtitle="",
                text=_join_paragraphs(paragraphs),
            )
        ]

    chunks: list[ChunkInput] = []
    chunk_index = 1
    for i in range(0, len(paragraphs), group_size):
        group = paragraphs[i : i + group_size]
        chunks.append(
            ChunkInput(
                chunk_index=chunk_index,
                chunk_type="paragraph",
                title_anchor=title,
                subtitle="",
                text=_join_paragraphs(group),
            )
        )
        chunk_index += 1
    return chunks


def _build_subtitle_chunks(title: str, article: CleanedArticle) -> list[ChunkInput]:
    chunks: list[ChunkInput] = []
    bucket: list[str] = []
    active_subtitle = ""
    chunk_index = 1

    def flush(current_subtitle: str, paragraphs: list[str]) -> None:
        nonlocal chunk_index
        if not paragraphs:
            return
        chunks.append(
            ChunkInput(
                chunk_index=chunk_index,
                chunk_type="subtitle" if current_subtitle else "paragraph",
                title_anchor=title,
                subtitle=current_subtitle,
                text=_join_paragraphs(paragraphs),
            )
        )
        chunk_index += 1

    for block in article.blocks:
        if block.kind == "subtitle":
            flush(active_subtitle, bucket)
            active_subtitle = block.text
            bucket = []
            continue
        bucket.append(block.text)

    flush(active_subtitle, bucket)
    return chunks


def split_article(title: str, article: CleanedArticle, group_size: int = 3) -> list[ChunkInput]:
    if article.has_subtitles:
        subtitle_chunks = _build_subtitle_chunks(title=title, article=article)
        if subtitle_chunks:
            return subtitle_chunks
    return _build_paragraph_chunks(title=title, paragraphs=article.paragraphs, group_size=group_size)


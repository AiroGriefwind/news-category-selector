from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup


CAPTION_PATTERN = re.compile(r"\[caption[^\]]*\].*?\[/caption\]", re.IGNORECASE | re.DOTALL)


@dataclass
class TextBlock:
    kind: str
    text: str
    level: str = ""


@dataclass
class CleanedArticle:
    blocks: list[TextBlock] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    has_subtitles: bool = False


def _normalize_text(value: str) -> str:
    decoded = html.unescape(value)
    condensed = re.sub(r"\s+", " ", decoded).strip()
    return condensed


def _extract_plain_paragraphs(text: str) -> list[str]:
    parts = [item.strip() for item in re.split(r"\n{2,}", text) if item.strip()]
    return [_normalize_text(item) for item in parts if _normalize_text(item)]


def _is_bold_subtitle_paragraph(node) -> bool:
    if node.name != "p":
        return False

    bold_children = node.find_all(["b", "strong"], recursive=False)
    if len(bold_children) != 1:
        return False

    paragraph_text = _normalize_text(node.get_text(" ", strip=True))
    bold_text = _normalize_text(bold_children[0].get_text(" ", strip=True))
    # Only treat as subtitle when the paragraph content is exactly the bold text.
    return bool(paragraph_text) and paragraph_text == bold_text


def clean_wp_content(raw_text: str) -> CleanedArticle:
    if not raw_text or not raw_text.strip():
        return CleanedArticle()

    no_caption = CAPTION_PATTERN.sub("", raw_text)
    soup = BeautifulSoup(no_caption, "html.parser")

    for tag_name in ("script", "style", "img"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    blocks: list[TextBlock] = []
    for node in soup.find_all(["h2", "h3", "h4", "p"]):
        text = _normalize_text(node.get_text(" ", strip=True))
        if not text:
            continue
        if node.name in {"h2", "h3", "h4"} or _is_bold_subtitle_paragraph(node):
            level = node.name if node.name in {"h2", "h3", "h4"} else "p-bold"
            blocks.append(TextBlock(kind="subtitle", text=text, level=level))
        else:
            blocks.append(TextBlock(kind="paragraph", text=text))

    if not blocks:
        paragraphs = _extract_plain_paragraphs(no_caption)
        blocks = [TextBlock(kind="paragraph", text=item) for item in paragraphs]
        return CleanedArticle(blocks=blocks, paragraphs=paragraphs, has_subtitles=False)

    paragraphs = [item.text for item in blocks if item.kind == "paragraph"]
    has_subtitles = any(item.kind == "subtitle" for item in blocks)
    return CleanedArticle(blocks=blocks, paragraphs=paragraphs, has_subtitles=has_subtitles)


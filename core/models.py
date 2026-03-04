from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Category(str, Enum):
    FINANCE = "财经"
    SPORTS = "体育"
    ENTERTAINMENT = "娱乐"
    CURRENT_AFFAIRS = "时事"
    TECHNOLOGY = "科技"
    OTHER = "其他"


class Confidence(str, Enum):
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


VALID_CATEGORIES = {item.value for item in Category}
VALID_CONFIDENCE = {item.value for item in Confidence}


def normalize_category(value: str) -> str:
    if value in VALID_CATEGORIES:
        return value
    return Category.OTHER.value


def normalize_confidence(value: str) -> str:
    if value in VALID_CONFIDENCE:
        return value
    return Confidence.LOW.value


def normalize_score(value: Any) -> int:
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(score, 100))


@dataclass
class TitleResult:
    primary: str
    secondary: list[str] = field(default_factory=list)
    confidence: str = Confidence.MEDIUM.value
    score: int = 50
    reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TitleResult":
        return cls(
            primary=normalize_category(str(data.get("primary", Category.OTHER.value))),
            secondary=[
                normalize_category(str(item))
                for item in data.get("secondary", [])
                if str(item).strip()
            ],
            confidence=normalize_confidence(str(data.get("confidence", Confidence.LOW.value))),
            score=normalize_score(data.get("score", 0)),
            reason=str(data.get("reason", "")).strip(),
        )


@dataclass
class ChunkResult:
    chunk_index: int
    chunk_type: str
    primary: str
    secondary: list[str] = field(default_factory=list)
    confidence: str = Confidence.MEDIUM.value
    score: int = 50
    key_signals: list[str] = field(default_factory=list)
    reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkResult":
        return cls(
            chunk_index=int(data.get("chunk_index", 0)),
            chunk_type=str(data.get("chunk_type", "paragraph")),
            primary=normalize_category(str(data.get("primary", Category.OTHER.value))),
            secondary=[
                normalize_category(str(item))
                for item in data.get("secondary", [])
                if str(item).strip()
            ],
            confidence=normalize_confidence(str(data.get("confidence", Confidence.LOW.value))),
            score=normalize_score(data.get("score", 0)),
            key_signals=[str(item).strip() for item in data.get("key_signals", []) if str(item).strip()],
            reason=str(data.get("reason", "")).strip(),
        )


@dataclass
class FinalResult:
    final_primary: str
    final_secondary: list[str] = field(default_factory=list)
    final_confidence: str = Confidence.MEDIUM.value
    final_score: int = 50
    evidence_summary: list[str] = field(default_factory=list)
    decision_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_primary": self.final_primary,
            "final_secondary": self.final_secondary,
            "final_confidence": self.final_confidence,
            "final_score": self.final_score,
            "evidence_summary": self.evidence_summary,
            "decision_reason": self.decision_reason,
        }


from __future__ import annotations

from collections import defaultdict

from core.models import (
    Category,
    ChunkResult,
    Confidence,
    FinalResult,
    TitleResult,
    normalize_confidence,
)


CONFIDENCE_WEIGHT = {
    Confidence.HIGH.value: 1.0,
    Confidence.MEDIUM.value: 0.6,
    Confidence.LOW.value: 0.25,
}


def _confidence_from_score(score: int, margin: float) -> str:
    if score >= 75 and margin >= 0.25:
        return Confidence.HIGH.value
    if score >= 55:
        return Confidence.MEDIUM.value
    return Confidence.LOW.value


def aggregate_results(
    title_result: TitleResult, chunk_results: list[ChunkResult], title_weight: float = 0.4
) -> FinalResult:
    title_weight = max(0.2, min(title_weight, 0.8))
    scoreboard: dict[str, float] = defaultdict(float)
    evidence: list[str] = []

    title_conf_weight = CONFIDENCE_WEIGHT.get(normalize_confidence(title_result.confidence), 0.6)
    scoreboard[title_result.primary] += title_weight * title_conf_weight
    for item in title_result.secondary:
        scoreboard[item] += title_weight * 0.35
    if title_result.reason:
        evidence.append(f"标题依据：{title_result.reason}")

    if chunk_results:
        per_chunk_weight = (1.0 - title_weight) / len(chunk_results)
        for result in chunk_results:
            conf_weight = CONFIDENCE_WEIGHT.get(normalize_confidence(result.confidence), 0.25)
            primary_gain = per_chunk_weight * conf_weight
            scoreboard[result.primary] += primary_gain
            for sec in result.secondary:
                scoreboard[sec] += primary_gain * 0.35

            if result.key_signals:
                evidence.append(f"分块{result.chunk_index}信号：{', '.join(result.key_signals[:3])}")
            elif result.reason:
                evidence.append(f"分块{result.chunk_index}依据：{result.reason}")

    if not scoreboard:
        return FinalResult(
            final_primary=Category.OTHER.value,
            final_secondary=[],
            final_confidence=Confidence.LOW.value,
            final_score=0,
            evidence_summary=["无有效证据"],
            decision_reason="未获得可用分类信息，归类为其他。",
        )

    ranking = sorted(scoreboard.items(), key=lambda item: item[1], reverse=True)
    primary, primary_value = ranking[0]
    secondary = [item[0] for item in ranking[1:3] if item[1] >= primary_value * 0.55]

    total = sum(scoreboard.values()) or 1.0
    final_score = int(round(primary_value / total * 100))
    second_value = ranking[1][1] if len(ranking) > 1 else 0.0
    margin = max(0.0, primary_value - second_value)
    final_confidence = _confidence_from_score(final_score, margin)

    evidence_summary = evidence[:5] if evidence else ["无显式证据摘要"]
    decision_reason = (
        f"主分类“{primary}”在加权投票中得分最高；"
        f"标题高权重且低置信度分块已降权处理。"
    )
    return FinalResult(
        final_primary=primary,
        final_secondary=secondary,
        final_confidence=final_confidence,
        final_score=final_score,
        evidence_summary=evidence_summary,
        decision_reason=decision_reason,
    )


from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, List

from ..agent.verification import VerificationReport


@dataclass
class ExampleMetrics:
    question: str
    expected_answers: List[str]
    model_answer: str
    accuracy: float
    citation_coverage: float
    hallucination: float
    latency_ms: float
    verification_status: str


@dataclass
class EvalSummary:
    total_examples: int
    mean_accuracy: float
    mean_citation_coverage: float
    hallucination_rate: float
    avg_latency_ms: float
    p95_latency_ms: float


def compute_accuracy(answer: str, expected_answers: Iterable[str]) -> float:
    normalized = (answer or "").lower()
    for expected in expected_answers:
        if expected and expected.lower() in normalized:
            return 1.0
    return 0.0


def compute_citation_coverage(verification: VerificationReport) -> float:
    if verification.overall_status == "no_citations":
        return 0.0
    if verification.missing_tags:
        return 0.0
    return 1.0


def compute_hallucination_flag(verification: VerificationReport) -> float:
    if verification.overall_status in {"failed", "no_citations"}:
        return 1.0
    if any(section.status == "llm_unsupported" for section in verification.sections):
        return 1.0
    return 0.0


def summarize_metrics(entries: List[ExampleMetrics]) -> EvalSummary:
    if not entries:
        return EvalSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _mean(values: Iterable[float]) -> float:
        values = list(values)
        return float(statistics.mean(values)) if values else 0.0

    latencies = [entry.latency_ms for entry in entries]
    p95 = _percentile(latencies, 0.95)

    return EvalSummary(
        total_examples=len(entries),
        mean_accuracy=_mean(entry.accuracy for entry in entries),
        mean_citation_coverage=_mean(entry.citation_coverage for entry in entries),
        hallucination_rate=_mean(entry.hallucination for entry in entries),
        avg_latency_ms=_mean(latencies),
        p95_latency_ms=p95,
    )


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * percentile
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)

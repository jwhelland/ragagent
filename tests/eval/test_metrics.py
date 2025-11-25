from __future__ import annotations

from ragagent.agent.verification import SectionReport, VerificationReport
from ragagent.eval.metrics import (
    ExampleMetrics,
    compute_accuracy,
    compute_citation_coverage,
    compute_hallucination_flag,
    summarize_metrics,
)


def dummy_verification(
    status: str = "supported",
    missing: list[str] | None = None,
    sections: list[SectionReport] | None = None,
) -> VerificationReport:
    return VerificationReport(
        overall_status=status,
        cited_tags=["S1"],
        missing_tags=missing or [],
        uncited_tags=[],
        sections=sections or [],
    )


def test_compute_accuracy_matches_expected():
    answer = "Hybrid retrieval merges vector and graph context."
    expected = ["graph context"]
    assert compute_accuracy(answer, expected) == 1.0
    assert compute_accuracy(answer, ["table only"]) == 0.0


def test_citation_coverage_requires_tags():
    ver = dummy_verification(status="supported")
    assert compute_citation_coverage(ver) == 1.0

    missing = dummy_verification(status="failed", missing=["S2"])
    assert compute_citation_coverage(missing) == 0.0


def test_hallucination_flag_detects_failures():
    ver = dummy_verification(status="failed")
    assert compute_hallucination_flag(ver) == 1.0

    weak_section = SectionReport(tag="S1", status="llm_unsupported")
    flagged = dummy_verification(status="attention", sections=[weak_section])
    assert compute_hallucination_flag(flagged) == 1.0


def test_summarize_metrics_handles_small_samples():
    entries = [
        ExampleMetrics(
            question="Q1",
            expected_answers=["A"],
            model_answer="A",
            accuracy=1.0,
            citation_coverage=1.0,
            hallucination=0.0,
            latency_ms=100.0,
            verification_status="supported",
        ),
        ExampleMetrics(
            question="Q2",
            expected_answers=["B"],
            model_answer="X",
            accuracy=0.0,
            citation_coverage=0.0,
            hallucination=1.0,
            latency_ms=200.0,
            verification_status="failed",
        ),
    ]
    summary = summarize_metrics(entries)
    assert summary.total_examples == 2
    assert summary.mean_accuracy == 0.5
    assert summary.hallucination_rate == 0.5
    assert summary.avg_latency_ms == 150.0

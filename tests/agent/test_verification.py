from __future__ import annotations

from ragagent.agent.verification import AnswerVerifier, _extract_verdict_label
from ragagent.retrieval.models import ContextBundle, ContextChunk, RetrievedChunk


def build_context(snippets: list[str]) -> ContextBundle:
    chunks: list[ContextChunk] = []
    citation_map = {}
    formatted_sections = []
    for idx, snippet in enumerate(snippets, start=1):
        tag = f"S{idx}"
        chunk = RetrievedChunk(
            chunk_id=f"doc{idx}:1",
            doc_id=f"doc{idx}",
            text=snippet,
            score=1.0,
            page=1,
        )
        chunks.append(ContextChunk(tag=tag, chunk=chunk, snippet=snippet))
        citation_map[tag] = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "page": chunk.page,
            "table_id": chunk.table_id,
            "origin": chunk.origin,
        }
        formatted_sections.append(f"[{tag}] {snippet}")
    return ContextBundle(
        chunks=chunks,
        formatted="\n\n".join(formatted_sections),
        citation_map=citation_map,
    )


def test_verify_supported_citation():
    context = build_context(
        ["Hybrid retrieval blends semantic search with graph expansion workflows."]
    )
    verifier = AnswerVerifier()

    answer = "This agent relies on hybrid retrieval that mixes dense search with graph context [S1]."
    report = verifier.verify(answer, context)

    assert report.overall_status == "supported"
    assert report.sections[0].status == "supported"


def test_verify_missing_citation_detected():
    context = build_context(["Chunking keeps 1200 character windows for doc fidelity."])
    verifier = AnswerVerifier()

    answer = "Tables are stored separately for better grounding [S2]."
    report = verifier.verify(answer, context)

    assert report.overall_status == "failed"
    assert "S2" in report.missing_tags
    assert report.sections[0].status == "missing"


def test_verify_weak_overlap_marks_attention():
    context = build_context(
        [
            "Financial statements include net income, cash flow, and EBITDA figures for 2024."
        ]
    )
    verifier = AnswerVerifier(min_overlap_tokens=2)

    answer = "The propulsion system relies on solid rocket boosters [S1]."
    report = verifier.verify(answer, context)

    assert report.overall_status == "attention"
    assert report.sections[0].status in {"weak", "llm_unsupported"}


def test_verify_handles_no_citations():
    context = build_context(["Neo4j tracks entities connected to chunk IDs."])
    verifier = AnswerVerifier()

    answer = "The answer forgets to cite evidence."
    report = verifier.verify(answer, context)

    assert report.overall_status == "no_citations"
    assert report.sections == []


def test_llm_verdict_supported_overrides_weak(monkeypatch):
    context = build_context(["Financial results cover revenue and profit."])
    verifier = AnswerVerifier(min_overlap_tokens=5, enable_llm=True)
    verifier._client = object()  # type: ignore[assignment]
    monkeypatch.setattr(
        verifier,
        "_llm_verify",
        lambda *args, **kwargs: "VERDICT: SUPPORTED - overlap okay",
    )

    answer = "Spacecraft propulsion relies on xenon thrusters [S1]."
    report = verifier.verify(answer, context)

    assert report.sections[0].status == "supported"


def test_llm_verdict_unsupported_marks_attention(monkeypatch):
    context = build_context(["Financial results cover revenue and profit."])
    verifier = AnswerVerifier(min_overlap_tokens=5, enable_llm=True)
    verifier._client = object()  # type: ignore[assignment]
    monkeypatch.setattr(
        verifier,
        "_llm_verify",
        lambda *args, **kwargs: "VERDICT: UNSUPPORTED - no evidence",
    )

    answer = "Spacecraft propulsion relies on xenon thrusters [S1]."
    report = verifier.verify(answer, context)

    assert report.sections[0].status == "llm_unsupported"


def test_extract_verdict_label_parses_prefix():
    assert _extract_verdict_label("VERDICT: SUPPORTED - ok") == "SUPPORTED"
    assert _extract_verdict_label("VERDICT: unsupported - nope") == "UNSUPPORTED"
    assert _extract_verdict_label("SUPPORTED") == "SUPPORTED"

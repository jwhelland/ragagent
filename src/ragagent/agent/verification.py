from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set

from openai import OpenAI

from ..logging_setup import get_logger
from ..retrieval.models import ContextBundle, ContextChunk


logger = get_logger(__name__)

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")

SectionStatus = Literal["supported", "missing", "weak", "llm_unsupported"]
OverallStatus = Literal["supported", "attention", "failed", "no_citations"]


@dataclass
class SectionReport:
    tag: str
    status: SectionStatus
    overlap_ratio: float | None = None
    overlap_tokens: int = 0
    issues: List[str] = field(default_factory=list)
    llm_verdict: Optional[str] = None


@dataclass
class VerificationReport:
    overall_status: OverallStatus
    cited_tags: List[str] = field(default_factory=list)
    missing_tags: List[str] = field(default_factory=list)
    uncited_tags: List[str] = field(default_factory=list)
    sections: List[SectionReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall_status": self.overall_status,
            "cited_tags": self.cited_tags,
            "missing_tags": self.missing_tags,
            "uncited_tags": self.uncited_tags,
            "sections": [
                {
                    "tag": sec.tag,
                    "status": sec.status,
                    "overlap_ratio": sec.overlap_ratio,
                    "overlap_tokens": sec.overlap_tokens,
                    "issues": sec.issues,
                    "llm_verdict": sec.llm_verdict,
                }
                for sec in self.sections
            ],
        }


class AnswerVerifier:
    def __init__(
        self,
        *,
        min_overlap_tokens: int = 2,
        min_overlap_ratio: float = 0.08,
        enable_llm: bool = False,
        openai_client: OpenAI | None = None,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self._min_overlap_tokens = min_overlap_tokens
        self._min_overlap_ratio = min_overlap_ratio
        self._enable_llm = enable_llm
        self._client = openai_client
        self._llm_model = llm_model

    def verify(self, answer: str, context: ContextBundle) -> VerificationReport:
        cited_tags = self._extract_tags(answer)
        if not cited_tags:
            return VerificationReport(
                overall_status="no_citations",
                cited_tags=[],
                uncited_tags=sorted(context.citation_map.keys()),
                sections=[],
            )

        tag_lookup = {chunk.tag: chunk for chunk in context.chunks}
        missing_tags: List[str] = []
        sections: List[SectionReport] = []
        answer_tokens = self._tokenize(answer)

        for tag in cited_tags:
            if tag not in context.citation_map:
                missing = SectionReport(tag=tag, status="missing", issues=["citation_not_in_context"])
                sections.append(missing)
                missing_tags.append(tag)
                continue
            chunk = tag_lookup.get(tag)
            if chunk is None:
                missing = SectionReport(tag=tag, status="missing", issues=["context_missing_snippet"])
                sections.append(missing)
                missing_tags.append(tag)
                continue
            overlap_tokens, overlap_ratio = self._compute_overlap(answer_tokens, chunk.snippet)
            if overlap_tokens >= self._min_overlap_tokens or overlap_ratio >= self._min_overlap_ratio:
                sections.append(
                    SectionReport(
                        tag=tag,
                        status="supported",
                        overlap_ratio=overlap_ratio,
                        overlap_tokens=overlap_tokens,
                    )
                )
            else:
                section = SectionReport(
                    tag=tag,
                    status="weak",
                    overlap_ratio=overlap_ratio,
                    overlap_tokens=overlap_tokens,
                    issues=["insufficient_term_overlap"],
                )
                if self._enable_llm and self._client:
                    llm_verdict = self._llm_verify(chunk, answer, tag)
                    if llm_verdict:
                        section.llm_verdict = llm_verdict
                        verdict_label = _extract_verdict_label(llm_verdict)
                        if verdict_label == "SUPPORTED":
                            section.status = "supported"
                            section.issues.clear()
                        elif verdict_label == "UNSUPPORTED":
                            section.status = "llm_unsupported"
                            section.issues.append("llm_marked_unsupported")
                sections.append(section)

        uncited_tags = sorted(set(context.citation_map.keys()) - set(cited_tags))
        if missing_tags:
            overall: OverallStatus = "failed"
        elif any(sec.status != "supported" for sec in sections):
            overall = "attention"
        else:
            overall = "supported"

        return VerificationReport(
            overall_status=overall,
            cited_tags=cited_tags,
            missing_tags=missing_tags,
            uncited_tags=uncited_tags,
            sections=sections,
        )

    def _extract_tags(self, answer: str) -> List[str]:
        matches = [f"S{match}" for match in CITATION_PATTERN.findall(answer or "")]
        # Preserve order but remove duplicates
        seen: Set[str] = set()
        ordered: List[str] = []
        for tag in matches:
            if tag not in seen:
                seen.add(tag)
                ordered.append(tag)
        return ordered

    def _tokenize(self, text: str) -> Set[str]:
        return set(TOKEN_PATTERN.findall((text or "").lower()))

    def _compute_overlap(self, answer_tokens: Set[str], snippet: str) -> tuple[int, float]:
        snippet_tokens = self._tokenize(snippet)
        if not snippet_tokens:
            return 0, 0.0
        overlap = answer_tokens & snippet_tokens
        ratio = len(overlap) / max(len(snippet_tokens), 1)
        return len(overlap), ratio

    def _llm_verify(self, chunk: ContextChunk, answer: str, tag: str) -> Optional[str]:
        if not self._client:
            logger.warning("llm_verification_requested_without_client")
            return None
        excerpt = self._extract_relevant_answer(answer, tag)
        prompt = (
            "You verify whether the provided evidence snippet supports the answer text.\n"
            "Respond with a single line formatted as:\n"
            "VERDICT: <SUPPORTED|INCONCLUSIVE|UNSUPPORTED> - <short reason>."
        )
        user_content = (
            f"Evidence snippet:\n{chunk.snippet}\n\n"
            f"Answer excerpt citing [{tag}]:\n{excerpt}"
        )
        try:
            completion = self._client.chat.completions.create(
                model=self._llm_model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_verification_failed", error=str(exc))
            return None
        message = completion.choices[0].message.content or ""
        return message.strip()

    def _extract_relevant_answer(self, answer: str, tag: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        tagged = [s.strip() for s in sentences if f"[{tag}]" in s]
        if tagged:
            return " ".join(tagged)
        return answer


def _extract_verdict_label(verdict: str) -> Optional[str]:
    if not verdict:
        return None
    prefix = verdict.split("-", 1)[0].strip()
    if ":" in prefix:
        prefix = prefix.split(":", 1)[1].strip()
    normalized = prefix.upper()
    if normalized in {"SUPPORTED", "UNSUPPORTED", "INCONCLUSIVE"}:
        return normalized
    return None

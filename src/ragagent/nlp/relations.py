from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from openai import OpenAI


# Core relation types (can be extended)
DEFAULT_RELATION_TYPES: Set[str] = {
    "CAUSES",  # Component CAUSES Symptom
    "REQUIRES",  # Step/Action REQUIRES Tool/Prerequisite
    "PART_OF",  # Subcomponent PART_OF Assembly
    "ALTERNATIVE",  # PartX ALTERNATIVE PartY (replacement)
    "LOCATED_IN",  # Component LOCATED_IN Area/Compartment
    "RESOLVES",  # Action RESOLVES Symptom
    "PREVENTS",  # Action PREVENTS Issue
    "SYMPTOM_OF",  # Symptom SYMPTOM_OF Component/Issue
}


@dataclass
class RelationTriple:
    source: str
    type: str
    target: str
    confidence: float = 0.6
    justification: str = ""


class LLMRelationExtractor:
    """
    LLM-backed relation extractor producing typed triples from a chunk.

    Usage:
        extractor = LLMRelationExtractor()
        triples = extractor.extract(chunk_text, entities)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_relations: int = 8,
        allowed_types: Optional[Iterable[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = model
        self._temperature = temperature
        self._max_relations = max_relations
        self._allowed_types: Set[str] = (
            set(allowed_types) if allowed_types else set(DEFAULT_RELATION_TYPES)
        )
        self._system_prompt = system_prompt or (
            "You extract precise, grounded relations from technical manuals. "
            "Return only valid relations that are explicitly supported by the text."
        )
        self._client: Optional[OpenAI] = None

    def _client_lazy(self) -> OpenAI:
        if self._client is None:
            # If api key is not provided, OpenAI client will read env var if available
            self._client = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
        return self._client

    def extract(self, text: str, entities: List[str]) -> List[RelationTriple]:
        """
        Run an LLM call to extract relation triples from the chunk.

        Args:
            text: Chunk text
            entities: Pre-extracted entities (strings). Use as candidate nodes.

        Returns:
            A list of validated, de-duplicated RelationTriple items.
        """
        if not text or not text.strip():
            return []

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": self._build_prompt(text=text, entities=entities),
            },
        ]
        try:
            resp = self._client_lazy().chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=messages,
            )
            content = (resp.choices[0].message.content or "").strip()
            data = _parse_json_payload(content)
            triples = self._coerce_triples(data)
            triples = self._validate_and_dedupe(triples)
            return triples[: self._max_relations]
        except Exception:
            # Be conservative on ingest failures
            return []

    def _build_prompt(self, *, text: str, entities: List[str]) -> str:
        allowed = ", ".join(sorted(self._allowed_types))
        entity_block = (
            "\n".join(f"- {e}" for e in entities[:50])
            if entities
            else "(none detected)"
        )
        schema = """
Return a JSON object only, no explanation, matching this schema exactly:

{
  "relations": [
    {
      "source": "string (use span or entity from text)",
      "type": "one of: ALTERNATIVE | CAUSES | LOCATED_IN | PART_OF | PREVENTS | REQUIRES | RESOLVES | SYMPTOM_OF",
      "target": "string (use span or entity from text)",
      "confidence": 0.0-1.0,
      "justification": "short quote or sentence from the text"
    }
  ]
}
""".strip()

        instructions = f"""
Extract explicit relations present in the text. Only include relations that are clearly supported by the wording.

Allowed relation types: {allowed}

If nothing is supported, return {{"relations":[]}}.

Entities (hints, optional): 
{entity_block}

Text:
\"\"\"{text.strip()[:8000]}\"\"\"

{schema}
""".strip()
        return instructions

    def _coerce_triples(self, data: Dict[str, Any]) -> List[RelationTriple]:
        rels = data.get("relations") if isinstance(data, dict) else None
        if not isinstance(rels, list):
            return []
        triples: List[RelationTriple] = []
        for item in rels:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source", "")).strip()
            typ = str(item.get("type", "")).strip().upper()
            tgt = str(item.get("target", "")).strip()
            try:
                conf = float(item.get("confidence", 0.6))
            except Exception:
                conf = 0.6
            just = str(item.get("justification", "")).strip()
            if not src or not tgt:
                continue
            triples.append(
                RelationTriple(
                    source=src,
                    type=typ,
                    target=tgt,
                    confidence=conf,
                    justification=just,
                )
            )
        return triples

    def _validate_and_dedupe(
        self, triples: List[RelationTriple]
    ) -> List[RelationTriple]:
        seen: Set[Tuple[str, str, str]] = set()
        out: List[RelationTriple] = []
        for t in triples:
            if t.type not in self._allowed_types:
                continue
            # normalize confidence
            if not (0.0 <= t.confidence <= 1.0):
                t.confidence = max(0.0, min(1.0, float(t.confidence or 0.6)))
            key = (t.source.lower().strip(), t.type, t.target.lower().strip())
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out


# Light-weight trigger to gate LLM usage on likely-high-value chunks
_TRIGGER_PATTERNS = [
    r"\berror\b",
    r"\bfault\b",
    r"\bfailure\b",
    r"\bwarning\b",
    r"\btroubleshoot(ing)?\b",
    r"\breplace(d|ment)?\b",
    r"\brequires?\b",
    r"\bsymptom(s)?\b",
    r"\bleak(s|age)?\b",
    r"\bnoise\b",
    r"\breset\b",
    r"\bcode\b",
    r"\berr( |\-)?\d{2,4}\b",
    r"\b(e|f)\-?\d{2,4}\b",
]

_TRIGGER_REGEXES = [re.compile(p, re.IGNORECASE) for p in _TRIGGER_PATTERNS]


def is_trigger_chunk(text: str) -> bool:
    """
    Return True if chunk text likely contains troubleshooting/relations content.
    """
    if not text:
        return False
    t = text[:8000]
    for rx in _TRIGGER_REGEXES:
        if rx.search(t):
            return True
    # Also trigger if the chunk has many imperative verbs; heuristic: presence of "must", "should", "ensure"
    if re.search(r"\b(must|should|ensure|verify)\b", t, re.IGNORECASE):
        return True
    return False


def _parse_json_payload(s: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction:
    - Prefer fenced code block ```json ... ```
    - Else take the first balanced-looking JSON object
    """
    if not s:
        return {"relations": []}

    # Look for ```json ... ```
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE
    )
    candidate = fenced.group(1) if fenced else None

    if not candidate:
        # Fallback: find first '{' and last '}' and try to parse
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start : end + 1]

    if not candidate:
        return {"relations": []}

    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {"relations": []}

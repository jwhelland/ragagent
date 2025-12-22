"""spaCy-based entity extractor with domain-specific patterns.

This module wraps a spaCy pipeline with an EntityRuler and a domain term matcher
to recognize both general and domain-specific entities. Entities are returned
with spans, context, and heuristic confidence scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import spacy
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from src.utils.config import SpacyConfig

# Extend Span with metadata about the match source.
Span.set_extension("source", default="model", force=True)
Span.set_extension("is_domain_match", default=False, force=True)


class ExtractedEntity(BaseModel):
    """Structured representation of an extracted entity."""

    model_config = ConfigDict(extra="forbid")

    text: str
    label: str
    confidence: float
    start_char: int
    end_char: int
    sentence: str
    context: str
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    source: str = "spacy"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpacyExtractor:
    """spaCy NER wrapper with custom patterns and confidence scoring."""

    def __init__(
        self,
        config: Optional[SpacyConfig] = None,
        nlp: Optional[Language] = None,
    ) -> None:
        self.config = config or SpacyConfig()
        self.nlp: Language = nlp or self._load_model(self.config.model)

        if nlp is None and "ner" not in self.nlp.pipe_names:
            self.nlp.add_pipe("ner", last=True)

        self.ruler = self._add_entity_ruler(self.config.custom_patterns)
        self.domain_matcher, self.domain_match_id_map = self._build_domain_matcher()

        logger.info(
            "Initialized SpacyExtractor",
            model=self.config.model,
            patterns=self.config.custom_patterns,
        )

    def extract_from_chunks(self, chunks: Iterable[Any]) -> Dict[str | None, List[ExtractedEntity]]:
        """Run NER over a collection of chunks.

        Args:
            chunks: Iterable of chunk-like objects. Each must expose a `content`
                attribute or key. Optional metadata: `chunk_id`, `document_id`.

        Returns:
            Mapping of chunk_id -> list of ExtractedEntity (chunk_id None if missing).
        """
        texts: List[str] = []
        metadata: List[Tuple[Optional[str], Optional[str]]] = []
        for chunk in chunks:
            text = getattr(chunk, "content", None) or chunk.get("content")
            if text is None:
                logger.warning("Skipping chunk without content", chunk=chunk)
                continue

            chunk_id = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None)
            if isinstance(chunk, dict):
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
            document_id = getattr(chunk, "document_id", None)
            if isinstance(chunk, dict):
                document_id = document_id or chunk.get("document_id")

            texts.append(text)
            metadata.append((chunk_id, document_id))

        results: Dict[str | None, List[ExtractedEntity]] = {}
        for doc, (chunk_id, document_id) in zip(
            self.nlp.pipe(texts, batch_size=self.config.batch_size), metadata
        ):
            entities = self._collect_entities(doc, chunk_id, document_id)
            results.setdefault(chunk_id, []).extend(entities)

        return results

    def _load_model(self, model_name: str) -> Language:
        """Load spaCy model with minimal validation."""
        try:
            return spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
                f"Install it with `python -m spacy download {model_name}`."
            ) from exc

    def _add_entity_ruler(self, patterns_path: str) -> EntityRuler:
        """Attach an EntityRuler with domain patterns."""
        before_component = "ner" if "ner" in self.nlp.pipe_names else None
        ruler = self.nlp.add_pipe(
            "entity_ruler",
            name="domain_entity_ruler",
            before=before_component,
            config={"validate": True},
        )

        path = Path(patterns_path)
        if not path.exists():
            logger.warning("Custom patterns file not found", path=str(path))
            return ruler

        patterns = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pattern_dict = json.loads(line)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to parse pattern line", line=line, error=str(exc))
                    continue
                # Mark EntityRuler matches via ent_id_
                label = pattern_dict.get("label")
                if label:
                    pattern_dict["id"] = f"pattern::{label}"
                patterns.append(pattern_dict)

        ruler.add_patterns(patterns)
        return ruler

    def _build_domain_matcher(self) -> Tuple[PhraseMatcher, Dict[int, str]]:
        """Create a PhraseMatcher for common domain terms."""
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        label_map: Dict[int, str] = {}

        domain_terms: Dict[str, Sequence[str]] = {
            "SYSTEM": [
                "bus",
                "system bus",
                "control system",
                "data acquisition system",
            ],
            "SUBSYSTEM": [
                "power subsystem",
                "thermal control subsystem",
                "communication subsystem",
                "data handling subsystem",
                "safety subsystem",
            ],
            "COMPONENT": [
                "motor controller",
                "sensor module",
                "temperature sensor",
                "power supply unit",
                "battery management unit",
                "embedded controller",
            ],
            "PARAMETER": [
                "state of charge",
                "bus voltage",
                "angular rate",
                "propellant pressure",
            ],
            "PROCEDURE": [
                "startup procedure",
                "shutdown procedure",
                "calibration procedure",
            ],
        }

        for label, terms in domain_terms.items():
            patterns = [self.nlp.make_doc(term) for term in terms]
            match_id = self.nlp.vocab.strings[label]
            matcher.add(label, [*patterns])
            label_map[match_id] = label

        return matcher, label_map

    def _collect_entities(
        self, doc: Doc, chunk_id: Optional[str], document_id: Optional[str]
    ) -> List[ExtractedEntity]:
        """Convert spaCy Doc entities to ExtractedEntity list with scoring."""
        collected: List[ExtractedEntity] = []

        spans = self._merge_domain_spans(doc)

        for ent in spans:
            source = self._infer_source(ent)
            context = self._context_window(doc.text, ent.start_char, ent.end_char, window=80)
            confidence = self._score_entity(ent, source, context)
            if confidence < self.config.confidence_threshold:
                continue

            metadata = {
                "source": source,
                "is_domain_match": ent._.is_domain_match,
                "ent_id": ent.ent_id_,
            }

            collected.append(
                ExtractedEntity(
                    text=ent.text,
                    label=ent.label_,
                    confidence=confidence,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    sentence=ent.sent.text.strip(),
                    context=context,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source="spacy",
                    metadata=metadata,
                )
            )

        return collected

    def _merge_domain_spans(self, doc: Doc) -> Tuple[Span, ...]:
        """Merge model/ruler entities with domain matcher spans."""
        matches = self.domain_matcher(doc)
        spans: List[Span] = []
        for match_id, start, end in matches:
            label = self.domain_match_id_map.get(match_id)
            if not label:
                continue
            span = Span(doc, start, end, label=label)
            span._.source = "domain"
            span._.is_domain_match = True
            spans.append(span)

        return tuple(filter_spans(list(doc.ents) + spans))

    def _infer_source(self, ent: Span) -> str:
        """Determine extraction source for a span."""
        if ent._.is_domain_match:
            return "domain_matcher"
        if ent.ent_id_ and ent.ent_id_.startswith("pattern::"):
            return "pattern"
        return ent._.source or "model"

    def _score_entity(self, ent: Span, source: str, context: str) -> float:
        """Heuristic confidence scoring bounded to [0, 1]."""
        base_scores = {
            "pattern": 0.72,
            "domain_matcher": 0.68,
            "model": 0.55,
        }
        base = base_scores.get(source, 0.60)

        length_boost = min(len(ent.text) / 50, 0.10)

        salient_tokens = ["system", "subsystem", "module", "unit", "procedure", "mode"]
        has_salience = any(token in context.lower() for token in salient_tokens)
        context_boost = 0.05 if has_salience else 0.0

        type_boost = 0.05 if ent.label_ in {"SYSTEM", "SUBSYSTEM", "COMPONENT"} else 0.0
        numeric_penalty = (
            -0.05
            if ent.label_ not in {"PARAMETER", "ANOMALY"}
            and any(char.isdigit() for char in ent.text)
            else 0.0
        )

        score = base + length_boost + context_boost + type_boost + numeric_penalty
        return max(0.0, min(score, 1.0))

    def _context_window(self, text: str, start: int, end: int, window: int = 80) -> str:
        """Return text snippet around an entity span."""
        prefix_start = max(start - window, 0)
        suffix_end = min(end + window, len(text))
        return text[prefix_start:start] + text[start:end] + text[end:suffix_end]

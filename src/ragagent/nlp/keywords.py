from __future__ import annotations

import os
from typing import List, Tuple

import spacy
from spacy.language import Language


_SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
_NLP: Language | None = None


def _get_nlp() -> Language:
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load(_SPACY_MODEL)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"spaCy model '{_SPACY_MODEL}' not available. Install it with: python -m spacy download {_SPACY_MODEL}"
            ) from e
    return _NLP


def extract_entities_and_phrases(text: str) -> Tuple[List[str], List[str]]:
    nlp = _get_nlp()
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    phrases = [nc.text for nc in getattr(doc, "noun_chunks", [])]
    if not phrases:
        phrases = list({t.lemma_ for t in doc if t.pos_ in {"NOUN", "PROPN"} and len(t) > 2})
    # De-duplicate while preserving order
    seen_e, seen_p = set(), set()
    uniq_entities, uniq_phrases = [], []
    for e in entities:
        if e not in seen_e:
            seen_e.add(e)
            uniq_entities.append(e)
    for p in phrases:
        if p not in seen_p:
            seen_p.add(p)
            uniq_phrases.append(p)
    return uniq_entities[:20], uniq_phrases[:20]


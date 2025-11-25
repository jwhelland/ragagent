from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    page: Optional[int] = None
    table_id: Optional[str] = None
    source_path: Optional[str] = None
    origin: str = "vector"
    seed_chunk_id: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorQueryResult:
    query: str
    embedding: Sequence[float]
    chunks: List[RetrievedChunk]


@dataclass
class ContextChunk:
    tag: str
    chunk: RetrievedChunk
    snippet: str


@dataclass
class ContextBundle:
    chunks: List[ContextChunk]
    formatted: str
    citation_map: Dict[str, Dict[str, Any]]

    def as_prompt_section(self) -> str:
        return self.formatted

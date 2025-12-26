"""Shared models for retrieval system."""

from __future__ import annotations

import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

from src.retrieval.graph_retriever import GraphPath


class RetrievalStrategy(str, Enum):
    """Hybrid retrieval strategy types."""

    VECTOR_ONLY = "vector_only"  # Use only vector search
    GRAPH_ONLY = "graph_only"  # Use only graph search
    HYBRID_PARALLEL = "hybrid_parallel"  # Execute both in parallel
    VECTOR_FIRST = "vector_first"  # Vector search, then graph for expansion
    GRAPH_FIRST = "graph_first"  # Graph search, then vector for content


class HybridChunk(BaseModel):
    """Unified chunk representation for hybrid retrieval."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Source document ID")
    content: str = Field(..., description="Chunk text content")
    level: int = Field(..., ge=1, le=4, description="Hierarchy level")

    # Scoring
    vector_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Vector similarity score"
    )
    graph_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Graph relevance score")
    entity_coverage_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Entity coverage score"
    )
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Entity confidence")
    diversity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Diversity score")
    final_score: float = Field(..., ge=0.0, le=1.0, description="Final fused score")

    # Metadata
    rank: int = Field(..., ge=1, description="Final rank in results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    entity_ids: List[str] = Field(default_factory=list, description="Entity IDs in chunk")
    graph_paths: List[str] = Field(
        default_factory=list, description="Graph path IDs this chunk appears in"
    )
    source: str = Field(..., description="Source retriever (vector, graph, or hybrid)")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class GeneratedResponse(BaseModel):
    """Structured representation of a generated response."""

    model_config = ConfigDict(extra="allow")

    answer: str = Field(..., description="Generated natural language answer")
    query_id: str = Field(..., description="Original query ID")
    chunks_used: List[str] = Field(default_factory=list, description="IDs of chunks used")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    timestamp: float = Field(default_factory=time.time, description="Generation timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class HybridRetrievalResult(BaseModel):
    """Result of hybrid retrieval operation."""

    model_config = ConfigDict(extra="allow")

    query_id: str = Field(..., description="Query identifier")
    query_text: str = Field(..., description="Original query text")
    strategy_used: RetrievalStrategy = Field(..., description="Retrieval strategy used")
    chunks: List[HybridChunk] = Field(default_factory=list, description="Retrieved chunks")
    graph_paths: List[GraphPath] = Field(
        default_factory=list, description="Graph paths (if applicable)"
    )
    answer: Optional[GeneratedResponse] = Field(
        None, description="Generated natural language answer"
    )

    # Statistics
    total_results: int = Field(..., ge=0, description="Total results before reranking")
    vector_results: int = Field(default=0, ge=0, description="Results from vector search")
    graph_results: int = Field(default=0, ge=0, description="Results from graph search")
    merged_results: int = Field(default=0, ge=0, description="Results after merging")

    # Timing
    retrieval_time_ms: float = Field(..., ge=0.0, description="Total retrieval time")
    vector_time_ms: Optional[float] = Field(None, ge=0.0, description="Vector retrieval time")
    graph_time_ms: Optional[float] = Field(None, ge=0.0, description="Graph retrieval time")
    merge_time_ms: Optional[float] = Field(None, ge=0.0, description="Merge and rerank time")

    # Metadata
    vector_success: bool = Field(default=True, description="Vector retrieval succeeded")
    graph_success: bool = Field(default=True, description="Graph retrieval succeeded")
    reranking_enabled: bool = Field(default=False, description="Reranking was applied")
    timestamp: datetime = Field(default_factory=datetime.now, description="Retrieval timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        data["strategy_used"] = self.strategy_used.value
        data["chunks"] = [c.to_dict() for c in self.chunks]
        data["graph_paths"] = [p.model_dump() for p in self.graph_paths]
        if self.answer:
            data["answer"] = self.answer.to_dict()
        return data

    def get_entity_ids(self) -> Set[str]:
        """Extract all unique entity IDs from results."""
        entity_ids: Set[str] = set()
        for chunk in self.chunks:
            entity_ids.update(chunk.entity_ids)
        return entity_ids

    def get_document_ids(self) -> Set[str]:
        """Get all unique document IDs from results."""
        return {chunk.document_id for chunk in self.chunks}


class SufficiencyResult(BaseModel):
    """Result of an information sufficiency check."""

    model_config = ConfigDict(frozen=True)

    is_sufficient: bool = Field(..., description="Whether gathered information is sufficient")
    missing_information: List[str] = Field(
        default_factory=list, description="List of missing information points"
    )
    reasoning: str = Field(..., description="Reasoning for the assessment")


class ResearchStep(BaseModel):
    """A single step in the iterative research process."""

    model_config = ConfigDict(frozen=True)

    step_number: int = Field(..., description="Step number (1-based)")
    timestamp: float = Field(default_factory=time.time, description="Step timestamp")
    sub_queries: List[str] = Field(
        default_factory=list, description="Sub-queries generated in this step"
    )
    sufficiency_check: Optional[SufficiencyResult] = Field(
        None, description="Sufficiency result at this step"
    )
    new_chunks_found: int = Field(default=0, description="Number of new chunks found")


class ResearchResult(BaseModel):
    """Result of a deep research operation."""

    model_config = ConfigDict(extra="allow")

    query_id: str = Field(..., description="Query identifier")
    original_query: str = Field(..., description="Original user query")
    final_answer: GeneratedResponse = Field(..., description="Final synthesized answer")
    steps: List[ResearchStep] = Field(default_factory=list, description="Research steps taken")
    total_chunks: int = Field(..., description="Total unique chunks accumulated")
    total_time_ms: float = Field(..., description="Total research time in milliseconds")
    accumulated_context: List[HybridChunk] = Field(
        default_factory=list, description="All accumulated chunks"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = self.model_dump()
        data["final_answer"] = self.final_answer.to_dict()
        data["accumulated_context"] = [c.to_dict() for c in self.accumulated_context]
        return data

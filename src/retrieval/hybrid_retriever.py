"""Hybrid retrieval combining vector and graph-based search (Task 4.4).

This module implements hybrid retrieval that combines:
- Vector-based semantic search (VectorRetriever)
- Graph-based relationship traversal (GraphRetriever)

Features:
- Parallel execution of both retrievers
- Intelligent result merging
- Configurable score fusion
- Diversity ranking
- Strategy selection based on query intent
- Graceful fallback when one retriever fails
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.retrieval.graph_retriever import GraphPath, GraphRetrievalResult, GraphRetriever
from src.retrieval.query_parser import ParsedQuery, QueryIntent
from src.retrieval.vector_retriever import RetrievalResult, RetrievedChunk, VectorRetriever
from src.storage.neo4j_manager import Neo4jManager
from src.utils.config import Config, HybridSearchConfig, RerankingConfig


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


class HybridRetriever:
    """Hybrid retriever combining vector and graph-based search."""

    def __init__(
        self,
        config: Optional[Config] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        neo4j_manager: Optional[Neo4jManager] = None,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            config: Configuration object
            vector_retriever: Vector retriever instance (created if None)
            graph_retriever: Graph retriever instance (created if None)
            neo4j_manager: Neo4j manager for fetching chunks (created if None)
        """
        self.config = config or Config.from_yaml()
        self.hybrid_config: HybridSearchConfig = self.config.retrieval.hybrid
        self.reranking_config: RerankingConfig = self.config.retrieval.reranking

        # Initialize retrievers
        self.vector_retriever = vector_retriever or VectorRetriever(config=self.config)
        self.graph_retriever = graph_retriever or GraphRetriever(config=self.config)

        # Initialize Neo4j for fetching chunks
        if neo4j_manager is None:
            self.neo4j = Neo4jManager(config=self.config.database)
            self.neo4j.connect()
        else:
            self.neo4j = neo4j_manager

        logger.info(
            "Initialized HybridRetriever",
            parallel_execution=self.hybrid_config.parallel_execution,
            strategy_selection=self.hybrid_config.strategy_selection,
            reranking_enabled=self.reranking_config.enabled,
        )

    def retrieve(
        self,
        query: ParsedQuery,
        strategy: Optional[RetrievalStrategy] = None,
        top_k: Optional[int] = None,
        timeout: float = 10.0,
    ) -> HybridRetrievalResult:
        """Retrieve relevant results using hybrid search.

        Args:
            query: ParsedQuery object from QueryParser
            strategy: Retrieval strategy (auto-selected if None)
            top_k: Number of final results to return (default from reranking config)
            timeout: Timeout for retrieval operations in seconds

        Returns:
            HybridRetrievalResult with merged and ranked results

        Raises:
            ValueError: If query is invalid
            TimeoutError: If retrieval exceeds timeout
        """
        start_time = time.time()

        # Select strategy if not specified
        if strategy is None:
            strategy = self._select_strategy(query)

        # Validate top_k
        top_k = top_k or self.reranking_config.max_results

        logger.info(
            "Starting hybrid retrieval",
            query_id=query.query_id,
            strategy=strategy.value,
            top_k=top_k,
        )

        # Execute retrieval based on strategy
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            result = self._retrieve_vector_only(query, top_k, timeout)
        elif strategy == RetrievalStrategy.GRAPH_ONLY:
            result = self._retrieve_graph_only(query, top_k, timeout)
        elif strategy == RetrievalStrategy.HYBRID_PARALLEL:
            result = self._retrieve_hybrid_parallel(query, top_k, timeout)
        elif strategy == RetrievalStrategy.VECTOR_FIRST:
            result = self._retrieve_vector_first(query, top_k, timeout)
        elif strategy == RetrievalStrategy.GRAPH_FIRST:
            result = self._retrieve_graph_first(query, top_k, timeout)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        result.retrieval_time_ms = total_time

        logger.info(
            "Hybrid retrieval completed",
            query_id=query.query_id,
            strategy=strategy.value,
            num_results=len(result.chunks),
            vector_success=result.vector_success,
            graph_success=result.graph_success,
            total_time_ms=round(total_time, 2),
        )

        return result

    def _select_strategy(self, query: ParsedQuery) -> RetrievalStrategy:
        """Select retrieval strategy based on query characteristics.

        Args:
            query: ParsedQuery object

        Returns:
            Selected retrieval strategy
        """
        # Check config strategy setting
        config_strategy = self.hybrid_config.strategy_selection

        if config_strategy != "auto":
            # Map config string to enum
            strategy_map = {
                "vector_first": RetrievalStrategy.VECTOR_FIRST,
                "graph_first": RetrievalStrategy.GRAPH_FIRST,
                "hybrid": RetrievalStrategy.HYBRID_PARALLEL,
            }
            return strategy_map.get(config_strategy, RetrievalStrategy.HYBRID_PARALLEL)

        # Auto-select based on query characteristics
        intent = query.intent
        has_entities = len(query.entity_mentions) > 0
        requires_graph = query.requires_graph_traversal

        # Semantic queries with no entities -> vector only
        if intent == QueryIntent.SEMANTIC and not has_entities:
            return RetrievalStrategy.VECTOR_ONLY

        # Structural queries with entities -> graph first
        if intent == QueryIntent.STRUCTURAL and has_entities:
            return RetrievalStrategy.GRAPH_FIRST

        # Procedural queries -> graph first (follow procedures)
        if intent == QueryIntent.PROCEDURAL:
            return RetrievalStrategy.GRAPH_FIRST

        # Hybrid queries or queries requiring graph -> parallel hybrid
        if intent == QueryIntent.HYBRID or requires_graph:
            return RetrievalStrategy.HYBRID_PARALLEL

        # Default: parallel hybrid for best coverage
        return RetrievalStrategy.HYBRID_PARALLEL

    def _retrieve_vector_only(
        self, query: ParsedQuery, top_k: int, timeout: float
    ) -> HybridRetrievalResult:
        """Retrieve using vector search only.

        Args:
            query: ParsedQuery object
            top_k: Number of results to return
            timeout: Timeout in seconds

        Returns:
            HybridRetrievalResult with vector results only
        """
        start_time = time.time()

        try:
            vector_result = self.vector_retriever.retrieve(
                query=query,
                top_k=top_k,
            )
            vector_time = (time.time() - start_time) * 1000

            # Convert to HybridChunks
            chunks = [
                self._vector_chunk_to_hybrid(chunk, rank)
                for rank, chunk in enumerate(vector_result.chunks, 1)
            ]

            return HybridRetrievalResult(
                query_id=query.query_id,
                query_text=query.original_text,
                strategy_used=RetrievalStrategy.VECTOR_ONLY,
                chunks=chunks,
                graph_paths=[],
                total_results=len(chunks),
                vector_results=len(chunks),
                graph_results=0,
                merged_results=len(chunks),
                retrieval_time_ms=vector_time,
                vector_time_ms=vector_time,
                graph_time_ms=None,
                merge_time_ms=0.0,
                vector_success=True,
                graph_success=True,  # Not attempted
                reranking_enabled=False,
            )

        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return self._empty_result(query, RetrievalStrategy.VECTOR_ONLY, vector_success=False)

    def _retrieve_graph_only(
        self, query: ParsedQuery, top_k: int, timeout: float
    ) -> HybridRetrievalResult:
        """Retrieve using graph search only.

        Args:
            query: ParsedQuery object
            top_k: Number of results to return
            timeout: Timeout in seconds

        Returns:
            HybridRetrievalResult with graph results only
        """
        start_time = time.time()

        try:
            graph_result = self.graph_retriever.retrieve(query=query)
            graph_time = (time.time() - start_time) * 1000

            # Extract chunks from graph paths
            merge_start = time.time()
            chunks = self._extract_chunks_from_graph(graph_result, query.entity_mentions)
            chunks = chunks[:top_k]  # Limit to top_k
            merge_time = (time.time() - merge_start) * 1000

            # Assign ranks
            for rank, chunk in enumerate(chunks, 1):
                chunk.rank = rank

            return HybridRetrievalResult(
                query_id=query.query_id,
                query_text=query.original_text,
                strategy_used=RetrievalStrategy.GRAPH_ONLY,
                chunks=chunks,
                graph_paths=graph_result.paths,
                total_results=len(chunks),
                vector_results=0,
                graph_results=len(chunks),
                merged_results=len(chunks),
                retrieval_time_ms=graph_time + merge_time,
                vector_time_ms=None,
                graph_time_ms=graph_time,
                merge_time_ms=merge_time,
                vector_success=True,  # Not attempted
                graph_success=True,
                reranking_enabled=False,
            )

        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return self._empty_result(query, RetrievalStrategy.GRAPH_ONLY, graph_success=False)

    def _retrieve_hybrid_parallel(
        self, query: ParsedQuery, top_k: int, timeout: float
    ) -> HybridRetrievalResult:
        """Retrieve using parallel vector and graph search.

        Args:
            query: ParsedQuery object
            top_k: Number of results to return
            timeout: Timeout in seconds

        Returns:
            HybridRetrievalResult with merged results
        """
        vector_result: Optional[RetrievalResult] = None
        graph_result: Optional[GraphRetrievalResult] = None
        vector_success = False
        graph_success = False
        vector_time = 0.0
        graph_time = 0.0

        # Execute both retrievers in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            vector_future = executor.submit(self._safe_vector_retrieve, query)
            graph_future = executor.submit(self._safe_graph_retrieve, query)

            # Collect results with timeout
            for future in as_completed([vector_future, graph_future], timeout=timeout):
                try:
                    if future == vector_future:
                        vector_result, vector_time = future.result()
                        vector_success = vector_result is not None
                    elif future == graph_future:
                        graph_result, graph_time = future.result()
                        graph_success = graph_result is not None
                except TimeoutError:
                    logger.warning("Retrieval operation timed out")
                except Exception as e:
                    logger.error(f"Retrieval operation failed: {e}")

        # Handle case where both failed
        if not vector_success and not graph_success:
            logger.error("Both vector and graph retrieval failed")
            return self._empty_result(
                query,
                RetrievalStrategy.HYBRID_PARALLEL,
                vector_success=False,
                graph_success=False,
            )

        # Merge results
        merge_start = time.time()
        chunks, graph_paths = self._merge_results(
            vector_result=vector_result,
            graph_result=graph_result,
            query=query,
            top_k=top_k,
        )
        merge_time = (time.time() - merge_start) * 1000

        # Calculate statistics
        vector_count = len(vector_result.chunks) if vector_result else 0
        graph_count = len(graph_result.chunk_ids) if graph_result else 0

        return HybridRetrievalResult(
            query_id=query.query_id,
            query_text=query.original_text,
            strategy_used=RetrievalStrategy.HYBRID_PARALLEL,
            chunks=chunks,
            graph_paths=graph_paths,
            total_results=vector_count + graph_count,
            vector_results=vector_count,
            graph_results=graph_count,
            merged_results=len(chunks),
            retrieval_time_ms=max(vector_time, graph_time) + merge_time,
            vector_time_ms=vector_time,
            graph_time_ms=graph_time,
            merge_time_ms=merge_time,
            vector_success=vector_success,
            graph_success=graph_success,
            reranking_enabled=self.reranking_config.enabled,
        )

    def _retrieve_vector_first(
        self, query: ParsedQuery, top_k: int, timeout: float
    ) -> HybridRetrievalResult:
        """Retrieve using vector first, then expand with graph.

        Args:
            query: ParsedQuery object
            top_k: Number of results to return
            timeout: Timeout in seconds

        Returns:
            HybridRetrievalResult with sequential results
        """
        # For now, delegate to hybrid parallel
        # TODO: Implement sequential vector-first strategy
        return self._retrieve_hybrid_parallel(query, top_k, timeout)

    def _retrieve_graph_first(
        self, query: ParsedQuery, top_k: int, timeout: float
    ) -> HybridRetrievalResult:
        """Retrieve using graph first, then supplement with vector.

        Args:
            query: ParsedQuery object
            top_k: Number of results to return
            timeout: Timeout in seconds

        Returns:
            HybridRetrievalResult with sequential results
        """
        # For now, delegate to hybrid parallel
        # TODO: Implement sequential graph-first strategy
        return self._retrieve_hybrid_parallel(query, top_k, timeout)

    def _safe_vector_retrieve(self, query: ParsedQuery) -> Tuple[Optional[RetrievalResult], float]:
        """Safely execute vector retrieval with error handling.

        Args:
            query: ParsedQuery object

        Returns:
            Tuple of (result, time_ms) or (None, 0.0) on failure
        """
        try:
            start_time = time.time()
            result = self.vector_retriever.retrieve(query=query)
            elapsed = (time.time() - start_time) * 1000
            return result, elapsed
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return None, 0.0

    def _safe_graph_retrieve(
        self, query: ParsedQuery
    ) -> Tuple[Optional[GraphRetrievalResult], float]:
        """Safely execute graph retrieval with error handling.

        Args:
            query: ParsedQuery object

        Returns:
            Tuple of (result, time_ms) or (None, 0.0) on failure
        """
        try:
            start_time = time.time()
            result = self.graph_retriever.retrieve(query=query)
            elapsed = (time.time() - start_time) * 1000
            return result, elapsed
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return None, 0.0

    def _merge_results(
        self,
        vector_result: Optional[RetrievalResult],
        graph_result: Optional[GraphRetrievalResult],
        query: ParsedQuery,
        top_k: int,
    ) -> Tuple[List[HybridChunk], List[GraphPath]]:
        """Merge and rerank results from vector and graph retrievers.

        Args:
            vector_result: Vector retrieval result (may be None)
            graph_result: Graph retrieval result (may be None)
            query: Original parsed query
            top_k: Number of final results to return

        Returns:
            Tuple of (merged_chunks, graph_paths)
        """
        # Collect chunks from both sources
        chunks_by_id: Dict[str, HybridChunk] = {}

        # Add vector chunks
        if vector_result:
            for chunk in vector_result.chunks:
                hybrid_chunk = self._vector_chunk_to_hybrid(chunk, rank=0)
                chunks_by_id[chunk.chunk_id] = hybrid_chunk

        # Add graph chunks
        if graph_result:
            graph_chunks = self._extract_chunks_from_graph(graph_result, query.entity_mentions)
            for chunk in graph_chunks:
                if chunk.chunk_id in chunks_by_id:
                    # Merge scores for duplicate chunks
                    existing = chunks_by_id[chunk.chunk_id]
                    existing.graph_score = chunk.graph_score
                    existing.entity_coverage_score = max(
                        existing.entity_coverage_score, chunk.entity_coverage_score
                    )
                    existing.graph_paths.extend(chunk.graph_paths)
                    existing.source = "hybrid"
                else:
                    chunks_by_id[chunk.chunk_id] = chunk

        # Convert to list
        all_chunks = list(chunks_by_id.values())

        # Apply score fusion
        fused_chunks = self._apply_score_fusion(all_chunks)

        # Apply diversity ranking if enabled
        if self.reranking_config.weights.get("diversity", 0.0) > 0:
            fused_chunks = self._apply_diversity_ranking(fused_chunks)

        # Sort by final score
        fused_chunks.sort(key=lambda c: c.final_score, reverse=True)

        # Take top-k and assign ranks
        top_chunks = fused_chunks[:top_k]
        for rank, chunk in enumerate(top_chunks, 1):
            chunk.rank = rank

        # Get graph paths (if any)
        graph_paths = graph_result.paths if graph_result else []

        return top_chunks, graph_paths

    def _apply_score_fusion(self, chunks: List[HybridChunk]) -> List[HybridChunk]:
        """Apply weighted score fusion to chunks.

        Args:
            chunks: List of HybridChunk objects

        Returns:
            Chunks with computed final_score
        """
        if not self.reranking_config.enabled:
            # Simple average of available scores
            for chunk in chunks:
                scores = []
                if chunk.vector_score is not None:
                    scores.append(chunk.vector_score)
                if chunk.graph_score is not None:
                    scores.append(chunk.graph_score)
                chunk.final_score = sum(scores) / len(scores) if scores else 0.0
            return chunks

        # Weighted fusion using config weights
        weights = self.reranking_config.weights

        for chunk in chunks:
            score = 0.0
            total_weight = 0.0

            # Vector similarity score
            if chunk.vector_score is not None:
                weight = weights.get("vector_similarity", 0.4)
                score += chunk.vector_score * weight
                total_weight += weight

            # Graph relevance score
            if chunk.graph_score is not None:
                weight = weights.get("graph_relevance", 0.3)
                score += chunk.graph_score * weight
                total_weight += weight

            # Entity coverage score
            weight = weights.get("entity_coverage", 0.15)
            score += chunk.entity_coverage_score * weight
            total_weight += weight

            # Confidence score
            weight = weights.get("confidence", 0.10)
            score += chunk.confidence_score * weight
            total_weight += weight

            # Normalize by total weight
            chunk.final_score = score / total_weight if total_weight > 0 else 0.0

        return chunks

    def _apply_diversity_ranking(self, chunks: List[HybridChunk]) -> List[HybridChunk]:
        """Apply diversity-aware reranking using MMR-like approach.

        Args:
            chunks: List of HybridChunk objects (sorted by score)

        Returns:
            Reranked chunks with diversity scores
        """
        if len(chunks) <= 1:
            return chunks

        diversity_weight = self.reranking_config.weights.get("diversity", 0.05)

        # Calculate diversity scores based on content similarity
        selected: List[HybridChunk] = []
        remaining = chunks.copy()

        # Select first chunk (highest score)
        selected.append(remaining.pop(0))
        selected[0].diversity_score = 1.0

        # Iteratively select remaining chunks
        while remaining:
            best_idx = 0
            best_score = float("-inf")

            for idx, candidate in enumerate(remaining):
                # Calculate diversity (dissimilarity to selected chunks)
                max_similarity = 0.0
                for selected_chunk in selected:
                    similarity = self._content_similarity(candidate.content, selected_chunk.content)
                    max_similarity = max(max_similarity, similarity)

                # Diversity score (1 - max similarity)
                diversity = 1.0 - max_similarity
                candidate.diversity_score = diversity

                # Combined score with diversity bonus
                combined = (
                    candidate.final_score * (1 - diversity_weight) + diversity * diversity_weight
                )

                if combined > best_score:
                    best_score = combined
                    best_idx = idx

            # Move best candidate to selected
            selected.append(remaining.pop(best_idx))

        return selected

    def _content_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple content similarity using Jaccard on words.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _extract_chunks_from_graph(
        self, graph_result: GraphRetrievalResult, entity_mentions: List[Any]
    ) -> List[HybridChunk]:
        """Extract chunk content from graph retrieval result.

        Args:
            graph_result: Graph retrieval result
            entity_mentions: Entity mentions from query for coverage scoring

        Returns:
            List of HybridChunk objects with graph scores
        """
        # Get unique chunk IDs from all paths
        chunk_ids = graph_result.chunk_ids

        if not chunk_ids:
            return []

        # Fetch chunk content from Neo4j
        chunks_data = self._fetch_chunks_by_ids(list(chunk_ids))

        # Create HybridChunk objects
        hybrid_chunks: List[HybridChunk] = []

        for chunk_data in chunks_data:
            # Find which paths contain this chunk
            containing_paths = [
                path.start_entity_id + "_" + path.end_entity_id
                for path in graph_result.paths
                if chunk_data["id"] in path.chunk_ids
            ]

            # Calculate graph score (average of path scores containing this chunk)
            path_scores = [
                path.score for path in graph_result.paths if chunk_data["id"] in path.chunk_ids
            ]
            graph_score = sum(path_scores) / len(path_scores) if path_scores else 0.5

            # Calculate entity coverage (how many query entities appear in this chunk)
            chunk_entities = set(chunk_data.get("entity_ids", []))
            query_entities = {m.normalized for m in entity_mentions}
            coverage = (
                len(chunk_entities & query_entities) / len(query_entities)
                if query_entities
                else 0.0
            )

            hybrid_chunk = HybridChunk(
                chunk_id=chunk_data["id"],
                document_id=chunk_data.get("document_id", ""),
                content=chunk_data.get("content", ""),
                level=chunk_data.get("level", 4),
                vector_score=None,
                graph_score=graph_score,
                entity_coverage_score=coverage,
                confidence_score=chunk_data.get("confidence", 0.5),
                diversity_score=0.0,
                final_score=graph_score,  # Will be recomputed in fusion
                rank=1,  # Placeholder, will be assigned after ranking
                metadata=chunk_data.get("metadata", {}),
                entity_ids=chunk_data.get("entity_ids", []),
                graph_paths=containing_paths,
                source="graph",
            )
            hybrid_chunks.append(hybrid_chunk)

        return hybrid_chunks

    def _fetch_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch chunk content from Neo4j by IDs.

        Args:
            chunk_ids: List of chunk IDs to fetch

        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not chunk_ids:
            return []

        query = """
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids
        RETURN c.id AS id,
               c.document_id AS document_id,
               c.content AS content,
               c.level AS level,
               c.metadata AS metadata,
               c.entity_ids AS entity_ids
        LIMIT 100
        """

        try:
            results = self.neo4j.execute_cypher(query, {"chunk_ids": chunk_ids})
            chunks = []
            for record in results:
                chunks.append(
                    {
                        "id": record["id"],
                        "document_id": record.get("document_id", ""),
                        "content": record.get("content", ""),
                        "level": record.get("level", 4),
                        "metadata": record.get("metadata", {}),
                        "entity_ids": record.get("entity_ids", []),
                    }
                )
            return chunks
        except Exception as e:
            logger.warning(f"Failed to fetch chunks from Neo4j: {e}")
            return []

    def _vector_chunk_to_hybrid(self, chunk: RetrievedChunk, rank: int) -> HybridChunk:
        """Convert VectorRetriever chunk to HybridChunk.

        Args:
            chunk: RetrievedChunk from vector retrieval
            rank: Rank in vector results (should be >= 1)

        Returns:
            HybridChunk object
        """
        return HybridChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=chunk.content,
            level=chunk.level,
            vector_score=chunk.normalized_score,
            graph_score=None,
            entity_coverage_score=0.0,  # Will be computed if needed
            confidence_score=0.5,
            diversity_score=0.0,
            final_score=chunk.normalized_score,
            rank=max(1, rank),  # Ensure rank is at least 1
            metadata=chunk.metadata,
            entity_ids=chunk.entity_ids,
            graph_paths=[],
            source="vector",
        )

    def _empty_result(
        self,
        query: ParsedQuery,
        strategy: RetrievalStrategy,
        vector_success: bool = True,
        graph_success: bool = True,
    ) -> HybridRetrievalResult:
        """Create empty result for failed retrievals.

        Args:
            query: Original query
            strategy: Strategy that was attempted
            vector_success: Whether vector retrieval succeeded
            graph_success: Whether graph retrieval succeeded

        Returns:
            Empty HybridRetrievalResult
        """
        return HybridRetrievalResult(
            query_id=query.query_id,
            query_text=query.original_text,
            strategy_used=strategy,
            chunks=[],
            graph_paths=[],
            total_results=0,
            vector_results=0,
            graph_results=0,
            merged_results=0,
            retrieval_time_ms=0.0,
            vector_time_ms=None,
            graph_time_ms=None,
            merge_time_ms=None,
            vector_success=vector_success,
            graph_success=graph_success,
            reranking_enabled=False,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get hybrid retrieval statistics.

        Returns:
            Dictionary with statistics
        """
        vector_stats = self.vector_retriever.get_statistics()
        graph_stats = self.graph_retriever.get_statistics()

        return {
            "hybrid_config": {
                "enabled": self.hybrid_config.enabled,
                "parallel_execution": self.hybrid_config.parallel_execution,
                "strategy_selection": self.hybrid_config.strategy_selection,
            },
            "reranking_config": {
                "enabled": self.reranking_config.enabled,
                "weights": self.reranking_config.weights,
                "max_results": self.reranking_config.max_results,
            },
            "vector_retriever": vector_stats,
            "graph_retriever": graph_stats,
        }

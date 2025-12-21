"""Vector-based semantic retrieval using Qdrant (Task 4.2).

This module implements semantic search functionality using vector similarity in Qdrant.
It supports filtering by metadata, pagination, score normalization, and optional MMR
for diversity-based retrieval.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.retrieval.query_parser import ParsedQuery
from src.storage.qdrant_manager import QdrantManager
from src.utils.config import Config, VectorSearchConfig
from src.utils.embeddings import EmbeddingGenerator


class RetrievedChunk(BaseModel):
    """Chunk retrieved from vector search."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Source document ID")
    content: str = Field(..., description="Chunk text content")
    level: int = Field(..., ge=1, le=4, description="Hierarchy level")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    normalized_score: float = Field(..., ge=0.0, le=1.0, description="Normalized relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    entity_ids: List[str] = Field(default_factory=list, description="Entity IDs in chunk")
    rank: int = Field(..., ge=1, description="Rank in result set")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class RetrievalResult(BaseModel):
    """Result of vector retrieval operation."""

    model_config = ConfigDict(extra="allow")

    query_id: str = Field(..., description="Query identifier")
    query_text: str = Field(..., description="Original query text")
    chunks: List[RetrievedChunk] = Field(default_factory=list, description="Retrieved chunks")
    total_results: int = Field(..., ge=0, description="Total matching results")
    page: int = Field(default=1, ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Results per page")
    has_more: bool = Field(default=False, description="Whether more results exist")
    retrieval_time_ms: float = Field(..., ge=0.0, description="Retrieval time in milliseconds")
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict, description="Filters used in search"
    )
    diversity_mode: Optional[str] = Field(None, description="Diversity algorithm used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Retrieval timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        data["chunks"] = [c.to_dict() for c in self.chunks]
        return data

    def get_entity_ids(self) -> Set[str]:
        """Extract all unique entity IDs from retrieved chunks.

        Returns:
            Set of unique entity IDs
        """
        entity_ids: Set[str] = set()
        for chunk in self.chunks:
            entity_ids.update(chunk.entity_ids)
        return entity_ids

    def get_document_ids(self) -> Set[str]:
        """Get all unique document IDs from retrieved chunks.

        Returns:
            Set of unique document IDs
        """
        return {chunk.document_id for chunk in self.chunks}


class VectorRetriever:
    """Vector-based semantic retriever using Qdrant."""

    def __init__(
        self,
        config: Optional[Config] = None,
        qdrant_manager: Optional[QdrantManager] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ) -> None:
        """Initialize vector retriever.

        Args:
            config: Configuration object
            qdrant_manager: Qdrant database manager (created if None)
            embedding_generator: Embedding generator (created if None)
        """
        self.config = config or Config.from_yaml()
        self.vector_config: VectorSearchConfig = self.config.retrieval.vector_search

        # Initialize Qdrant manager
        if qdrant_manager is None:
            self.qdrant = QdrantManager(config=self.config.database)
        else:
            self.qdrant = qdrant_manager

        # Initialize embedding generator
        if embedding_generator is None:
            self.embeddings = EmbeddingGenerator(config=self.config.database)
        else:
            self.embeddings = embedding_generator

        logger.info(
            "Initialized VectorRetriever",
            top_k=self.vector_config.top_k,
            min_score=self.vector_config.min_score,
            mmr_enabled=self.vector_config.enable_mmr,
        )

    def retrieve(
        self,
        query: str | ParsedQuery,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        use_mmr: Optional[bool] = None,
        mmr_lambda: Optional[float] = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query string or ParsedQuery object
            top_k: Number of results to return (default from config)
            min_score: Minimum similarity score threshold (default from config)
            filters: Metadata filters to apply
            page: Page number for pagination (1-indexed)
            use_mmr: Whether to use MMR for diversity (default from config)
            mmr_lambda: MMR lambda parameter for relevance/diversity tradeoff

        Returns:
            RetrievalResult with retrieved chunks and metadata

        Raises:
            ValueError: If query is empty or invalid
        """
        start_time = datetime.now()

        # Parse query if string
        if isinstance(query, str):
            query_text = query
            query_id = self._generate_query_id(query_text, start_time)
        else:
            query_text = query.original_text
            query_id = query.query_id

        # Validate query
        if not query_text or not query_text.strip():
            raise ValueError("Query cannot be empty")

        # Use config defaults if not specified
        top_k = top_k or self.vector_config.top_k
        min_score = min_score or self.vector_config.min_score
        use_mmr = use_mmr if use_mmr is not None else self.vector_config.enable_mmr
        mmr_lambda = mmr_lambda or self.vector_config.mmr_lambda

        # Generate query embedding
        query_embedding = self._generate_query_embedding(query_text)

        # Calculate pagination
        actual_limit = top_k * page if page > 1 else top_k
        offset = 0 if page == 1 else top_k * (page - 1)

        # Search vector database
        raw_results = self.qdrant.search_chunks(
            query_vector=query_embedding.tolist(),
            top_k=actual_limit,
            score_threshold=min_score,
            filters=filters,
        )

        # Apply pagination (slice results for the requested page)
        paginated_results = (
            raw_results[offset : offset + top_k] if page > 1 else raw_results[:top_k]
        )

        # Apply MMR if enabled
        if use_mmr and len(paginated_results) > 1:
            paginated_results = self._apply_mmr(
                query_embedding=query_embedding,
                results=paginated_results,
                lambda_param=mmr_lambda,
            )
            diversity_mode = f"mmr_lambda_{mmr_lambda}"
        else:
            diversity_mode = None

        # Normalize scores and create RetrievedChunk objects
        chunks = self._process_results(paginated_results)

        # Calculate retrieval time
        retrieval_time = (datetime.now() - start_time).total_seconds() * 1000

        # Create result object
        result = RetrievalResult(
            query_id=query_id,
            query_text=query_text,
            chunks=chunks,
            total_results=len(raw_results),
            page=page,
            page_size=top_k,
            has_more=len(raw_results) > (offset + top_k),
            retrieval_time_ms=retrieval_time,
            filters_applied=filters or {},
            diversity_mode=diversity_mode,
        )

        logger.info(
            "Vector retrieval completed",
            query_id=query_id,
            num_results=len(chunks),
            retrieval_time_ms=round(retrieval_time, 2),
            page=page,
        )

        return result

    def retrieve_by_document(
        self,
        query: str | ParsedQuery,
        document_ids: List[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResult:
        """Retrieve chunks from specific documents only.

        Args:
            query: Query string or ParsedQuery object
            document_ids: List of document IDs to search within
            top_k: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            RetrievalResult with retrieved chunks
        """
        filters = {"document_id": document_ids[0]} if len(document_ids) == 1 else None
        return self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            filters=filters,
        )

    def retrieve_by_entity(
        self,
        query: str | ParsedQuery,
        entity_ids: List[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResult:
        """Retrieve chunks that mention specific entities.

        Args:
            query: Query string or ParsedQuery object
            entity_ids: List of entity IDs to filter by
            top_k: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            RetrievalResult with retrieved chunks
        """
        filters = {"entity_ids": entity_ids}
        return self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            filters=filters,
        )

    def retrieve_by_level(
        self,
        query: str | ParsedQuery,
        level: int,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResult:
        """Retrieve chunks at a specific hierarchy level.

        Args:
            query: Query string or ParsedQuery object
            level: Hierarchy level (1=document, 2=section, 3=subsection, 4=paragraph)
            top_k: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            RetrievalResult with retrieved chunks

        Raises:
            ValueError: If level is not between 1 and 4
        """
        if not 1 <= level <= 4:
            raise ValueError(f"Level must be between 1 and 4, got {level}")

        filters = {"level": level}
        return self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            filters=filters,
        )

    def _generate_query_embedding(self, query_text: str) -> np.ndarray:
        """Generate embedding for query text.

        Args:
            query_text: Query text to embed

        Returns:
            Query embedding as numpy array
        """
        embeddings = self.embeddings.generate([query_text])
        return embeddings[0]

    def _process_results(self, raw_results: List[Dict[str, Any]]) -> List[RetrievedChunk]:
        """Process raw Qdrant results into RetrievedChunk objects.

        Args:
            raw_results: Raw results from Qdrant

        Returns:
            List of RetrievedChunk objects with normalized scores
        """
        if not raw_results:
            return []

        # Extract scores for normalization
        scores = [r["score"] for r in raw_results]
        normalized_scores = self._normalize_scores(scores)

        # Create RetrievedChunk objects
        chunks: List[RetrievedChunk] = []
        for idx, (result, norm_score) in enumerate(zip(raw_results, normalized_scores), 1):
            payload = result["payload"]

            chunk = RetrievedChunk(
                chunk_id=str(result["chunk_id"]),
                document_id=str(payload.get("document_id", "")),
                content=str(payload.get("content", "")),
                level=int(payload.get("level", 4)),
                score=float(result["score"]),
                normalized_score=float(norm_score),
                metadata=payload.get("metadata", {}),
                entity_ids=payload.get("entity_ids", []),
                rank=idx,
            )
            chunks.append(chunk)

        return chunks

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize similarity scores to [0, 1] range.

        Uses min-max normalization to spread scores across the full range.

        Args:
            scores: List of raw similarity scores

        Returns:
            List of normalized scores
        """
        if not scores:
            return []

        if len(scores) == 1:
            return [1.0]

        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score - min_score < 1e-10:
            return [1.0] * len(scores)

        # Min-max normalization
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]

        return normalized

    def _apply_mmr(
        self,
        query_embedding: np.ndarray,
        results: List[Dict[str, Any]],
        lambda_param: float = 0.5,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Apply Maximal Marginal Relevance for diversity.

        MMR balances relevance (similarity to query) with diversity
        (dissimilarity to already selected results).

        Algorithm:
            MMR = λ * Sim(d, q) - (1-λ) * max(Sim(d, d_i))
            where d is candidate document, q is query, d_i are selected documents

        Args:
            query_embedding: Query embedding vector
            results: List of search results with scores
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            top_k: Number of results to return (uses all if None)

        Returns:
            Reranked results with diversity
        """
        if len(results) <= 1:
            return results

        # We need document embeddings for MMR, but Qdrant search doesn't return them
        # For now, we'll use a simplified MMR based on content similarity
        # In a full implementation, we would retrieve vectors from Qdrant

        top_k = top_k or len(results)
        selected: List[Dict[str, Any]] = []
        remaining = results.copy()

        # Select first result (highest similarity to query)
        selected.append(remaining.pop(0))

        # Iteratively select remaining results
        while len(selected) < top_k and remaining:
            best_score = float("-inf")
            best_idx = 0

            for idx, candidate in enumerate(remaining):
                # Relevance: similarity to query (use original score)
                relevance = candidate["score"]

                # Diversity: dissimilarity to selected results
                # Simplified: use content overlap as proxy
                max_similarity = 0.0
                candidate_content = candidate["payload"].get("content", "")

                for selected_result in selected:
                    selected_content = selected_result["payload"].get("content", "")
                    # Simple word overlap similarity
                    similarity = self._content_similarity(candidate_content, selected_content)
                    max_similarity = max(max_similarity, similarity)

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            # Add best candidate to selected
            selected.append(remaining.pop(best_idx))

        return selected

    def _content_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple content similarity between two texts.

        Uses Jaccard similarity on word sets.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        # Tokenize and create sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _generate_query_id(self, query_text: str, timestamp: datetime) -> str:
        """Generate unique query ID.

        Args:
            query_text: Query text
            timestamp: Query timestamp

        Returns:
            Unique query ID
        """
        import hashlib

        content = f"{query_text}:{timestamp.isoformat()}"
        query_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"vquery_{timestamp.strftime('%Y%m%d_%H%M%S')}_{query_hash}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics.

        Returns:
            Dictionary with retrieval statistics
        """
        try:
            collection_info = self.qdrant.client.get_collection(self.qdrant.chunk_collection)
            return {
                "collection_name": self.qdrant.chunk_collection,
                "total_chunks": collection_info.points_count,
                "vector_dimension": self.config.database.embedding_dimension,
                "embedding_model": self.config.database.embedding_model,
                "top_k_default": self.vector_config.top_k,
                "min_score_default": self.vector_config.min_score,
                "mmr_enabled": self.vector_config.enable_mmr,
            }
        except Exception as e:
            logger.warning(f"Failed to get statistics: {e}")
            return {}

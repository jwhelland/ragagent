from __future__ import annotations

from typing import List, Sequence

from ..embeddings.client import EmbeddingsClient
from ..logging_setup import get_logger
from ..vectorstore.qdrant_store import QdrantStore
from .models import RetrievedChunk, VectorQueryResult


logger = get_logger(__name__)


class VectorRetriever:
    """
    Perform primary semantic retrieval over chunk embeddings stored in Qdrant.

    This retriever:
    - Embeds the user query with the configured ``EmbeddingsClient``.
    - Searches the Qdrant collection for the top-k most similar points.
    - Hydrates each hit into a ``RetrievedChunk`` using the stored payload
      (text, document metadata, entities, etc.).

    It is typically used as the first-stage retriever, whose results can then
    be expanded by the graph retriever and assembled into a final context
    window for the LLM.
    """
    def __init__(
        self,
        store: QdrantStore,
        embedder: EmbeddingsClient,
        *,
        top_k: int = 8,
        score_threshold: float | None = None,
    ) -> None:
        """
        Initialize a VectorRetriever.

        Args:
            store: Backing ``QdrantStore`` used for vector search and payloads.
            embedder: ``EmbeddingsClient`` used to embed user queries.
            top_k: Default number of nearest-neighbor results to request
                when ``retrieve`` is called without an explicit ``top_k``.
            score_threshold: Optional minimum similarity score; results
                below this value are filtered out by the vector store.
        """
        self._store = store
        self._embedder = embedder
        self._top_k = top_k
        self._score_threshold = score_threshold

    def retrieve(self, query: str, *, top_k: int | None = None) -> VectorQueryResult:
        """
        Run a semantic search query against the vector store.

        Workflow:
        1. Embed the natural-language ``query`` using the embeddings service.
        2. Execute a nearest-neighbor search in Qdrant.
        3. Convert each returned point into a ``RetrievedChunk`` using the
           stored payload and similarity score.

        Args:
            query: User query text to embed and search with.
            top_k: Optional override for the number of neighbors to fetch;
                defaults to the instance's configured ``top_k``.

        Returns:
            A ``VectorQueryResult`` containing the query embedding and an
            ordered list of vector-origin ``RetrievedChunk`` objects.
        """
        vector = self._embed_query(query)
        limit = top_k or self._top_k
        points = self._store.search(vector, top_k=limit, score_threshold=self._score_threshold)
        chunks: List[RetrievedChunk] = []
        for rank, point in enumerate(points, start=1):
            payload = point.payload or {}
            text = payload.get("text")
            if not text:
                logger.warning("vector_hit_missing_text", chunk_id=point.id)
                continue
            chunk_id = payload.get("chunk_id") or str(point.id)
            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=payload.get("doc_id", "unknown"),
                text=text,
                page=payload.get("page"),
                table_id=payload.get("table_id"),
                source_path=payload.get("source_path"),
                score=float(point.score or 0.0),
                origin="vector",
                entities=list(payload.get("entities") or []),
                metadata={
                    "rank": rank,
                    "sha256": payload.get("sha256"),
                },
            )
            chunks.append(chunk)
        return VectorQueryResult(query=query, embedding=vector, chunks=chunks)

    def _embed_query(self, query: str) -> Sequence[float]:
        """
        Embed a single query string using the embeddings client.

        Args:
            query: Text to embed.

        Returns:
            A single embedding vector suitable for passing to Qdrant.

        Raises:
            RuntimeError: If the embeddings service returns no vectors.
        """
        vectors = self._embedder.embed([query])
        if not vectors:
            raise RuntimeError("embeddings_service_returned_empty_vector")
        return vectors[0]

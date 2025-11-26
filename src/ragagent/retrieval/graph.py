from __future__ import annotations

from typing import List, Sequence, Optional

from ..graph.store import GraphStore
from ..logging_setup import get_logger
from ..nlp.keywords import extract_entities_and_phrases
from ..vectorstore.qdrant_store import QdrantStore
from .models import RetrievedChunk, VectorQueryResult


logger = get_logger(__name__)


class GraphRetriever:
    """
    Retrieve additional context by expanding seed vector hits with
    graph-based signals (shared entities) from Neo4j.

    This component takes the top-k results from the vector store as
    "seed" sections and then:
    - Finds other sections that share entities with those seeds.
    - Optionally falls back to an entity search based on the user
      question if the graph expansion is too sparse.
    - Hydrates the resulting section IDs back into full chunks using
      the vector store payloads and assigns heuristic scores.

    It is intended to be composed with the primary vector retriever
    and the context assembler, not used directly by API callers.
    """
    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: QdrantStore,
        *,
        seed_limit: int = 4,
        related_per_seed: int = 2,
        fallback_entity_terms: int = 5,
        use_entity_relations: bool = False,
        relation_types: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize a GraphRetriever.

        Args:
            graph_store: GraphStore instance used for Neo4j/Graphiti queries.
            vector_store: QdrantStore used to hydrate section IDs into chunks.
            seed_limit: Maximum number of top vector hits to treat as seeds.
            related_per_seed: Maximum number of related sections to retrieve
                per seed section from the graph.
            fallback_entity_terms: Maximum number of entity terms from the
                question to use when performing fallback entity search.
        """
        self._graph = graph_store
        self._vector_store = vector_store
        self._seed_limit = seed_limit
        self._related_per_seed = related_per_seed
        self._fallback_entity_terms = fallback_entity_terms
        self._use_entity_relations = use_entity_relations
        self._relation_types = relation_types

    def expand(
        self,
        question: str,
        vector_result: VectorQueryResult,
        *,
        max_results: int = 4,
    ) -> List[RetrievedChunk]:
        """
        Expand vector search results with graph-related sections.

        Workflow:
        1. Take up to ``seed_limit`` chunks from ``vector_result`` as seeds.
        2. Ask the graph store for sections that share entities with those
           seeds (co-mention graph expansion).
        3. If there are fewer hits than ``max_results``, fall back to
           searching sections by entities extracted from the question text.
        4. Hydrate resulting section IDs into full ``RetrievedChunk`` objects
           by fetching payloads from the vector store.

        Args:
            question: The user query text, used only for fallback entity search.
            vector_result: Result of the initial vector retrieval step.
            max_results: Maximum number of graph-derived chunks to return.

        Returns:
            A list of graph-origin ``RetrievedChunk`` instances, ordered by
            a heuristic graph rank and truncated to ``max_results`` items.
        """
        seeds = vector_result.chunks[: self._seed_limit]
        seed_ids = {chunk.chunk_id for chunk in seeds}
        related_rows = self._graph.get_related_sections(list(seed_ids), limit_per_seed=self._related_per_seed)
        if self._use_entity_relations:
            rel_rows = self._graph.get_sections_via_entity_relations(
                list(seed_ids),
                relation_types=self._relation_types,
                limit_per_seed=self._related_per_seed,
            )
            # Prioritize relation-driven rows ahead of plain co-mentions
            related_rows = (rel_rows or []) + related_rows

        if len(related_rows) < max_results:
            entities, phrases = extract_entities_and_phrases(question)
            search_terms = entities or phrases
            if search_terms:
                fallback_rows = self._graph.search_sections_by_entities(
                    search_terms[: self._fallback_entity_terms],
                    limit=max_results * 2,
                )
                related_rows.extend(fallback_rows)

        hydrated = self._hydrate_rows(related_rows, seed_ids, max_results)
        return hydrated

    def _hydrate_rows(
        self,
        rows: Sequence[dict],
        seed_ids: set[str],
        max_results: int,
    ) -> List[RetrievedChunk]:
        """
        Convert raw graph query rows into hydrated ``RetrievedChunk`` objects.

        This method:
        - De-duplicates rows against seed IDs and previously seen chunk IDs.
        - Trims the candidate list to at most ``max_results * 2`` entries
          before hydration to limit Qdrant lookups.
        - Fetches payloads from the vector store and builds ``RetrievedChunk``
          objects with a simple heuristic score that rewards more shared
          entities.

        Args:
            rows: Raw dictionaries returned from ``GraphStore`` queries.
            seed_ids: Chunk IDs that came from the original vector search
                and should not be re-emitted as graph results.
            max_results: Maximum number of hydrated chunks to return.

        Returns:
            A list of hydrated ``RetrievedChunk`` instances enriched with
            graph metadata such as ``graph_rank`` and ``shared_entities``.
        """
        ordered_rows: List[dict] = []
        seen_ids: set[str] = set(seed_ids)
        for row in rows:
            chunk_id = row.get("chunk_id")
            if not chunk_id or chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            ordered_rows.append(row)
            if len(ordered_rows) >= max_results * 2:
                break

        payloads = self._vector_store.fetch_by_ids([row["chunk_id"] for row in ordered_rows if "chunk_id" in row])
        results: List[RetrievedChunk] = []
        for rank, row in enumerate(ordered_rows, start=1):
            chunk_id = row.get("chunk_id")
            if not chunk_id:
                continue
            payload = payloads.get(chunk_id, {})
            text = payload.get("text")
            if not text:
                logger.warning("graph_hit_missing_text", chunk_id=chunk_id)
                continue
            entities = row.get("entities") or payload.get("entities") or []
            score = 0.35 + 0.05 * min(len(entities), 5)
            meta = {
                "graph_rank": rank,
                "shared_entities": entities,
            }
            if "relation_types" in row:
                meta["relation_types"] = row.get("relation_types") or []
            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=row.get("doc_id") or payload.get("doc_id", "unknown"),
                text=text,
                page=row.get("page") or payload.get("page"),
                table_id=payload.get("table_id"),
                source_path=payload.get("source_path"),
                score=score,
                origin="graph",
                seed_chunk_id=row.get("seed_chunk_id"),
                entities=list(entities),
                metadata=meta,
            )
            results.append(chunk)
            if len(results) >= max_results:
                break
        return results

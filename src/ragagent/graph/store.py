from __future__ import annotations

from typing import Any, Dict, List, Optional

from datetime import datetime, timezone

import asyncio
import re
from neo4j import GraphDatabase

try:
    # Prefer official Graphiti Core client if available
    from graphiti_core import Graphiti  # type: ignore
    from graphiti_core.nodes import EpisodeType  # type: ignore
except Exception:  # noqa: BLE001
    Graphiti = None  # type: ignore
    EpisodeType = None  # type: ignore


class GraphStore:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._graphiti: Optional[Graphiti] = None
        if Graphiti is not None:
            try:
                self._graphiti = Graphiti(uri, user, password)  # type: ignore[call-arg]
            except Exception:
                self._graphiti = None
        self._graphiti_ready = False

    def close(self):
        try:
            self._driver.close()
        finally:
            if self._graphiti:
                try:
                    asyncio.run(self._graphiti.close())  # type: ignore[func-returns-value]
                except Exception:
                    pass

    def _run(self, cypher: str, params: Dict[str, Any]) -> None:
        # If using Graphiti, prefer its high-level methods; for raw cypher fallback to driver
        with self._driver.session() as s:
            s.run(cypher, **params)

    def _query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        with self._driver.session() as s:
            result = s.run(cypher, **params)
            return [record.data() for record in result]

    def upsert_document(self, doc_id: str, sha256: str, path: str) -> None:
        cypher = "MERGE (d:Document {id: $id}) SET d.sha256=$sha, d.path=$path RETURN d"
        self._run(cypher, {"id": doc_id, "sha": sha256, "path": path})

    def upsert_section(self, doc_id: str, page: int, chunk_id: str) -> None:
        cypher = "MERGE (s:Section {chunk_id: $chunk_id}) SET s.document_id=$doc_id, s.page=$page RETURN s"
        self._run(cypher, {"chunk_id": chunk_id, "doc_id": doc_id, "page": page})

    def link_document_section(self, doc_id: str, chunk_id: str) -> None:
        cypher = (
            "MATCH (d:Document {id:$doc_id}) "
            "MATCH (s:Section {chunk_id:$chunk_id}) "
            "MERGE (d)-[:HAS_SECTION]->(s)"
        )
        self._run(cypher, {"doc_id": doc_id, "chunk_id": chunk_id})

    def upsert_entity(self, key: str) -> None:
        cypher = "MERGE (e:Entity {key:$key}) RETURN e"
        self._run(cypher, {"key": key})

    def link_section_entity(self, chunk_id: str, key: str) -> None:
        cypher = (
            "MATCH (s:Section {chunk_id:$chunk_id}) "
            "MATCH (e:Entity {key:$key}) "
            "MERGE (s)-[:MENTIONS]->(e)"
        )
        self._run(cypher, {"chunk_id": chunk_id, "key": key})

    def upsert_entity_relation(
        self,
        source_key: str,
        relation_type: str,
        target_key: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create or update a typed relation between two entities:
        (Entity {key:source_key})-[:RELATION_TYPE {..props..}]->(Entity {key:target_key})

        - relation_type is sanitized to uppercase with underscores and validated by regex.
        - properties (confidence, justification, chunk_id, doc_id, page, etc.) are merged onto the relationship.
        """
        rel_type = (relation_type or "").upper().strip().replace(" ", "_")
        if not rel_type or not re.match(r"^[A-Z_]{1,32}$", rel_type):
            raise ValueError(f"invalid_relation_type: {relation_type!r}")

        cypher = (
            "MATCH (s:Entity {key:$source}) "
            "MATCH (t:Entity {key:$target}) "
            f"MERGE (s)-[r:{rel_type}]->(t) "
            "SET r += $props"
        )
        params: Dict[str, Any] = {
            "source": source_key,
            "target": target_key,
            "props": properties or {},
        }
        self._run(cypher, params)

    def add_episode_for_chunk(
        self,
        *,
        name: str,
        body: str,
        doc_id: str,
        page: int | None,
        source_desc: str = "PDF chunk",
    ) -> None:
        return

    def get_related_sections(
        self, chunk_ids: List[str], limit_per_seed: int = 2
    ) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []
        limit = max(1, limit_per_seed) * len(chunk_ids)
        cypher = """
        MATCH (seed:Section)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(other:Section)
        WHERE seed.chunk_id IN $chunk_ids AND other.chunk_id <> seed.chunk_id
        WITH seed, other, apoc.coll.toSet(collect(e.key)) AS shared_entities
        RETURN seed.chunk_id AS seed_chunk_id,
               other.chunk_id AS chunk_id,
               other.document_id AS doc_id,
               other.page AS page,
               shared_entities AS entities
        ORDER BY size(shared_entities) DESC
        LIMIT $limit
        """
        try:
            return self._query(cypher, {"chunk_ids": chunk_ids, "limit": limit})
        except Exception:
            # APOC might be unavailable; fallback without apoc
            cypher_no_apoc = """
            MATCH (seed:Section)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(other:Section)
            WHERE seed.chunk_id IN $chunk_ids AND other.chunk_id <> seed.chunk_id
            WITH seed, other, collect(DISTINCT e.key) AS shared_entities
            RETURN seed.chunk_id AS seed_chunk_id,
                   other.chunk_id AS chunk_id,
                   other.document_id AS doc_id,
                   other.page AS page,
                   shared_entities AS entities
            ORDER BY size(shared_entities) DESC
            LIMIT $limit
            """
            return self._query(cypher_no_apoc, {"chunk_ids": chunk_ids, "limit": limit})

    def search_sections_by_entities(
        self, entities: List[str], limit: int = 5
    ) -> List[Dict[str, Any]]:
        if not entities:
            return []
        cypher = """
        MATCH (s:Section)-[:MENTIONS]->(e:Entity)
        WHERE e.key IN $entities
        WITH s, collect(DISTINCT e.key) AS matched_entities
        RETURN s.chunk_id AS chunk_id,
               s.document_id AS doc_id,
               s.page AS page,
               matched_entities AS entities
        ORDER BY size(matched_entities) DESC
        LIMIT $limit
        """
        return self._query(cypher, {"entities": entities, "limit": limit})

    def get_sections_via_entity_relations(
        self,
        seed_chunk_ids: List[str],
        relation_types: Optional[List[str]] = None,
        limit_per_seed: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Expand from seed sections via typed entity-to-entity relations:
        (seed:Section)-[:MENTIONS]->(e1:Entity)-[r:TYPE]->(e2:Entity)&lt;-[:MENTIONS]-(other:Section)

        Returns rows shaped like get_related_sections(), plus relation_types for transparency:
        {
          seed_chunk_id, chunk_id, doc_id, page, entities, relation_types
        }
        """
        if not seed_chunk_ids:
            return []
        limit = max(1, limit_per_seed) * len(seed_chunk_ids)

        cypher = """
        MATCH (seed:Section)-[:MENTIONS]->(e1:Entity)
        MATCH (e1)-[r]->(e2:Entity)
        MATCH (other:Section)-[:MENTIONS]->(e2)
        WHERE seed.chunk_id IN $chunk_ids AND other.chunk_id <> seed.chunk_id
        """
        params: Dict[str, Any] = {"chunk_ids": seed_chunk_ids, "limit": limit}

        if relation_types:
            # Filter on given relation types using WHERE type(r) IN $rel_types
            cypher += "AND type(r) IN $rel_types\n"
            params["rel_types"] = [t.upper() for t in relation_types]

        cypher += """
        WITH seed, other, collect(DISTINCT e2.key) AS related_entities, collect(DISTINCT type(r)) AS relation_types
        RETURN seed.chunk_id AS seed_chunk_id,
               other.chunk_id AS chunk_id,
               other.document_id AS doc_id,
               other.page AS page,
               related_entities AS entities,
               relation_types AS relation_types
        ORDER BY size(related_entities) DESC
        LIMIT $limit
        """
        return self._query(cypher, params)

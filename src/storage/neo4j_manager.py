"""Neo4j graph database manager for entity, relationship, and candidate storage."""

import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from neo4j import GraphDatabase, Session
from neo4j.exceptions import Neo4jError
from neo4j.time import Date, DateTime, Time

from src.storage.schemas import (
    CandidateStatus,
    Chunk,
    Entity,
    EntityCandidate,
    EntityStatus,
    EntityType,
    Relationship,
    RelationshipCandidate,
    RelationshipType,
)
from src.utils.config import DatabaseConfig


class Neo4jManager:
    """Manager for Neo4j graph database operations.

    Handles connection pooling, schema creation, CRUD operations for entities
    and relationships, and query execution.

    Attributes:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        database: Neo4j database name
        driver: Neo4j driver instance
    """

    def __init__(self, config: DatabaseConfig):
        """Initialize Neo4j manager with configuration.

        Args:
            config: Database configuration
        """
        self.uri = config.neo4j_uri
        self.user = config.neo4j_user
        self.password = config.neo4j_password
        self.database = config.neo4j_database
        self.driver = None
        self._connected = False

    def connect(self, debug: bool = False) -> None:
        """Establish connection to Neo4j database.

        Args:
            debug: Whether to enable debug logging and notifications

        Raises:
            Neo4jError: If connection fails
        """
        try:
            # Configure notification severity based on debug flag
            # Suppress warnings (like missing relationship types) unless debugging
            notifications_min_severity = "WARNING" if debug else "OFF"

            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=50,
                notifications_min_severity=notifications_min_severity,
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Neo4jError as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    @staticmethod
    def _convert_neo4j_temporals(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Neo4j temporal types to ISO format strings.

        Args:
            data: Dictionary that may contain Neo4j DateTime, Date, or Time objects

        Returns:
            Dictionary with temporal objects converted to ISO strings
        """
        converted = {}
        for key, value in data.items():
            if isinstance(value, (DateTime, Date, Time)):
                # Convert to ISO format string
                converted[key] = value.isoformat()
            elif isinstance(value, list):
                # Handle lists that might contain temporal objects
                converted[key] = [
                    item.isoformat() if isinstance(item, (DateTime, Date, Time)) else item
                    for item in value
                ]
            else:
                converted[key] = value
        return converted

    def close(self) -> None:
        """Close connection to Neo4j database."""
        if self.driver:
            self.driver.close()
            self._connected = False
            logger.info("Closed Neo4j connection")

    @contextmanager
    def session(self) -> Session:
        """Context manager for Neo4j session.

        Yields:
            Neo4j session instance

        Raises:
            RuntimeError: If not connected to database
        """
        if not self._connected or not self.driver:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")

        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    def create_schema(self) -> None:
        """Create Neo4j schema with constraints and indexes.

        Creates:
        - Uniqueness constraints on entity IDs
        - Indexes on canonical_name, entity_type
        - Full-text search indexes on entity properties
        - Relationship indexes
        """
        with self.session() as session:
            # Create uniqueness constraints for all entity types
            entity_types = [et.value for et in EntityType]
            for entity_type in entity_types:
                try:
                    session.run(
                        f"CREATE CONSTRAINT {entity_type.lower()}_id_unique IF NOT EXISTS "
                        f"FOR (n:{entity_type}) REQUIRE n.id IS UNIQUE"
                    )
                    logger.info(f"Created uniqueness constraint for {entity_type}")
                except Neo4jError as e:
                    logger.warning(f"Could not create constraint for {entity_type}: {e}")

            # Create constraints for candidate nodes
            try:
                session.run(
                    "CREATE CONSTRAINT entity_candidate_id_unique IF NOT EXISTS "
                    "FOR (c:EntityCandidate) REQUIRE c.id IS UNIQUE"
                )
                session.run(
                    "CREATE CONSTRAINT entity_candidate_key_unique IF NOT EXISTS "
                    "FOR (c:EntityCandidate) REQUIRE c.candidate_key IS UNIQUE"
                )
                logger.info("Created uniqueness constraints for EntityCandidate")
            except Neo4jError as e:
                logger.warning(f"Could not create constraints for EntityCandidate: {e}")

            try:
                session.run(
                    "CREATE CONSTRAINT relationship_candidate_id_unique IF NOT EXISTS "
                    "FOR (c:RelationshipCandidate) REQUIRE c.id IS UNIQUE"
                )
                session.run(
                    "CREATE CONSTRAINT relationship_candidate_key_unique IF NOT EXISTS "
                    "FOR (c:RelationshipCandidate) REQUIRE c.candidate_key IS UNIQUE"
                )
                logger.info("Created uniqueness constraints for RelationshipCandidate")
            except Neo4jError as e:
                logger.warning(f"Could not create constraints for RelationshipCandidate: {e}")

            # Create constraint for Chunk nodes
            try:
                session.run(
                    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                    "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
                )
                logger.info("Created uniqueness constraint for Chunk")
            except Neo4jError as e:
                logger.warning(f"Could not create constraint for Chunk: {e}")

            # Create indexes on canonical_name for all entity types
            for entity_type in entity_types:
                try:
                    session.run(
                        f"CREATE INDEX {entity_type.lower()}_canonical_name IF NOT EXISTS "
                        f"FOR (n:{entity_type}) ON (n.canonical_name)"
                    )
                    logger.info(f"Created canonical_name index for {entity_type}")
                except Neo4jError as e:
                    logger.warning(f"Could not create canonical_name index for {entity_type}: {e}")

            # Candidate indexes for common filters
            try:
                session.run(
                    "CREATE INDEX entity_candidate_type_idx IF NOT EXISTS "
                    "FOR (c:EntityCandidate) ON (c.candidate_type)"
                )
                session.run(
                    "CREATE INDEX entity_candidate_status_idx IF NOT EXISTS "
                    "FOR (c:EntityCandidate) ON (c.status)"
                )
                session.run(
                    "CREATE INDEX entity_candidate_conf_idx IF NOT EXISTS "
                    "FOR (c:EntityCandidate) ON (c.confidence_score)"
                )
                logger.info("Created indexes for EntityCandidate")
            except Neo4jError as e:
                logger.warning(f"Could not create EntityCandidate indexes: {e}")

            try:
                session.run(
                    "CREATE INDEX relationship_candidate_type_idx IF NOT EXISTS "
                    "FOR (c:RelationshipCandidate) ON (c.type)"
                )
                session.run(
                    "CREATE INDEX relationship_candidate_status_idx IF NOT EXISTS "
                    "FOR (c:RelationshipCandidate) ON (c.status)"
                )
                session.run(
                    "CREATE INDEX relationship_candidate_conf_idx IF NOT EXISTS "
                    "FOR (c:RelationshipCandidate) ON (c.confidence_score)"
                )
                logger.info("Created indexes for RelationshipCandidate")
            except Neo4jError as e:
                logger.warning(f"Could not create RelationshipCandidate indexes: {e}")

            # Create indexes on status for filtering (per-label; Neo4j indexes require a label)
            for entity_type in entity_types:
                try:
                    session.run(
                        f"CREATE INDEX {entity_type.lower()}_status_idx IF NOT EXISTS "
                        f"FOR (n:{entity_type}) ON (n.status)"
                    )
                    logger.info(f"Created status index for {entity_type}")
                except Neo4jError as e:
                    logger.warning(f"Could not create status index for {entity_type}: {e}")

            # Create full-text search index for entities
            try:
                # Drop existing index if it exists
                session.run("DROP INDEX entity_fulltext IF EXISTS")

                # Create full-text index on all entity types
                entity_labels = "|".join(entity_types)
                session.run(
                    f"CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
                    f"FOR (n:{entity_labels}) "
                    f"ON EACH [n.canonical_name, n.aliases, n.description]"
                )
                logger.info("Created full-text search index for entities")
            except Neo4jError as e:
                logger.warning(f"Could not create full-text search index: {e}")

            # Create full-text search index for candidates
            try:
                session.run("DROP INDEX entity_candidate_fulltext IF EXISTS")
                session.run(
                    "CREATE FULLTEXT INDEX entity_candidate_fulltext IF NOT EXISTS "
                    "FOR (c:EntityCandidate) "
                    "ON EACH [c.canonical_name, c.aliases, c.description]"
                )
                logger.info("Created full-text search index for EntityCandidate")
            except Neo4jError as e:
                logger.warning(f"Could not create EntityCandidate full-text index: {e}")

            try:
                session.run("DROP INDEX relationship_candidate_fulltext IF EXISTS")
                session.run(
                    "CREATE FULLTEXT INDEX relationship_candidate_fulltext IF NOT EXISTS "
                    "FOR (c:RelationshipCandidate) "
                    "ON EACH [c.source, c.target, c.description]"
                )
                logger.info("Created full-text search index for RelationshipCandidate")
            except Neo4jError as e:
                logger.warning(f"Could not create RelationshipCandidate full-text index: {e}")

            # Create index on document_id for chunks
            try:
                session.run(
                    "CREATE INDEX chunk_document_id IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)"
                )
                logger.info("Created document_id index for chunks")
            except Neo4jError as e:
                logger.warning(f"Could not create document_id index: {e}")

            # Create constraint for Topic nodes
            try:
                session.run(
                    "CREATE CONSTRAINT topic_name_unique IF NOT EXISTS "
                    "FOR (t:Topic) REQUIRE t.name IS UNIQUE"
                )
                logger.info("Created uniqueness constraint for Topic")
            except Neo4jError as e:
                logger.warning(f"Could not create constraint for Topic: {e}")

            # Relationship property indexes require a concrete relationship type in Neo4j.
            # Since our relationships use the relationship *type* itself (e.g. :DEPENDS_ON),
            # indexing a `r.type` property isn't needed here.

            logger.info("Neo4j schema creation completed")

    @staticmethod
    def _deterministic_uuid(key: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    # Topic Operations

    def save_document_topics(self, document_id: str, topics: List[str]) -> None:
        """Link a document to one or more Topic nodes.

        Args:
            document_id: The ID of the document.
            topics: A list of topic names.
        """
        if not topics:
            return

        with self.session() as session:
            # We use MERGE for topics to ensure we don't duplicate them.
            # We use MERGE for the relationship to avoid duplicate links.
            query = """
            MATCH (d:DOCUMENT {id: $document_id})
            UNWIND $topics AS topic_name
            MERGE (t:Topic {name: topic_name})
            MERGE (d)-[:HAS_TOPIC]->(t)
            """
            session.run(query, document_id=document_id, topics=topics)
            logger.debug(f"Linked document {document_id} to topics: {topics}")

    def list_topics(self) -> List[str]:
        """List all available topic names.

        Returns:
            List of topic names.
        """
        with self.session() as session:
            query = """
            MATCH (t:Topic)
            RETURN t.name as name
            ORDER BY t.name
            """
            result = session.run(query)
            return [record["name"] for record in result]

    def get_document_ids_by_topic(self, topic: str) -> List[str]:
        """Get IDs of documents linked to a specific topic.

        Args:
            topic: Topic name.

        Returns:
            List of document IDs.
        """
        with self.session() as session:
            query = """
            MATCH (d:DOCUMENT)-[:HAS_TOPIC]->(t:Topic {name: $topic})
            RETURN d.id as id
            """
            result = session.run(query, topic=topic)
            return [record["id"] for record in result]

    # Candidate Operations

    def upsert_entity_candidate_aggregate(self, candidate: EntityCandidate) -> str:
        """Upsert an EntityCandidate, aggregating mention counts and seen locations.

        Note:
            Neo4j node properties cannot store nested structures; provenance is stored as a
            list of JSON strings in `provenance_events`.
        """
        with self.session() as session:
            now = datetime.now().isoformat()
            candidate_id = candidate.id or self._deterministic_uuid(
                f"entity_candidate:{candidate.candidate_key}"
            )

            props = candidate.to_neo4j_dict()
            props["id"] = candidate_id
            props["last_seen"] = now

            query = """
            MERGE (c:EntityCandidate {candidate_key: $candidate_key})
            ON CREATE SET c += $props
            ON MATCH SET
                c.last_seen = $last_seen,
                c.mention_count = coalesce(c.mention_count, 0) + $mention_count,
                c.confidence_score = CASE
                    WHEN $confidence_score > coalesce(c.confidence_score, 0.0)
                    THEN $confidence_score
                    ELSE coalesce(c.confidence_score, 0.0)
                END
            WITH c
            SET c.source_documents = coalesce(c.source_documents, []) +
                [d IN $source_documents WHERE NOT d IN coalesce(c.source_documents, [])]
            SET c.chunk_ids = coalesce(c.chunk_ids, []) +
                [ch IN $chunk_ids WHERE NOT ch IN coalesce(c.chunk_ids, [])]
            SET c.provenance_events = coalesce(c.provenance_events, []) +
                [p IN $provenance_events WHERE NOT p IN coalesce(c.provenance_events, [])]
            WITH c
            FOREACH (doc_id IN $source_documents |
                MERGE (d:DOCUMENT {id: doc_id})
                MERGE (c)-[:MENTIONED_IN]->(d)
            )
            RETURN c.id as id
            """
            result = session.run(
                query,
                candidate_key=candidate.candidate_key,
                props=props,
                last_seen=now,
                mention_count=int(candidate.mention_count),
                confidence_score=float(candidate.confidence_score),
                source_documents=list(candidate.source_documents),
                chunk_ids=list(candidate.chunk_ids),
                aliases=list(candidate.aliases),
                conflicting_types=list(candidate.conflicting_types),
                provenance_events=list(candidate.provenance_events),
            )
            return result.single()["id"]

    def get_total_entity_candidate_count(
        self,
        *,
        status: Optional[str] = None,
        candidate_types: Optional[List[EntityType]] = None,
        min_confidence: Optional[float] = None,
    ) -> int:
        """Get the total count of entity candidates matching the criteria."""
        with self.session() as session:
            query = """
            MATCH (c:EntityCandidate)
            WHERE ($status IS NULL OR c.status = $status)
            AND ($min_confidence IS NULL OR c.confidence_score >= $min_confidence)
            AND (
                $candidate_types IS NULL
                OR c.candidate_type IN $candidate_types
            )
            RETURN count(c) as count
            """
            result = session.run(
                query,
                status=status,
                min_confidence=min_confidence,
                candidate_types=[t.value for t in candidate_types] if candidate_types else None,
            )
            return result.single()["count"]

    def get_total_relationship_candidate_count(
        self,
        *,
        status: Optional[str] = None,
        rel_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> int:
        """Get the total count of relationship candidates matching the criteria."""
        with self.session() as session:
            query = """
            MATCH (c:RelationshipCandidate)
            WHERE ($status IS NULL OR c.status = $status)
            AND ($rel_type IS NULL OR c.type = $rel_type)
            AND ($min_confidence IS NULL OR c.confidence_score >= $min_confidence)
            RETURN count(c) as count
            """
            result = session.run(
                query,
                status=status,
                rel_type=rel_type,
                min_confidence=min_confidence,
            )
            return result.single()["count"]

    def get_entity_candidates(
        self,
        *,
        status: Optional[str] = None,
        candidate_types: Optional[List[EntityType]] = None,
        min_confidence: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query EntityCandidate nodes by status/type/confidence."""
        with self.session() as session:
            query = """
            MATCH (c:EntityCandidate)
            WHERE ($status IS NULL OR c.status = $status)
            AND ($min_confidence IS NULL OR c.confidence_score >= $min_confidence)
            AND (
                $candidate_types IS NULL
                OR c.candidate_type IN $candidate_types
            )
            RETURN c
            ORDER BY c.confidence_score DESC, c.mention_count DESC
            SKIP $offset
            LIMIT $limit
            """
            result = session.run(
                query,
                status=status,
                min_confidence=min_confidence,
                candidate_types=[t.value for t in candidate_types] if candidate_types else None,
                limit=limit,
                offset=offset,
            )
            return [self._convert_neo4j_temporals(dict(record["c"])) for record in result]

    def get_entity_candidate_statistics(self) -> Dict[str, Any]:
        """Compute basic statistics about EntityCandidate nodes."""
        with self.session() as session:
            totals = session.run(
                """
                MATCH (c:EntityCandidate)
                RETURN
                    count(c) as total,
                    count(CASE WHEN c.status = 'pending' THEN 1 END) as pending,
                    count(CASE WHEN c.status = 'approved' THEN 1 END) as approved,
                    count(CASE WHEN c.status = 'rejected' THEN 1 END) as rejected
                """
            ).single()

            by_type = session.run(
                """
                MATCH (c:EntityCandidate)
                RETURN c.candidate_type as candidate_type, count(c) as count
                ORDER BY count DESC
                """
            ).data()

            return {"totals": dict(totals), "by_type": by_type}

    def get_entity_candidate(self, query: str) -> Optional[Dict[str, Any]]:
        """Fetch a single EntityCandidate by id, candidate_key, or canonical_name."""
        with self.session() as session:
            result = session.run(
                """
                MATCH (c:EntityCandidate)
                WHERE c.id = $query
                   OR c.candidate_key = $query
                   OR toLower(c.canonical_name) = toLower($query)
                RETURN c
                LIMIT 1
                """,
                {"query": query},
            ).single()

            return self._convert_neo4j_temporals(dict(result["c"])) if result else None

    def search_entity_candidates(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search EntityCandidate nodes by query string."""
        with self.session() as session:
            cypher = """
            CALL db.index.fulltext.queryNodes('entity_candidate_fulltext', $query)
            YIELD node, score
            RETURN node as c, score
            ORDER BY score DESC
            LIMIT $limit
            """
            result = session.run(cypher, {"query": query, "limit": limit})
            items: List[Dict[str, Any]] = []
            for record in result:
                item = dict(record["c"])
                item["_score"] = float(record["score"])
                items.append(item)
            return items

    def find_entity_candidate_merge_suggestions(
        self, query: str, *, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Suggest potentially mergeable EntityCandidates via full-text search."""
        with self.session() as session:
            cypher = """
            CALL db.index.fulltext.queryNodes('entity_candidate_fulltext', $query)
            YIELD node, score
            RETURN node as c, score
            ORDER BY score DESC
            LIMIT $limit
            """
            result = session.run(cypher, {"query": query, "limit": limit})
            suggestions: List[Dict[str, Any]] = []
            for record in result:
                item = dict(record["c"])
                item["_score"] = float(record["score"])
                suggestions.append(item)
            return suggestions

    def update_entity_candidate_status(self, identifier: str, status: CandidateStatus) -> bool:
        """Update the status of an EntityCandidate by id or candidate_key."""
        with self.session() as session:
            result = session.run(
                """
                MATCH (c:EntityCandidate)
                WHERE c.id = $identifier OR c.candidate_key = $identifier
                SET c.status = $status, c.last_seen = datetime()
                RETURN c.candidate_key as candidate_key
                """,
                identifier=identifier,
                status=status.value,
            ).single()
            return bool(result)

    def update_entity_candidate(self, identifier: str, properties: Dict[str, Any]) -> bool:
        """Update arbitrary properties for an EntityCandidate."""
        normalized_props = dict(properties)
        if "status" in normalized_props and isinstance(normalized_props["status"], CandidateStatus):
            normalized_props["status"] = normalized_props["status"].value
        if "candidate_type" in normalized_props and isinstance(
            normalized_props["candidate_type"], EntityType
        ):
            normalized_props["candidate_type"] = normalized_props["candidate_type"].value

        with self.session() as session:
            result = session.run(
                """
                MATCH (c:EntityCandidate)
                WHERE c.id = $identifier OR c.candidate_key = $identifier
                SET c += $props
                RETURN c.candidate_key as candidate_key
                """,
                identifier=identifier,
                props=normalized_props,
            ).single()
            return bool(result)

    def get_relationship_candidates_involving_keys(
        self,
        keys: List[str],
        *,
        statuses: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return RelationshipCandidate nodes where source/target key matches any of the given keys."""
        if not keys:
            return []
        with self.session() as session:
            result = session.run(
                """
                MATCH (c:RelationshipCandidate)
                WHERE ($statuses IS NULL OR c.status IN $statuses)
                  AND any(k IN $keys WHERE c.candidate_key STARTS WITH (k + ':')
                                   OR c.candidate_key ENDS WITH (':' + k))
                RETURN c
                LIMIT $limit
                """,
                keys=keys,
                statuses=statuses,
                limit=limit,
            )
            return [self._convert_neo4j_temporals(dict(record["c"])) for record in result]

    def get_pending_relationships_with_peer_status(
        self,
        keys: List[str],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Find pending relationship candidates involving the given candidate-key fragments.
        Returns the relationship candidate plus the status of the *other* endpoint.

        Args:
            keys: List of normalized key fragments identifying the 'focal' entity.
        """
        if not keys:
            return []

        # RelationshipCandidate.candidate_key is built as "<source_key>:<type>:<target_key>".
        # Using candidate_key matching avoids brittle raw-string comparisons.
        with self.session() as session:
            query = """
            MATCH (rc:RelationshipCandidate {status: 'pending'})
            WHERE any(k IN $keys WHERE rc.candidate_key STARTS WITH (k + ':')
                                OR rc.candidate_key ENDS WITH (':' + k))

            WITH rc,
                 CASE
                    WHEN any(k IN $keys WHERE rc.candidate_key STARTS WITH (k + ':')) THEN
                        split(rc.candidate_key, ':')[-1]
                    ELSE
                        split(rc.candidate_key, ':')[0]
                 END as peer_key,
                 CASE
                    WHEN any(k IN $keys WHERE rc.candidate_key STARTS WITH (k + ':')) THEN rc.target
                    ELSE rc.source
                 END as peer_name

            // Check if peer exists as a Candidate (match by key OR name)
            OPTIONAL MATCH (peer_cand:EntityCandidate)
            WHERE peer_cand.candidate_key = peer_key
               OR toLower(peer_cand.canonical_name) = toLower(peer_name)
               OR any(a IN peer_cand.aliases WHERE toLower(a) = toLower(peer_name))

            // Check if peer exists as an Approved Entity (match by key OR name)
            OPTIONAL MATCH (peer_ent)
            WHERE any(l IN labels(peer_ent) WHERE l IN $entity_types)
              AND (peer_ent.candidate_key = peer_key
                   OR toLower(peer_ent.canonical_name) = toLower(peer_name)
                   OR any(a IN peer_ent.aliases WHERE toLower(a) = toLower(peer_name)))

            RETURN rc,
                   peer_name,
                   peer_cand.id as peer_candidate_id,
                   peer_cand.candidate_key as peer_candidate_key,
                   peer_cand.status as peer_candidate_status,
                   peer_ent.id as peer_entity_id,
                   peer_ent.status as peer_entity_status
            LIMIT $limit
            """
            entity_types = [et.value for et in EntityType]

            result = session.run(query, keys=keys, entity_types=entity_types, limit=limit)

            rows = []
            for record in result:
                row = {
                    "relationship": self._convert_neo4j_temporals(dict(record["rc"])),
                    "peer_name": record["peer_name"],
                    "peer_candidate": None,
                    "peer_entity": None,
                }

                if record["peer_candidate_key"]:
                    row["peer_candidate"] = {
                        "id": record["peer_candidate_id"],
                        "candidate_key": record["peer_candidate_key"],
                        "status": record["peer_candidate_status"],
                    }

                if record["peer_entity_id"]:
                    row["peer_entity"] = {
                        "id": record["peer_entity_id"],
                        "status": record["peer_entity_status"],
                    }
                rows.append(row)

            return rows

    def update_relationship_candidate_status(
        self, identifier: str, status: CandidateStatus
    ) -> bool:
        """Update the status of a RelationshipCandidate by id or candidate_key."""
        with self.session() as session:
            result = session.run(
                """
                MATCH (c:RelationshipCandidate)
                WHERE c.id = $identifier OR c.candidate_key = $identifier
                SET c.status = $status, c.last_seen = datetime()
                RETURN c.candidate_key as candidate_key
                """,
                identifier=identifier,
                status=status.value,
            ).single()
            return bool(result)

    def upsert_relationship(self, relationship: Relationship) -> str:
        """Create or update a relationship (idempotent by relationship.id)."""
        with self.session() as session:
            cypher = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            WHERE any(label IN labels(source) WHERE label IN $entity_types)
              AND any(label IN labels(target) WHERE label IN $entity_types)
            MERGE (source)-[r:{relationship.type.value} {{id: $rel_id}}]->(target)
            SET r += $props
            RETURN r.id as id
            """
            record = session.run(
                cypher,
                source_id=relationship.source_entity_id,
                target_id=relationship.target_entity_id,
                rel_id=relationship.id,
                props=relationship.to_neo4j_dict(),
                entity_types=[et.value for et in EntityType],
            ).single()
            if not record:
                raise Neo4jError(
                    f"Could not upsert relationship: source or target entity not found "
                    f"({relationship.source_entity_id} -> {relationship.target_entity_id})"
                )
            return record["id"]

    def upsert_relationship_candidate_aggregate(self, candidate: RelationshipCandidate) -> str:
        """Upsert a RelationshipCandidate, aggregating mention counts and seen locations."""
        with self.session() as session:
            now = datetime.now().isoformat()
            candidate_id = candidate.id or self._deterministic_uuid(
                f"relationship_candidate:{candidate.candidate_key}"
            )

            props = candidate.to_neo4j_dict()
            props["id"] = candidate_id
            props["last_seen"] = now

            query = """
            MERGE (c:RelationshipCandidate {candidate_key: $candidate_key})
            ON CREATE SET c += $props
            ON MATCH SET
                c.last_seen = $last_seen,
                c.mention_count = coalesce(c.mention_count, 0) + $mention_count,
                c.confidence_score = CASE
                    WHEN $confidence_score > coalesce(c.confidence_score, 0.0)
                    THEN $confidence_score
                    ELSE coalesce(c.confidence_score, 0.0)
                END
            WITH c
            SET c.source_documents = coalesce(c.source_documents, []) +
                [d IN $source_documents WHERE NOT d IN coalesce(c.source_documents, [])]
            SET c.chunk_ids = coalesce(c.chunk_ids, []) +
                [ch IN $chunk_ids WHERE NOT ch IN coalesce(c.chunk_ids, [])]
            SET c.provenance_events = coalesce(c.provenance_events, []) +
                [p IN $provenance_events WHERE NOT p IN coalesce(c.provenance_events, [])]
            WITH c
            FOREACH (doc_id IN $source_documents |
                MERGE (d:DOCUMENT {id: doc_id})
                MERGE (c)-[:MENTIONED_IN]->(d)
            )
            RETURN c.id as id
            """
            result = session.run(
                query,
                candidate_key=candidate.candidate_key,
                props=props,
                last_seen=now,
                mention_count=int(candidate.mention_count),
                confidence_score=float(candidate.confidence_score),
                source_documents=list(candidate.source_documents),
                chunk_ids=list(candidate.chunk_ids),
                provenance_events=list(candidate.provenance_events),
            )
            return result.single()["id"]

    def get_relationship_candidates(
        self,
        *,
        status: Optional[str] = None,
        rel_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query RelationshipCandidate nodes by status/type/confidence."""
        with self.session() as session:
            query = """
            MATCH (c:RelationshipCandidate)
            WHERE ($status IS NULL OR c.status = $status)
            AND ($rel_type IS NULL OR c.type = $rel_type)
            AND ($min_confidence IS NULL OR c.confidence_score >= $min_confidence)
            RETURN c
            ORDER BY c.confidence_score DESC, c.mention_count DESC
            SKIP $offset
            LIMIT $limit
            """
            result = session.run(
                query,
                status=status,
                rel_type=rel_type,
                min_confidence=min_confidence,
                limit=limit,
                offset=offset,
            )
            return [self._convert_neo4j_temporals(dict(record["c"])) for record in result]

    def get_relationship_candidate(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Fetch a single RelationshipCandidate by id or candidate_key."""
        with self.session() as session:
            result = session.run(
                """
                MATCH (c:RelationshipCandidate)
                WHERE c.id = $identifier OR c.candidate_key = $identifier
                RETURN c
                """,
                identifier=identifier,
            ).single()

            if result:
                return self._convert_neo4j_temporals(dict(result["c"]))
            return None

    def drop_schema(self) -> None:
        """Drop all constraints and indexes (use with caution)."""
        with self.session() as session:
            # Drop all constraints
            constraints = session.run("SHOW CONSTRAINTS").data()
            for constraint in constraints:
                try:
                    session.run(f"DROP CONSTRAINT {constraint['name']} IF EXISTS")
                    logger.info(f"Dropped constraint {constraint['name']}")
                except Neo4jError as e:
                    logger.warning(f"Could not drop constraint {constraint['name']}: {e}")

            # Drop all indexes
            indexes = session.run("SHOW INDEXES").data()
            for index in indexes:
                try:
                    session.run(f"DROP INDEX {index['name']} IF EXISTS")
                    logger.info(f"Dropped index {index['name']}")
                except Neo4jError as e:
                    logger.warning(f"Could not drop index {index['name']}: {e}")

            logger.info("Neo4j schema dropped")

    # Entity CRUD Operations

    def create_entity(self, entity: Entity) -> str:
        """Create an entity node in Neo4j.

        Note:
            This uses CREATE and will fail if an entity with the same ID already exists
            (given uniqueness constraints). For idempotent behavior, use
            upsert_entity().

        Args:
            entity: Entity instance to create

        Returns:
            Entity ID

        Raises:
            Neo4jError: If entity creation fails
        """
        with self.session() as session:
            query = f"""
            CREATE (n:Entity:{entity.entity_type.value} $props)
            RETURN n.id as id
            """
            result = session.run(query, props=entity.to_neo4j_dict())
            entity_id = result.single()["id"]
            logger.debug(f"Created entity {entity_id} of type {entity.entity_type.value}")
            return entity_id

    def upsert_entity(self, entity: Entity) -> str:
        """Create or update an entity node in Neo4j (idempotent).

        Uses MERGE on id and overwrites properties with the provided values.
        Also creates MENTIONED_IN relationships to source documents.

        Args:
            entity: Entity instance to upsert

        Returns:
            Entity ID
        """
        with self.session() as session:
            query = f"""
            MERGE (n:Entity:{entity.entity_type.value} {{id: $entity_id}})
            SET n += $props
            WITH n
            FOREACH (doc_id IN $source_documents |
                MERGE (d:DOCUMENT {{id: doc_id}})
                MERGE (n)-[:MENTIONED_IN]->(d)
            )
            RETURN n.id as id
            """
            props = entity.to_neo4j_dict()
            result = session.run(
                query,
                entity_id=entity.id,
                props=props,
                source_documents=entity.source_documents,
            )
            entity_id = result.single()["id"]
            logger.debug(f"Upserted entity {entity_id} of type {entity.entity_type.value}")
            return entity_id

    def get_entity(
        self, entity_id: str, entity_type: Optional[EntityType] = None
    ) -> Optional[Dict[str, Any]]:
        """Get an entity by ID.

        Args:
            entity_id: Entity ID
            entity_type: Optional entity type for optimization

        Returns:
            Entity properties as dictionary, or None if not found
        """
        with self.session() as session:
            if entity_type:
                query = f"""
                MATCH (n:{entity_type.value} {{id: $entity_id}})
                RETURN n
                """
            else:
                query = """
                MATCH (n {id: $entity_id})
                WHERE any(label IN labels(n) WHERE label IN $entity_types)
                RETURN n
                """
            result = session.run(
                query,
                entity_id=entity_id,
                entity_types=[et.value for et in EntityType] if not entity_type else None,
            )
            record = result.single()
            if record:
                return dict(record["n"])
            return None

    def get_entity_by_canonical_name(
        self, canonical_name: str, entity_type: Optional[EntityType] = None
    ) -> Optional[Dict[str, Any]]:
        """Get an entity by canonical name.

        Args:
            canonical_name: Canonical name of entity
            entity_type: Optional entity type for optimization

        Returns:
            Entity properties as dictionary, or None if not found
        """
        with self.session() as session:
            if entity_type:
                query = f"""
                MATCH (n:{entity_type.value} {{canonical_name: $canonical_name}})
                RETURN n
                """
            else:
                query = """
                MATCH (n {canonical_name: $canonical_name})
                WHERE any(label IN labels(n) WHERE label IN $entity_types)
                RETURN n
                """
            result = session.run(
                query,
                canonical_name=canonical_name,
                entity_types=[et.value for et in EntityType] if not entity_type else None,
            )
            record = result.single()
            if record:
                return dict(record["n"])
            return None

    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Update entity properties.

        Args:
            entity_id: Entity ID
            properties: Properties to update

        Returns:
            True if entity was updated, False if not found
        """
        with self.session() as session:
            query = """
            MATCH (n {id: $entity_id})
            WHERE any(label IN labels(n) WHERE label IN $entity_types)
            SET n += $properties
            RETURN n.id as id
            """
            result = session.run(
                query,
                entity_id=entity_id,
                properties=properties,
                entity_types=[et.value for et in EntityType],
            )
            if result.single():
                logger.debug(f"Updated entity {entity_id}")
                return True
            return False

    def create_mentioned_in_relationships(
        self, entity_id: str, document_ids: List[str]
    ) -> int:
        """Create MENTIONED_IN relationships between an entity and documents.

        This is used when merging candidates into existing entities to ensure
        cross-document mentions are properly linked in the graph.

        Args:
            entity_id: Entity ID
            document_ids: List of document IDs to link

        Returns:
            Number of relationships created
        """
        if not document_ids:
            return 0

        with self.session() as session:
            query = """
            MATCH (n {id: $entity_id})
            WHERE any(label IN labels(n) WHERE label IN $entity_types)
            WITH n
            UNWIND $doc_ids AS doc_id
            MERGE (d:DOCUMENT {id: doc_id})
            MERGE (n)-[r:MENTIONED_IN]->(d)
            RETURN count(r) as created
            """
            result = session.run(
                query,
                entity_id=entity_id,
                doc_ids=document_ids,
                entity_types=[et.value for et in EntityType],
            )
            record = result.single()
            created = int(record["created"]) if record else 0
            if created > 0:
                logger.debug(
                    f"Created {created} MENTIONED_IN relationships for entity {entity_id}"
                )
            return created

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity node and its relationships.

        This is primarily used by curation undo operations.
        """
        with self.session() as session:
            query = """
            MATCH (n {id: $entity_id})
            WHERE any(label IN labels(n) WHERE label IN $entity_types)
            DETACH DELETE n
            RETURN count(n) as deleted
            """
            record = session.run(
                query, entity_id=entity_id, entity_types=[et.value for et in EntityType]
            ).single()
            deleted = int(record["deleted"]) if record else 0
            if deleted > 0:
                logger.debug(f"Deleted entity {entity_id}")
                return True
            return False

    def deactivate_document_data(self, document_id: str) -> bool:
        """Mark all chunks and relationship candidates for a document as inactive.

        This is used during incremental updates to 'soft-delete' old data before
        re-ingesting a modified document.

        Args:
            document_id: Document ID

        Returns:
            True if data was deactivated
        """
        with self.session() as session:
            query = """
            MATCH (c:Chunk {document_id: $document_id})
            SET c.status = 'inactive'
            WITH c
            MATCH (rc:RelationshipCandidate)
            WHERE $document_id IN rc.source_documents
            SET rc.status = 'inactive'
            RETURN count(c) as deactivated_chunks
            """
            session.run(query, document_id=document_id)
            return True

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks.

        Args:
            document_id: Document ID

        Returns:
            True if document was deleted, False if not found
        """
        with self.session() as session:
            # Delete chunks first, then the document itself
            query = """
            MATCH (d:DOCUMENT {id: $document_id})
            OPTIONAL MATCH (c:Chunk {document_id: $document_id})
            DETACH DELETE c, d
            RETURN count(d) as deleted
            """
            result = session.run(query, document_id=document_id)
            deleted = result.single()["deleted"]
            if deleted > 0:
                logger.debug(f"Deleted document {document_id} and its chunks")
                return True
            return False

    def search_entities(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
        status: Optional[EntityStatus] = None,
    ) -> List[Dict[str, Any]]:
        """Search entities using full-text search.

        Args:
            query: Search query
            entity_types: Optional list of entity types to filter
            limit: Maximum number of results
            status: Optional status filter

        Returns:
            List of matching entities with scores
        """
        with self.session() as session:
            cypher_query = """
            CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
            YIELD node, score
            """

            # Add entity type filter
            if entity_types:
                cypher_query += """
                WHERE any(label IN labels(node) WHERE label IN $entity_types)
                """
            else:
                cypher_query += """
                WHERE any(label IN labels(node) WHERE label IN $all_entity_types)
                """

            # Add status filter
            if status:
                cypher_query += """
                AND node.status = $status
                """

            cypher_query += """
            RETURN node, score
            ORDER BY score DESC
            LIMIT $limit
            """

            result = session.run(
                cypher_query,
                {
                    "query": query,
                    "entity_types": [et.value for et in entity_types] if entity_types else None,
                    "all_entity_types": [et.value for et in EntityType],
                    "status": status.value if status else None,
                    "limit": limit,
                },
            )

            entities = []
            for record in result:
                entity_dict = dict(record["node"])
                entity_dict["search_score"] = record["score"]
                entities.append(entity_dict)

            return entities

    def list_entities(
        self,
        entity_type: Optional[EntityType] = None,
        status: Optional[EntityStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List entities with optional filtering.

        Args:
            entity_type: Optional entity type filter
            status: Optional status filter
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of entities
        """
        with self.session() as session:
            if entity_type:
                query = f"""
                MATCH (n:{entity_type.value})
                """
            else:
                query = """
                MATCH (n)
                WHERE any(label IN labels(n) WHERE label IN $entity_types)
                """

            if status:
                if entity_type:
                    query += " WHERE n.status = $status"
                else:
                    query += " AND n.status = $status"

            query += """
            RETURN n
            ORDER BY n.canonical_name
            SKIP $offset
            LIMIT $limit
            """

            result = session.run(
                query,
                entity_types=[et.value for et in EntityType] if not entity_type else None,
                status=status.value if status else None,
                limit=limit,
                offset=offset,
            )

            return [dict(record["n"]) for record in result]

    # Relationship CRUD Operations

    def create_relationship(self, relationship: Relationship) -> str:
        """Create a relationship between two entities.

        Args:
            relationship: Relationship instance to create

        Returns:
            Relationship ID

        Raises:
            Neo4jError: If relationship creation fails or entities don't exist
        """
        with self.session() as session:
            query = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            WHERE any(label IN labels(source) WHERE label IN $entity_types)
            AND any(label IN labels(target) WHERE label IN $entity_types)
            CREATE (source)-[r:{relationship.type.value} $props]->(target)
            RETURN r.id as id
            """
            result = session.run(
                query,
                source_id=relationship.source_entity_id,
                target_id=relationship.target_entity_id,
                props=relationship.to_neo4j_dict(),
                entity_types=[et.value for et in EntityType],
            )

            record = result.single()
            if not record:
                raise Neo4jError(
                    f"Could not create relationship: source or target entity not found "
                    f"({relationship.source_entity_id} -> {relationship.target_entity_id})"
                )

            relationship_id = record["id"]
            logger.debug(
                f"Created relationship {relationship_id} of type {relationship.type.value} "
                f"({relationship.source_entity_id} -> {relationship.target_entity_id})"
            )
            return relationship_id

    def get_relationship(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Get a relationship by ID.

        Args:
            relationship_id: Relationship ID

        Returns:
            Relationship properties including source and target IDs, or None if not found
        """
        with self.session() as session:
            query = """
            MATCH (source)-[r {id: $relationship_id}]->(target)
            RETURN r, source.id as source_id, target.id as target_id
            """
            result = session.run(query, relationship_id=relationship_id)
            record = result.single()
            if record:
                rel_dict = dict(record["r"])
                rel_dict["source_entity_id"] = record["source_id"]
                rel_dict["target_entity_id"] = record["target_id"]
                return rel_dict
            return None

    def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[RelationshipType] = None,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        """Get relationships for an entity.

        Args:
            entity_id: Entity ID
            relationship_type: Optional relationship type filter
            direction: Direction of relationships ('outgoing', 'incoming', 'both')

        Returns:
            List of relationships with source and target information
        """
        with self.session() as session:
            if direction == "outgoing":
                match_pattern = "(source {id: $entity_id})-[r]->(target)"
            elif direction == "incoming":
                match_pattern = "(source)-[r]->(target {id: $entity_id})"
            else:  # both
                match_pattern = (
                    "(source)-[r]-(target) WHERE source.id = $entity_id OR target.id = $entity_id"
                )

            query = f"""
            MATCH {match_pattern}
            """

            if relationship_type:
                query += """
                WHERE type(r) = $rel_type
                """

            query += """
            RETURN r, source.id as source_id, source.canonical_name as source_name,
                   target.id as target_id, target.canonical_name as target_name,
                   type(r) as rel_type
            """

            result = session.run(
                query,
                entity_id=entity_id,
                rel_type=relationship_type.value if relationship_type else None,
            )

            relationships = []
            for record in result:
                rel_dict = dict(record["r"])
                rel_dict["source_entity_id"] = record["source_id"]
                rel_dict["source_name"] = record["source_name"]
                rel_dict["target_entity_id"] = record["target_id"]
                rel_dict["target_name"] = record["target_name"]
                rel_dict["type"] = record["rel_type"]
                relationships.append(rel_dict)

            return relationships

    def update_relationship(self, relationship_id: str, properties: Dict[str, Any]) -> bool:
        """Update relationship properties.

        Args:
            relationship_id: Relationship ID
            properties: Properties to update

        Returns:
            True if relationship was updated, False if not found
        """
        with self.session() as session:
            query = """
            MATCH ()-[r {id: $relationship_id}]->()
            SET r += $properties
            RETURN r.id as id
            """
            result = session.run(query, relationship_id=relationship_id, properties=properties)
            if result.single():
                logger.debug(f"Updated relationship {relationship_id}")
                return True
            return False

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship.

        Args:
            relationship_id: Relationship ID

        Returns:
            True if relationship was deleted, False if not found
        """
        with self.session() as session:
            query = """
            MATCH ()-[r {id: $relationship_id}]->()
            DELETE r
            RETURN count(r) as deleted
            """
            result = session.run(query, relationship_id=relationship_id)
            deleted = result.single()["deleted"]
            if deleted > 0:
                logger.debug(f"Deleted relationship {relationship_id}")
                return True
            return False

    # Chunk Operations

    def create_chunk(self, chunk: Chunk) -> str:
        """Create a chunk node in Neo4j.

        Note:
            This uses CREATE and will fail if a chunk with the same ID already exists
            (given uniqueness constraints). For idempotent behavior, use
            upsert_chunk().

        Args:
            chunk: Chunk instance to create

        Returns:
            Chunk ID
        """
        with self.session() as session:
            query = """
            CREATE (c:Chunk $props)
            RETURN c.id as id
            """
            result = session.run(query, props=chunk.to_neo4j_dict())
            chunk_id = result.single()["id"]
            logger.debug(f"Created chunk {chunk_id}")
            return chunk_id

    def upsert_chunk(self, chunk: Chunk) -> str:
        """Create or update a chunk node in Neo4j (idempotent).

        Uses MERGE on id and overwrites properties with the provided values.
        Also creates structural relationships:
        - (Chunk)-[:PART_OF]->(Document)
        - (Chunk)-[:PARENT_CHUNK]->(ParentChunk)

        Args:
            chunk: Chunk instance to upsert

        Returns:
            Chunk ID
        """
        with self.session() as session:
            query = """
            MERGE (c:Chunk {id: $chunk_id})
            SET c += $props
            WITH c
            MATCH (d:DOCUMENT {id: $document_id})
            MERGE (c)-[:PART_OF]->(d)
            WITH c
            CALL {
                WITH c
                WITH c
                WHERE $parent_chunk_id IS NOT NULL
                MATCH (p:Chunk {id: $parent_chunk_id})
                MERGE (c)-[:PARENT_CHUNK]->(p)
                RETURN count(*) as _
            }
            RETURN c.id as id
            """
            props = chunk.to_neo4j_dict()
            result = session.run(
                query,
                chunk_id=chunk.id,
                props=props,
                document_id=chunk.document_id,
                parent_chunk_id=chunk.parent_chunk_id,
            )
            chunk_id = result.single()["id"]
            logger.debug(f"Upserted chunk {chunk_id}")
            return chunk_id

    def delete_chunks_by_document(self, document_id: str) -> int:
        """Delete all Chunk nodes belonging to a document.

        Args:
            document_id: Document ID

        Returns:
            Number of deleted Chunk nodes.
        """
        with self.session() as session:
            query = """
            MATCH (c:Chunk {document_id: $document_id})
            WITH c
            DETACH DELETE c
            RETURN count(*) as deleted
            """
            result = session.run(query, document_id=document_id)
            deleted = result.single()["deleted"]
            logger.debug(f"Deleted {deleted} chunks for document {document_id}")
            return int(deleted)

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk properties as dictionary, or None if not found
        """
        with self.session() as session:
            query = """
            MATCH (c:Chunk {id: $chunk_id})
            RETURN c
            """
            result = session.run(query, chunk_id=chunk_id)
            record = result.single()
            if record:
                return dict(record["c"])
            return None

    def get_chunks_by_document(
        self, document_id: str, level: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a document.

        Args:
            document_id: Document ID
            level: Optional level filter

        Returns:
            List of chunks
        """
        with self.session() as session:
            query = """
            MATCH (c:Chunk {document_id: $document_id})
            """

            if level is not None:
                query += """
                WHERE c.level = $level
                """

            query += """
            RETURN c
            ORDER BY c.level, c.hierarchy_path
            """

            result = session.run(query, document_id=document_id, level=level)
            return [self._convert_neo4j_temporals(dict(record["c"])) for record in result]

    # Graph Traversal Operations

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
        relationship_types: Optional[List[RelationshipType]] = None,
    ) -> List[Dict[str, Any]]:
        """Find paths between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_depth: Maximum path length
            relationship_types: Optional list of relationship types to traverse

        Returns:
            List of paths (each path is a dict with nodes and relationships)
        """
        with self.session() as session:
            if relationship_types:
                rel_types = "|".join([rt.value for rt in relationship_types])
                match_pattern = f"(source {{id: $source_id}})-[r:{rel_types}*1..{max_depth}]-(target {{id: $target_id}})"
            else:
                match_pattern = (
                    f"(source {{id: $source_id}})-[r*1..{max_depth}]-(target {{id: $target_id}})"
                )

            query = f"""
            MATCH path = {match_pattern}
            RETURN path
            LIMIT 10
            """

            result = session.run(query, source_id=source_id, target_id=target_id)

            paths = []
            for record in result:
                path = record["path"]
                path_dict = {
                    "nodes": [dict(node) for node in path.nodes],
                    "relationships": [dict(rel) for rel in path.relationships],
                    "length": len(path.relationships),
                }
                paths.append(path_dict)

            return paths

    def traverse_relationships(
        self,
        entity_id: str,
        relationship_types: List[RelationshipType],
        max_depth: int = 3,
        direction: str = "outgoing",
    ) -> List[Dict[str, Any]]:
        """Traverse relationships from an entity.

        Args:
            entity_id: Starting entity ID
            relationship_types: List of relationship types to traverse
            max_depth: Maximum traversal depth
            direction: Direction of traversal ('outgoing', 'incoming', 'both')

        Returns:
            List of reached entities with their paths
        """
        with self.session() as session:
            rel_types = "|".join([rt.value for rt in relationship_types])

            if direction == "outgoing":
                match_pattern = f"(start {{id: $entity_id}})-[r:{rel_types}*1..{max_depth}]->(end)"
            elif direction == "incoming":
                match_pattern = f"(start {{id: $entity_id}})<-[r:{rel_types}*1..{max_depth}]-(end)"
            else:  # both
                match_pattern = f"(start {{id: $entity_id}})-[r:{rel_types}*1..{max_depth}]-(end)"

            query = f"""
            MATCH path = {match_pattern}
            RETURN DISTINCT end, length(path) as depth
            ORDER BY depth
            """

            result = session.run(query, entity_id=entity_id)

            entities = []
            for record in result:
                entity_dict = dict(record["end"])
                entity_dict["depth"] = record["depth"]
                entities.append(entity_dict)

            return entities

    # Utility Methods

    def health_check(self) -> bool:
        """Check if Neo4j connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            with self.session() as session:
                result = session.run("RETURN 1")
                return result.single() is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with entity and relationship counts
        """
        with self.session() as session:
            # Count entities by type
            entity_counts = {}
            for entity_type in EntityType:
                query = f"""
                MATCH (n:{entity_type.value})
                RETURN count(n) as count
                """
                result = session.run(query)
                entity_counts[entity_type.value] = result.single()["count"]

            # Count relationships by type
            relationship_counts = {}
            for rel_type in RelationshipType:
                query = f"""
                MATCH ()-[r:{rel_type.value}]->()
                RETURN count(r) as count
                """
                result = session.run(query)
                relationship_counts[rel_type.value] = result.single()["count"]

            # Count chunks
            query = """
            MATCH (c:Chunk)
            RETURN count(c) as count
            """
            result = session.run(query)
            chunk_count = result.single()["count"]

            # Total counts
            query = """
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label IN $entity_types)
            RETURN count(n) as count
            """
            result = session.run(query, entity_types=[et.value for et in EntityType])
            total_entities = result.single()["count"]

            query = """
            MATCH ()-[r]->()
            RETURN count(r) as count
            """
            result = session.run(query)
            total_relationships = result.single()["count"]

            return {
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "total_chunks": chunk_count,
                "entities_by_type": entity_counts,
                "relationships_by_type": relationship_counts,
            }

    def get_existing_relationship_types(self) -> List[str]:
        """Get list of relationship types that actually exist in the database.

        Returns:
            List of relationship type strings
        """
        with self.session() as session:
            result = session.run(
                """
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN relationshipType
                """
            )
            return [record["relationshipType"] for record in result]

    def clear_database(self) -> None:
        """Clear all nodes and relationships from database (use with caution).

        Warning:
            This will delete all data in the database!
        """
        with self.session() as session:
            query = """
            MATCH (n)
            DETACH DELETE n
            """
            session.run(query)
            logger.warning("Cleared all data from Neo4j database")

    def execute_cypher(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query.

        Args:
            query: Cypher query string
            parameters: Optional query parameters

        Returns:
            List of result records as dictionaries
        """
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

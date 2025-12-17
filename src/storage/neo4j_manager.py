"""Neo4j graph database manager for entity and relationship storage."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

from neo4j import GraphDatabase, Session
from neo4j.exceptions import Neo4jError

from src.storage.schemas import (
    Chunk,
    Entity,
    EntityStatus,
    EntityType,
    Relationship,
    RelationshipType,
)
from src.utils.config import DatabaseConfig

logger = logging.getLogger(__name__)


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

    def connect(self) -> None:
        """Establish connection to Neo4j database.

        Raises:
            Neo4jError: If connection fails
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password), max_connection_pool_size=50
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Neo4jError as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

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

            # Create index on entity_type (for querying across all entity types)
            try:
                session.run(
                    "CREATE INDEX entity_type_idx IF NOT EXISTS "
                    "FOR (n) ON (n.entity_type)"
                )
                logger.info("Created entity_type index")
            except Neo4jError as e:
                logger.warning(f"Could not create entity_type index: {e}")

            # Create index on status for filtering
            try:
                session.run("CREATE INDEX entity_status_idx IF NOT EXISTS " "FOR (n) ON (n.status)")
                logger.info("Created status index")
            except Neo4jError as e:
                logger.warning(f"Could not create status index: {e}")

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

            # Create index on document_id for chunks
            try:
                session.run(
                    "CREATE INDEX chunk_document_id IF NOT EXISTS " "FOR (c:Chunk) ON (c.document_id)"
                )
                logger.info("Created document_id index for chunks")
            except Neo4jError as e:
                logger.warning(f"Could not create document_id index: {e}")

            # Create index on relationship type
            try:
                session.run("CREATE INDEX rel_type_idx IF NOT EXISTS " "FOR ()-[r]-() ON (r.type)")
                logger.info("Created relationship type index")
            except Neo4jError as e:
                logger.warning(f"Could not create relationship type index: {e}")

            logger.info("Neo4j schema creation completed")

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
            CREATE (n:{entity.entity_type.value} $props)
            RETURN n.id as id
            """
            result = session.run(query, props=entity.to_neo4j_dict())
            entity_id = result.single()["id"]
            logger.debug(f"Created entity {entity_id} of type {entity.entity_type.value}")
            return entity_id

    def upsert_entity(self, entity: Entity) -> str:
        """Create or update an entity node in Neo4j (idempotent).

        Uses MERGE on id and overwrites properties with the provided values.

        Args:
            entity: Entity instance to upsert

        Returns:
            Entity ID
        """
        with self.session() as session:
            query = f"""
            MERGE (n:{entity.entity_type.value} {{id: $entity_id}})
            SET n += $props
            RETURN n.id as id
            """
            props = entity.to_neo4j_dict()
            result = session.run(query, entity_id=entity.id, props=props)
            entity_id = result.single()["id"]
            logger.debug(f"Upserted entity {entity_id} of type {entity.entity_type.value}")
            return entity_id

    def get_entity(self, entity_id: str, entity_type: Optional[EntityType] = None) -> Optional[Dict[str, Any]]:
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
                query, entity_id=entity_id, properties=properties, entity_types=[et.value for et in EntityType]
            )
            if result.single():
                logger.debug(f"Updated entity {entity_id}")
                return True
            return False

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships.

        Args:
            entity_id: Entity ID

        Returns:
            True if entity was deleted, False if not found
        """
        with self.session() as session:
            query = """
            MATCH (n {id: $entity_id})
            WHERE any(label IN labels(n) WHERE label IN $entity_types)
            DETACH DELETE n
            RETURN count(n) as deleted
            """
            result = session.run(query, entity_id=entity_id, entity_types=[et.value for et in EntityType])
            deleted = result.single()["deleted"]
            if deleted > 0:
                logger.debug(f"Deleted entity {entity_id}")
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
                type_labels = [et.value for et in entity_types]
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
                query=query,
                entity_types=[et.value for et in entity_types] if entity_types else None,
                all_entity_types=[et.value for et in EntityType],
                status=status.value if status else None,
                limit=limit,
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
                query += """
                WHERE n.status = $status
                """

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
                match_pattern = "(source)-[r]-(target) WHERE source.id = $entity_id OR target.id = $entity_id"

            query = f"""
            MATCH {match_pattern}
            """

            if relationship_type:
                query += f"""
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

        Args:
            chunk: Chunk instance to upsert

        Returns:
            Chunk ID
        """
        with self.session() as session:
            query = """
            MERGE (c:Chunk {id: $chunk_id})
            SET c += $props
            RETURN c.id as id
            """
            props = chunk.to_neo4j_dict()
            result = session.run(query, chunk_id=chunk.id, props=props)
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

    def get_chunks_by_document(self, document_id: str, level: Optional[int] = None) -> List[Dict[str, Any]]:
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
            return [dict(record["c"]) for record in result]

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
                match_pattern = f"(source {{id: $source_id}})-[r*1..{max_depth}]-(target {{id: $target_id}})"

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

    def execute_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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

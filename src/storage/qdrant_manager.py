"""Qdrant vector database manager for document chunks and entity embeddings.

This module provides a wrapper around the Qdrant client for managing vector collections,
performing similarity searches, and handling batch operations for the Graph RAG system.

Collections:
    - document_chunks: Stores embeddings for hierarchical document chunks
    - entities: Stores embeddings for entity descriptions

Features:
    - Collection creation with HNSW indexing
    - Vector upsert and search operations
    - Batch operations for efficient processing
    - Connection pooling
    - Health checks and monitoring
    - Error handling and retry logic
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from ..utils.config import DatabaseConfig


class QdrantManager:
    """Manager class for Qdrant vector database operations.

    Handles collection management, vector operations, and search functionality
    for document chunks and entity embeddings.

    Attributes:
        client: Qdrant client instance
        config: Database configuration
        chunk_collection: Name of the document chunks collection
        entity_collection: Name of the entities collection
    """

    def __init__(
        self,
        config: DatabaseConfig,
        chunk_collection: str = "document_chunks",
        entity_collection: str = "entities",
    ) -> None:
        """Initialize Qdrant manager with configuration.

        Args:
            config: Database configuration with Qdrant connection details
            chunk_collection: Name for document chunks collection
            entity_collection: Name for entities collection

        Raises:
            ConnectionError: If unable to connect to Qdrant
        """
        self.config = config
        self.chunk_collection = chunk_collection
        self.entity_collection = entity_collection

        # Initialize Qdrant client with connection pooling
        try:
            # Local/in-memory mode (used by unit tests, no server required)
            if getattr(config, "qdrant_location", ""):
                self.client = QdrantClient(
                    location=config.qdrant_location,
                    api_key=config.qdrant_api_key or None,
                    timeout=30.0,
                    prefer_grpc=False,
                )
                logger.info(f"Connected to Qdrant in local mode at {config.qdrant_location}")
            else:
                # Remote mode
                prefer_grpc = bool(getattr(config, "qdrant_prefer_grpc", False))
                grpc_port = int(getattr(config, "qdrant_grpc_port", 6334))

                kwargs: Dict[str, Any] = {
                    "host": config.qdrant_host,
                    "port": config.qdrant_port,
                    "https": config.qdrant_https,
                    "timeout": 30.0,
                    "prefer_grpc": prefer_grpc,
                }
                if prefer_grpc:
                    kwargs["grpc_port"] = grpc_port
                if config.qdrant_api_key:
                    kwargs["api_key"] = config.qdrant_api_key

                self.client = QdrantClient(**kwargs)

                logger.info(
                    f"Connected to Qdrant at {config.qdrant_host}:{config.qdrant_port} (prefer_grpc={prefer_grpc})"
                )

            # Verify connection
            self.client.get_collections()

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Unable to connect to Qdrant: {e}") from e

    def create_collections(
        self,
        recreate: bool = False,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
    ) -> None:
        """Create or recreate Qdrant collections with proper configuration.

        Creates two collections:
        1. document_chunks: For hierarchical document chunk embeddings
        2. entities: For entity description embeddings

        Both collections use:
        - Cosine similarity as distance metric
        - HNSW indexing for fast approximate search
        - Payload indexes for efficient filtering

        Args:
            recreate: If True, delete existing collections before creating
            hnsw_m: Number of edges per node in HNSW graph (default: 16)
                   Higher values = better recall but more memory
            hnsw_ef_construct: Size of dynamic candidate list for construction (default: 100)
                              Higher values = better quality but slower indexing

        Raises:
            Exception: If collection creation fails
        """
        try:
            # Delete collections if recreate is True
            if recreate:
                self._delete_collection_if_exists(self.chunk_collection)
                self._delete_collection_if_exists(self.entity_collection)
                logger.info("Deleted existing collections for recreation")

            # Create document_chunks collection
            self._create_chunk_collection(hnsw_m, hnsw_ef_construct)

            # Create entities collection
            self._create_entity_collection(hnsw_m, hnsw_ef_construct)

            logger.info("Successfully created Qdrant collections")

        except Exception as e:
            logger.error(f"Failed to create collections: {e}")
            raise

    def _delete_collection_if_exists(self, collection_name: str) -> None:
        """Delete a collection if it exists.

        Args:
            collection_name: Name of the collection to delete
        """
        try:
            collections = self.client.get_collections().collections
            if any(col.name == collection_name for col in collections):
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Error checking/deleting collection {collection_name}: {e}")

    def _create_chunk_collection(self, hnsw_m: int, hnsw_ef_construct: int) -> None:
        """Create the document_chunks collection.

        Collection stores embeddings for hierarchical document chunks with metadata
        including document references, hierarchy levels, and entity associations.

        Payload schema:
        - chunk_id: UUID of the chunk
        - document_id: UUID of the source document
        - level: Hierarchy level (1=document, 2=section, 3=subsection, 4=paragraph)
        - content: Text content of the chunk
        - metadata: Dict with document_title, section_title, page_numbers, etc.
        - entity_ids: List of entity UUIDs mentioned in the chunk
        - timestamp: ISO format timestamp

        Args:
            hnsw_m: Number of edges per node in HNSW graph
            hnsw_ef_construct: Size of dynamic candidate list for construction
        """
        if not self.collection_exists(self.chunk_collection):
            self.client.create_collection(
                collection_name=self.chunk_collection,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimension,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=hnsw_m,
                        ef_construct=hnsw_ef_construct,
                        full_scan_threshold=10000,  # Use full scan for small collections
                    ),
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20000,  # Start indexing after 20k vectors
                    memmap_threshold=50000,  # Use memory mapping after 50k vectors
                ),
                # Enable on-disk storage for large collections
                on_disk_payload=False,  # Keep payload in memory for faster access
            )

            # Create payload indexes for efficient filtering
            # Index on document_id for retrieving all chunks from a document
            self.client.create_payload_index(
                collection_name=self.chunk_collection,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            # Index on level for filtering by hierarchy
            self.client.create_payload_index(
                collection_name=self.chunk_collection,
                field_name="level",
                field_schema=PayloadSchemaType.INTEGER,
            )

            # Index on entity_ids for finding chunks mentioning specific entities
            self.client.create_payload_index(
                collection_name=self.chunk_collection,
                field_name="entity_ids",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            logger.info(f"Created collection: {self.chunk_collection}")

    def _create_entity_collection(self, hnsw_m: int, hnsw_ef_construct: int) -> None:
        """Create the entities collection.

        Collection stores embeddings for entity descriptions to enable semantic
        entity search and similarity matching.

        Payload schema:
        - entity_id: UUID of the entity
        - canonical_name: Normalized entity name
        - entity_type: Type of entity (SYSTEM, SUBSYSTEM, COMPONENT, etc.)
        - description: Full text description of the entity
        - aliases: List of alternative names
        - related_entity_ids: List of related entity UUIDs

        Args:
            hnsw_m: Number of edges per node in HNSW graph
            hnsw_ef_construct: Size of dynamic candidate list for construction
        """
        if not self.collection_exists(self.entity_collection):
            self.client.create_collection(
                collection_name=self.entity_collection,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimension,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=hnsw_m,
                        ef_construct=hnsw_ef_construct,
                        full_scan_threshold=10000,
                    ),
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000,  # Entities collection typically smaller
                    memmap_threshold=25000,
                ),
                on_disk_payload=False,
            )

            # Create payload indexes
            # Index on entity_type for filtering by type
            self.client.create_payload_index(
                collection_name=self.entity_collection,
                field_name="entity_type",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            # Index on entity_id for direct lookup
            self.client.create_payload_index(
                collection_name=self.entity_collection,
                field_name="entity_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

            logger.info(f"Created collection: {self.entity_collection}")

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections().collections
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Upsert document chunks with their embeddings.

        Args:
            chunks: List of chunk dictionaries with payload data
            vectors: List of embedding vectors (same length as chunks)
            batch_size: Number of vectors to upsert per batch

        Returns:
            Number of chunks successfully upserted

        Raises:
            ValueError: If chunks and vectors lengths don't match
            Exception: If upsert operation fails
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match vectors count ({len(vectors)})"
            )

        if not chunks:
            logger.warning("No chunks to upsert")
            return 0

        try:
            total_upserted = 0

            # Process in batches for efficiency
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_vectors = vectors[i : i + batch_size]

                # Create PointStruct objects
                points = [
                    PointStruct(
                        id=str(chunk["chunk_id"]),  # Use chunk_id as point ID
                        vector=vector,
                        payload=self._prepare_chunk_payload(chunk),
                    )
                    for chunk, vector in zip(batch_chunks, batch_vectors)
                ]

                # Upsert batch
                self.client.upsert(
                    collection_name=self.chunk_collection,
                    points=points,
                    wait=True,  # Wait for operation to complete
                )

                total_upserted += len(points)

            logger.info(f"Upserted {total_upserted} chunks to {self.chunk_collection}")
            return total_upserted

        except Exception as e:
            logger.error(f"Failed to upsert chunks: {e}")
            raise

    def _prepare_chunk_payload(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare chunk payload for storage.

        Ensures all required fields are present and properly formatted.

        Args:
            chunk: Chunk dictionary

        Returns:
            Prepared payload dictionary
        """
        payload = {
            "chunk_id": str(chunk["chunk_id"]),
            "document_id": str(chunk["document_id"]),
            "level": chunk.get("level", 4),
            "content": chunk.get("content", ""),
            "metadata": chunk.get("metadata", {}),
            "timestamp": chunk.get("timestamp", ""),
        }

        # Convert entity_ids to list of strings
        if "entity_ids" in chunk.get("metadata", {}):
            payload["entity_ids"] = [str(eid) for eid in chunk["metadata"]["entity_ids"]]
        else:
            payload["entity_ids"] = []

        return payload

    def upsert_entities(
        self,
        entities: List[Dict[str, Any]],
        vectors: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Upsert entities with their embeddings.

        Args:
            entities: List of entity dictionaries with payload data
            vectors: List of embedding vectors (same length as entities)
            batch_size: Number of vectors to upsert per batch

        Returns:
            Number of entities successfully upserted

        Raises:
            ValueError: If entities and vectors lengths don't match
            Exception: If upsert operation fails
        """
        if len(entities) != len(vectors):
            raise ValueError(
                f"Entities count ({len(entities)}) must match vectors count ({len(vectors)})"
            )

        if not entities:
            logger.warning("No entities to upsert")
            return 0

        try:
            total_upserted = 0

            # Process in batches
            for i in range(0, len(entities), batch_size):
                batch_entities = entities[i : i + batch_size]
                batch_vectors = vectors[i : i + batch_size]

                # Create PointStruct objects
                points = [
                    PointStruct(
                        id=str(entity["entity_id"]),  # Use entity_id as point ID
                        vector=vector,
                        payload=self._prepare_entity_payload(entity),
                    )
                    for entity, vector in zip(batch_entities, batch_vectors)
                ]

                # Upsert batch
                self.client.upsert(
                    collection_name=self.entity_collection,
                    points=points,
                    wait=True,
                )

                total_upserted += len(points)

            logger.info(f"Upserted {total_upserted} entities to {self.entity_collection}")
            return total_upserted

        except Exception as e:
            logger.error(f"Failed to upsert entities: {e}")
            raise

    def _prepare_entity_payload(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare entity payload for storage.

        Args:
            entity: Entity dictionary

        Returns:
            Prepared payload dictionary
        """
        payload = {
            "entity_id": str(entity["entity_id"]),
            "canonical_name": entity.get("canonical_name", ""),
            "entity_type": entity.get("entity_type", ""),
            "description": entity.get("description", ""),
            "aliases": entity.get("aliases", []),
        }

        # Convert related_entity_ids to list of strings
        if "related_entity_ids" in entity:
            payload["related_entity_ids"] = [str(eid) for eid in entity["related_entity_ids"]]
        else:
            payload["related_entity_ids"] = []

        return payload

    def search_chunks(
        self,
        query_vector: List[float],
        top_k: int = 20,
        score_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar document chunks."""
        try:
            search_filter = self._build_filter(filters) if filters else None

            # qdrant-client local mode does not expose `.search`; use `query_points`.
            resp = self.client.query_points(
                collection_name=self.chunk_collection,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,
            )

            formatted_results: List[Dict[str, Any]] = []
            for point in getattr(resp, "points", []) or []:
                formatted_results.append(
                    {
                        "chunk_id": point.id,
                        "score": point.score,
                        "payload": point.payload,
                    }
                )

            logger.info(f"Found {len(formatted_results)} chunks with score >= {score_threshold}")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search chunks: {e}")
            raise

    def search_entities(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: float = 0.5,
        entity_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar entities."""
        try:
            search_filter = None
            if entity_types:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="entity_type",
                            match=MatchAny(any=entity_types),
                        )
                    ]
                )

            resp = self.client.query_points(
                collection_name=self.entity_collection,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,
            )

            formatted_results: List[Dict[str, Any]] = []
            for point in getattr(resp, "points", []) or []:
                formatted_results.append(
                    {
                        "entity_id": point.id,
                        "score": point.score,
                        "payload": point.payload,
                    }
                )

            logger.info(f"Found {len(formatted_results)} entities with score >= {score_threshold}")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            raise

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from filter dictionary.

        Args:
            filters: Dictionary with filter conditions

        Returns:
            Qdrant Filter object
        """
        conditions = []

        # Filter by document_id
        if "document_id" in filters:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(filters["document_id"])),
                )
            )

        # Filter by level
        if "level" in filters:
            conditions.append(
                FieldCondition(
                    key="level",
                    match=MatchValue(value=filters["level"]),
                )
            )

        # Filter by entity_ids (chunks mentioning specific entities)
        if "entity_ids" in filters:
            entity_ids = filters["entity_ids"]
            if not isinstance(entity_ids, list):
                entity_ids = [entity_ids]
            conditions.append(
                FieldCondition(
                    key="entity_ids",
                    match=MatchAny(any=[str(eid) for eid in entity_ids]),
                )
            )

        return Filter(must=conditions) if conditions else None

    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            True if deletion successful

        Raises:
            Exception: If deletion fails
        """
        try:
            self.client.delete(
                collection_name=self.chunk_collection,
                points_selector=models.PointIdsList(points=[str(cid) for cid in chunk_ids]),
            )
            logger.info(f"Deleted {len(chunk_ids)} chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            raise

    def delete_chunks_by_document(self, document_id: str) -> bool:
        """Delete all chunks belonging to a document.

        Args:
            document_id: UUID of the document

        Returns:
            True if deletion successful

        Raises:
            Exception: If deletion fails
        """
        try:
            self.client.delete(
                collection_name=self.chunk_collection,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=str(document_id)),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"Deleted all chunks for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete chunks for document: {e}")
            raise

    def delete_entities(self, entity_ids: List[str]) -> bool:
        """Delete entities by their IDs.

        Args:
            entity_ids: List of entity IDs to delete

        Returns:
            True if deletion successful

        Raises:
            Exception: If deletion fails
        """
        try:
            self.client.delete(
                collection_name=self.entity_collection,
                points_selector=models.PointIdsList(points=[str(eid) for eid in entity_ids]),
            )
            logger.info(f"Deleted {len(entity_ids)} entities")
            return True

        except Exception as e:
            logger.error(f"Failed to delete entities: {e}")
            raise

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk by its ID.

        Args:
            chunk_id: UUID of the chunk

        Returns:
            Chunk data with payload or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name=self.chunk_collection,
                ids=[str(chunk_id)],
                with_payload=True,
                with_vectors=False,
            )

            if result:
                return {
                    "chunk_id": result[0].id,
                    "payload": result[0].payload,
                }
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {e}")
            return None

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an entity by its ID.

        Args:
            entity_id: UUID of the entity

        Returns:
            Entity data with payload or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name=self.entity_collection,
                ids=[str(entity_id)],
                with_payload=True,
                with_vectors=False,
            )

            if result:
                return {
                    "entity_id": result[0].id,
                    "payload": result[0].payload,
                }
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve entity {entity_id}: {e}")
            return None

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection.

        Note:
            Local/in-memory Qdrant returns a slightly different CollectionInfo shape
            (e.g., no `vectors_count`). This method normalizes fields for callers/tests.
        """
        try:
            info = self.client.get_collection(collection_name)

            points_count = int(getattr(info, "points_count", 0) or 0)
            vectors_count = int(getattr(info, "vectors_count", points_count) or points_count)
            segments_count = int(getattr(info, "segments_count", 0) or 0)
            status = getattr(info, "status", None)

            vectors_cfg = None
            try:
                vectors_cfg = info.config.params.vectors  # type: ignore[attr-defined]
            except Exception:
                vectors_cfg = None

            distance = getattr(vectors_cfg, "distance", None)
            if hasattr(distance, "value"):
                distance = distance.value
            elif distance is not None:
                distance = str(distance)

            size = getattr(vectors_cfg, "size", None)

            hnsw_cfg = getattr(getattr(info, "config", None), "hnsw_config", None)
            hnsw_m = getattr(hnsw_cfg, "m", None)
            hnsw_ef = getattr(hnsw_cfg, "ef_construct", None)

            return {
                "name": collection_name,
                "vectors_count": vectors_count,
                "points_count": points_count,
                "segments_count": segments_count,
                "status": status,
                "config": {
                    "distance": distance,
                    "size": size,
                    "hnsw_config": {"m": hnsw_m, "ef_construct": hnsw_ef},
                },
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    def health_check(self) -> Tuple[bool, str]:
        """Check if Qdrant is healthy and accessible.

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            # Check if we can list collections
            collections = self.client.get_collections()

            # Check if our collections exist
            collection_names = {col.name for col in collections.collections}
            chunks_exists = self.chunk_collection in collection_names
            entities_exists = self.entity_collection in collection_names

            if chunks_exists and entities_exists:
                message = "Qdrant is healthy. All collections exist."
                logger.info(message)
                return True, message
            else:
                missing = []
                if not chunks_exists:
                    missing.append(self.chunk_collection)
                if not entities_exists:
                    missing.append(self.entity_collection)

                message = f"Qdrant is accessible but missing collections: {', '.join(missing)}"
                logger.warning(message)
                return False, message

        except Exception as e:
            message = f"Qdrant health check failed: {e}"
            logger.error(message)
            return False, message

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about both collections.

        Returns:
            Dictionary with statistics for chunks and entities collections
        """
        stats = {}

        try:
            if self.collection_exists(self.chunk_collection):
                chunk_info = self.get_collection_info(self.chunk_collection)
                stats["chunks"] = {
                    "count": chunk_info["points_count"],
                    "vectors": chunk_info["vectors_count"],
                    "segments": chunk_info["segments_count"],
                    "status": chunk_info["status"],
                }

            if self.collection_exists(self.entity_collection):
                entity_info = self.get_collection_info(self.entity_collection)
                stats["entities"] = {
                    "count": entity_info["points_count"],
                    "vectors": entity_info["vectors_count"],
                    "segments": entity_info["segments_count"],
                    "status": entity_info["status"],
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def batch_search_chunks(
        self,
        query_vectors: List[List[float]],
        top_k: int = 20,
        score_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Perform batch search for multiple query vectors."""
        try:
            search_filter = self._build_filter(filters) if filters else None

            all_results: List[List[Dict[str, Any]]] = []
            for query_vector in query_vectors:
                resp = self.client.query_points(
                    collection_name=self.chunk_collection,
                    query=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    query_filter=search_filter,
                    with_payload=True,
                    with_vectors=False,
                )

                formatted_results = [
                    {"chunk_id": p.id, "score": p.score, "payload": p.payload}
                    for p in getattr(resp, "points", []) or []
                ]
                all_results.append(formatted_results)

            logger.info(f"Completed batch search for {len(query_vectors)} queries")
            return all_results

        except Exception as e:
            logger.error(f"Failed to perform batch search: {e}")
            raise

    def close(self) -> None:
        """Close the Qdrant client connection.

        Should be called when the manager is no longer needed.
        """
        try:
            if hasattr(self, "client"):
                self.client.close()
                logger.info("Closed Qdrant client connection")
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")

    def __enter__(self) -> "QdrantManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

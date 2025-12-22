"""Example usage of QdrantManager for the Graph RAG system.

This script demonstrates:
1. Initializing QdrantManager
2. Creating collections
3. Upserting document chunks and entities
4. Performing searches
5. Using filters and batch operations
"""

import random
import uuid
from datetime import UTC, datetime

from src.storage.qdrant_manager import QdrantManager
from src.utils.config import DatabaseConfig


def generate_random_vector(dimension: int = 768) -> list[float]:
    """Generate a random normalized embedding vector.

    Args:
        dimension: Vector dimension (default: 768 for BGE embeddings)

    Returns:
        Normalized vector
    """
    random.seed()
    vector = [random.random() for _ in range(dimension)]
    # Normalize
    magnitude = sum(x**2 for x in vector) ** 0.5
    vector = [x / magnitude for x in vector]
    return vector


def main() -> None:
    """Demonstrate QdrantManager usage."""

    # 1. Initialize QdrantManager with configuration
    print("1. Initializing QdrantManager...")
    config = DatabaseConfig(
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_api_key="",  # No API key for local instance
        qdrant_https=False,
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=768,
    )

    manager = QdrantManager(config=config)

    # 2. Create collections
    print("\n2. Creating collections...")
    manager.create_collections(recreate=True)
    print("   Collections created successfully!")

    # Check health
    is_healthy, message = manager.health_check()
    print(f"   Health check: {message}")

    # 3. Prepare sample document chunks
    print("\n3. Preparing sample document chunks...")
    doc_id = str(uuid.uuid4())

    chunks = []
    vectors = []

    for i in range(5):
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "document_id": doc_id,
            "level": 3,  # subsection level
            "content": f"This chunk discusses power system procedures. "
            f"Section {i} covers initialization and monitoring steps.",
            "metadata": {
                "document_title": "Power System Manual",
                "section_title": f"Power Management Section {i}",
                "page_numbers": [i * 2 + 1, i * 2 + 2],
                "hierarchy_path": f"1.2.{i}",
                "entity_ids": [str(uuid.uuid4()), str(uuid.uuid4())],
                "has_tables": i % 2 == 0,
                "has_figures": i % 3 == 0,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }
        chunks.append(chunk)
        vectors.append(generate_random_vector())

    # 4. Upsert chunks
    print("\n4. Upserting chunks to Qdrant...")
    count = manager.upsert_chunks(chunks, vectors)
    print(f"   Upserted {count} chunks")

    # 5. Prepare sample entities
    print("\n5. Preparing sample entities...")
    entities = [
        {
            "entity_id": str(uuid.uuid4()),
            "canonical_name": "power_subsystem",
            "entity_type": "SYSTEM",
            "description": "The electrical power subsystem manages power generation, "
            "storage, and distribution throughout the system",
            "aliases": ["EPS", "Electrical Power System", "Power System"],
            "related_entity_ids": [],
        },
        {
            "entity_id": str(uuid.uuid4()),
            "canonical_name": "solar_array",
            "entity_type": "SUBSYSTEM",
            "description": "Solar array panels that convert sunlight into electrical power",
            "aliases": ["Solar Panel", "SA", "Photovoltaic Array"],
            "related_entity_ids": [],
        },
        {
            "entity_id": str(uuid.uuid4()),
            "canonical_name": "battery_pack",
            "entity_type": "COMPONENT",
            "description": "Lithium-ion battery pack for energy storage during eclipse",
            "aliases": ["Battery", "Energy Storage Unit"],
            "related_entity_ids": [],
        },
    ]

    entity_vectors = [generate_random_vector() for _ in entities]

    # 6. Upsert entities
    print("\n6. Upserting entities to Qdrant...")
    count = manager.upsert_entities(entities, entity_vectors)
    print(f"   Upserted {count} entities")

    # 7. Search for similar chunks
    print("\n7. Searching for similar chunks...")
    query_vector = generate_random_vector()
    results = manager.search_chunks(query_vector=query_vector, top_k=3, score_threshold=0.0)

    print(f"   Found {len(results)} similar chunks:")
    for i, result in enumerate(results):
        print(f"   [{i+1}] Chunk ID: {result['chunk_id'][:8]}...")
        print(f"       Score: {result['score']:.4f}")
        print(f"       Content: {result['payload']['content'][:80]}...")

    # 8. Search with filters
    print("\n8. Searching chunks with filters...")
    results = manager.search_chunks(
        query_vector=query_vector,
        top_k=5,
        score_threshold=0.0,
        filters={"document_id": doc_id, "level": 3},
    )

    print(f"   Found {len(results)} chunks from document {doc_id[:8]}...")

    # 9. Search entities
    print("\n9. Searching for similar entities...")
    entity_query = generate_random_vector()
    results = manager.search_entities(
        query_vector=entity_query, top_k=2, entity_types=["SYSTEM", "SUBSYSTEM"]
    )

    print(f"   Found {len(results)} similar entities:")
    for i, result in enumerate(results):
        payload = result["payload"]
        print(f"   [{i+1}] {payload['canonical_name']} ({payload['entity_type']})")
        print(f"       Score: {result['score']:.4f}")
        print(f"       Description: {payload['description'][:80]}...")

    # 10. Batch search
    print("\n10. Performing batch search...")
    query_vectors = [generate_random_vector() for _ in range(3)]
    batch_results = manager.batch_search_chunks(
        query_vectors=query_vectors, top_k=2, score_threshold=0.0
    )

    print(f"   Batch search completed for {len(query_vectors)} queries")
    for i, results in enumerate(batch_results):
        print(f"   Query {i+1}: Found {len(results)} results")

    # 11. Get collection statistics
    print("\n11. Collection statistics:")
    stats = manager.get_stats()
    print(f"   Chunks: {stats['chunks']['count']} points")
    print(f"   Entities: {stats['entities']['count']} points")

    # 12. Retrieve specific items
    print("\n12. Retrieving specific items by ID...")
    chunk_id = chunks[0]["chunk_id"]
    chunk = manager.get_chunk_by_id(chunk_id)
    if chunk:
        print(f"   Retrieved chunk: {chunk['payload']['content'][:60]}...")

    entity_id = entities[0]["entity_id"]
    entity = manager.get_entity_by_id(entity_id)
    if entity:
        print(f"   Retrieved entity: {entity['payload']['canonical_name']}")

    # 13. Cleanup example (optional)
    print("\n13. Cleanup operations...")

    # Delete a single chunk
    deleted = manager.delete_chunks([chunks[0]["chunk_id"]])
    print(f"   Deleted chunks: {deleted}")

    # Delete all chunks from a document
    deleted = manager.delete_chunks_by_document(doc_id)
    print(f"   Deleted all chunks from document: {deleted}")

    # Delete an entity
    deleted = manager.delete_entities([entities[0]["entity_id"]])
    print(f"   Deleted entities: {deleted}")

    # 14. Close connection
    print("\n14. Closing connection...")
    manager.close()
    print("   Connection closed")

    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()

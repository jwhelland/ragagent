# Qdrant Manager Guide

## Overview

The `QdrantManager` class provides a comprehensive interface for managing vector databases in the Graph RAG system. It handles two main collections:

1. **document_chunks**: Stores embeddings for hierarchical document chunks
2. **entities**: Stores embeddings for entity descriptions

## Features

- ✓ Collection creation with HNSW indexing
- ✓ Proper vector dimensions (768 for BGE embeddings)
- ✓ Payload schema with indexes on: `document_id`, `entity_ids`, `entity_type`
- ✓ Cosine similarity as distance metric
- ✓ HNSW index configuration for optimal performance
- ✓ Vector upsert and search operations
- ✓ Batch operations support
- ✓ Health check and collection info methods
- ✓ Connection pooling
- ✓ Error handling and logging

## Architecture

### Collection Schema

#### document_chunks Collection

**Purpose**: Store hierarchical document chunk embeddings with metadata for semantic search.

**Vector Configuration**:
- Dimension: 768 (BGE small model)
- Distance metric: Cosine similarity
- HNSW parameters:
  - `m=16`: Number of edges per node (balance between recall and memory)
  - `ef_construct=100`: Construction quality (higher = better quality, slower indexing)

**Payload Structure**:
```python
{
    "chunk_id": "uuid",              # Unique chunk identifier
    "document_id": "uuid",           # Source document reference
    "level": 3,                      # Hierarchy level (1-4)
    "content": "text content...",    # Chunk text
    "metadata": {
        "document_title": "...",
        "section_title": "...",
        "page_numbers": [5, 6],
        "hierarchy_path": "1.2.3",
        "entity_ids": ["uuid1", "uuid2"],  # Entities in chunk
        "has_tables": false,
        "has_figures": true
    },
    "entity_ids": ["uuid1", "uuid2"], # Top-level for indexing
    "timestamp": "2024-12-14T..."
}
```

**Indexes**:
- `document_id` (keyword): Fast filtering by document
- `level` (integer): Filter by hierarchy level
- `entity_ids` (keyword): Find chunks mentioning specific entities

#### entities Collection

**Purpose**: Store entity description embeddings for semantic entity search.

**Vector Configuration**:
- Dimension: 768
- Distance metric: Cosine similarity
- HNSW parameters: Same as chunks

**Payload Structure**:
```python
{
    "entity_id": "uuid",
    "canonical_name": "power_subsystem",
    "entity_type": "SYSTEM",
    "description": "Full description...",
    "aliases": ["EPS", "Electrical Power System"],
    "related_entity_ids": ["uuid1", "uuid2"]
}
```

**Indexes**:
- `entity_type` (keyword): Filter by entity type
- `entity_id` (keyword): Direct entity lookup

## Usage

### Initialization

```python
from src.storage.qdrant_manager import QdrantManager
from src.utils.config import DatabaseConfig

# Create configuration
config = DatabaseConfig(
    qdrant_host="localhost",
    qdrant_port=6333,
    qdrant_api_key="",  # Optional for cloud instance
    qdrant_https=False,
    embedding_model="BAAI/bge-small-en-v1.5",
    embedding_dimension=768,
)

# Initialize manager
manager = QdrantManager(config=config)
```

### Collection Management

```python
# Create collections (first time setup)
manager.create_collections(
    recreate=False,      # Set to True to delete and recreate
    hnsw_m=16,          # HNSW edges per node
    hnsw_ef_construct=100  # Construction quality
)

# Check if collections exist
if manager.collection_exists("document_chunks"):
    print("Chunks collection exists")

# Get collection information
info = manager.get_collection_info("document_chunks")
print(f"Collection has {info['points_count']} points")

# Health check
is_healthy, message = manager.health_check()
print(message)
```

### Upserting Chunks

```python
# Prepare chunks
chunks = [
    {
        "chunk_id": "uuid-1",
        "document_id": "doc-uuid",
        "level": 3,
        "content": "Chunk text content...",
        "metadata": {
            "document_title": "Manual",
            "section_title": "Section 1",
            "page_numbers": [1, 2],
            "hierarchy_path": "1.1.1",
            "entity_ids": ["entity-uuid-1", "entity-uuid-2"],
            "has_tables": False,
            "has_figures": True,
        },
        "timestamp": "2024-12-14T10:00:00Z",
    },
    # More chunks...
]

# Generate embeddings (using your embedding model)
vectors = [generate_embedding(chunk["content"]) for chunk in chunks]

# Upsert to Qdrant
count = manager.upsert_chunks(chunks, vectors, batch_size=100)
print(f"Upserted {count} chunks")
```

### Searching Chunks

```python
# Basic search
query_vector = generate_embedding("satellite power systems")
results = manager.search_chunks(
    query_vector=query_vector,
    top_k=20,
    score_threshold=0.5  # Minimum similarity score
)

for result in results:
    print(f"Chunk: {result['chunk_id']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['payload']['content'][:100]}...")
```

### Searching with Filters

```python
# Filter by document
results = manager.search_chunks(
    query_vector=query_vector,
    top_k=10,
    filters={"document_id": "specific-doc-uuid"}
)

# Filter by hierarchy level (get section-level chunks)
results = manager.search_chunks(
    query_vector=query_vector,
    top_k=10,
    filters={"level": 2}
)

# Filter by entity mentions (chunks mentioning specific entities)
results = manager.search_chunks(
    query_vector=query_vector,
    top_k=10,
    filters={"entity_ids": ["entity-uuid-1", "entity-uuid-2"]}
)

# Combine filters
results = manager.search_chunks(
    query_vector=query_vector,
    top_k=10,
    filters={
        "document_id": "doc-uuid",
        "level": 3,
        "entity_ids": ["entity-uuid-1"]
    }
)
```

### Working with Entities

```python
# Prepare entities
entities = [
    {
        "entity_id": "entity-uuid-1",
        "canonical_name": "power_subsystem",
        "entity_type": "SYSTEM",
        "description": "Manages satellite electrical power...",
        "aliases": ["EPS", "Electrical Power System"],
        "related_entity_ids": ["related-uuid-1"],
    },
    # More entities...
]

# Generate embeddings for descriptions
entity_vectors = [generate_embedding(e["description"]) for e in entities]

# Upsert entities
count = manager.upsert_entities(entities, entity_vectors)

# Search entities
query_vector = generate_embedding("power management")
results = manager.search_entities(
    query_vector=query_vector,
    top_k=5,
    score_threshold=0.6,
    entity_types=["SYSTEM", "SUBSYSTEM"]  # Optional type filter
)

for result in results:
    entity = result["payload"]
    print(f"{entity['canonical_name']} ({entity['entity_type']})")
    print(f"Score: {result['score']:.4f}")
```

### Batch Operations

```python
# Batch search for multiple queries
query_vectors = [
    generate_embedding(query1),
    generate_embedding(query2),
    generate_embedding(query3),
]

batch_results = manager.batch_search_chunks(
    query_vectors=query_vectors,
    top_k=10,
    score_threshold=0.5
)

# batch_results is a list of result lists
for i, results in enumerate(batch_results):
    print(f"Query {i+1}: Found {len(results)} results")
```

### Retrieval Operations

```python
# Get specific chunk by ID
chunk = manager.get_chunk_by_id("chunk-uuid")
if chunk:
    print(chunk["payload"]["content"])

# Get specific entity by ID
entity = manager.get_entity_by_id("entity-uuid")
if entity:
    print(entity["payload"]["canonical_name"])
```

### Delete Operations

```python
# Delete specific chunks
deleted = manager.delete_chunks(["chunk-uuid-1", "chunk-uuid-2"])

# Delete all chunks from a document
deleted = manager.delete_chunks_by_document("doc-uuid")

# Delete specific entities
deleted = manager.delete_entities(["entity-uuid-1", "entity-uuid-2"])
```

### Statistics and Monitoring

```python
# Get collection statistics
stats = manager.get_stats()
print(f"Chunks: {stats['chunks']['count']}")
print(f"Entities: {stats['entities']['count']}")

# Get detailed collection info
info = manager.get_collection_info("document_chunks")
print(f"Vectors: {info['vectors_count']}")
print(f"Segments: {info['segments_count']}")
print(f"Status: {info['status']}")

# Health check
is_healthy, message = manager.health_check()
if not is_healthy:
    print(f"Warning: {message}")
```

### Context Manager Usage

```python
# Use as context manager for automatic cleanup
with QdrantManager(config=config) as manager:
    manager.create_collections()
    # ... perform operations ...
# Connection automatically closed
```

## Configuration

### HNSW Parameters

The HNSW (Hierarchical Navigable Small World) algorithm parameters affect performance:

**`m` (edges per node)**:
- Default: 16
- Range: 4-64
- Higher values:
  - Better recall (more accurate results)
  - More memory usage
  - Slower insertion

**`ef_construct` (construction quality)**:
- Default: 100
- Range: 4-512
- Higher values:
  - Better quality index
  - Slower indexing
  - No impact on search speed

**Recommendations**:
- **Small collections (<10K vectors)**: `m=16`, `ef_construct=100`
- **Medium collections (10K-100K)**: `m=16`, `ef_construct=200`
- **Large collections (>100K)**: `m=24`, `ef_construct=300`

### Optimizer Configuration

**`indexing_threshold`**: Number of vectors before starting indexing
- Chunks: 20,000
- Entities: 10,000

**`memmap_threshold`**: Number of vectors before using memory mapping
- Chunks: 50,000
- Entities: 25,000

## Performance Optimization

### Batch Sizes

- **Upsert operations**: Use `batch_size=100` for optimal throughput
- **Search operations**: Use batch search for multiple queries
- **Large datasets**: Increase batch size to 500-1000

### Indexing Strategy

```python
# For large initial loads, use higher thresholds
manager.create_collections(
    hnsw_m=24,              # Better recall for large collections
    hnsw_ef_construct=200   # Higher quality
)
```

### Search Performance

- Use `score_threshold` to reduce result processing
- Apply filters to reduce search space
- Use appropriate `top_k` values (don't over-fetch)

## Error Handling

```python
from qdrant_client.http.exceptions import UnexpectedResponse

try:
    manager.create_collections()
except ConnectionError as e:
    print(f"Cannot connect to Qdrant: {e}")
except UnexpectedResponse as e:
    print(f"Qdrant API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Integration with Ingestion Pipeline

```python
# In ingestion pipeline
from src.storage.qdrant_manager import QdrantManager
from src.utils.embeddings import EmbeddingGenerator

# Initialize
manager = QdrantManager(config=db_config)
embedder = EmbeddingGenerator(config=embedding_config)

# Process documents
for document in documents:
    # Parse and chunk document
    chunks = chunker.chunk_document(document)

    # Generate embeddings
    texts = [chunk["content"] for chunk in chunks]
    vectors = embedder.generate_embeddings(texts)

    # Store in Qdrant
    manager.upsert_chunks(chunks, vectors)
```

## Integration with Retrieval System

```python
# In retrieval system
from src.storage.qdrant_manager import QdrantManager
from src.storage.neo4j_manager import Neo4jManager

# Vector-first retrieval
query_vector = embedder.generate_embedding(user_query)
chunks = manager.search_chunks(
    query_vector=query_vector,
    top_k=20,
    score_threshold=0.5
)

# Extract entity IDs from chunks
entity_ids = set()
for chunk in chunks:
    entity_ids.update(chunk["payload"]["entity_ids"])

# Expand with graph traversal
graph_manager = Neo4jManager(config=db_config)
related_entities = graph_manager.get_related_entities(entity_ids)
```

## Best Practices

1. **Always use batch operations** for multiple items
2. **Set appropriate score thresholds** to filter low-quality results
3. **Use filters** to reduce search space when possible
4. **Monitor collection sizes** and adjust HNSW parameters
5. **Use connection pooling** (enabled by default)
6. **Close connections** when done (or use context manager)
7. **Handle errors gracefully** with try-except blocks
8. **Log operations** for debugging and monitoring
9. **Test with sample data** before full ingestion
10. **Backup collections** regularly using Qdrant snapshots

## Troubleshooting

### Connection Issues

```python
# Check if Qdrant is running
is_healthy, message = manager.health_check()
if not is_healthy:
    print(f"Problem: {message}")
```

### Performance Issues

- Check collection size: Large collections may need parameter tuning
- Verify HNSW parameters are appropriate for collection size
- Use filters to reduce search space
- Consider using memory mapping for large collections

### Memory Issues

- Enable on-disk storage for large collections
- Increase `memmap_threshold` to use disk storage sooner
- Use batch processing to limit memory usage

### Search Quality Issues

- Increase `hnsw_m` for better recall
- Lower `score_threshold` to get more results
- Verify embeddings are generated correctly
- Check if query and document embeddings use same model

## Advanced Usage

### Custom Distance Metrics

While Cosine similarity is recommended for text embeddings, you can use other metrics:

```python
from qdrant_client.models import Distance

# Euclidean distance
Distance.EUCLID

# Dot product (for normalized vectors)
Distance.DOT
```

### Payload Filtering Operators

```python
# Range filtering (numeric fields)
filters = {
    "level": {"gte": 2, "lte": 4}  # Level between 2 and 4
}

# Multiple values (OR operation)
filters = {
    "entity_type": ["SYSTEM", "SUBSYSTEM"]
}
```

### Vector Search with Score Normalization

```python
# Search returns raw cosine similarity scores (0-1)
# Higher score = more similar

results = manager.search_chunks(
    query_vector=query_vector,
    top_k=10,
    score_threshold=0.7  # Only results with >0.7 similarity
)

# Normalize scores for ranking
max_score = max(r["score"] for r in results) if results else 1.0
for result in results:
    result["normalized_score"] = result["score"] / max_score
```

## See Also

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Graph RAG Architecture](../plans/graph-rag-architecture.md)
- [Neo4j Manager Guide](neo4j_manager_guide.md)

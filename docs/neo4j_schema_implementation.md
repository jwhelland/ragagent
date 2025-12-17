# Neo4j Schema and Manager Implementation

## Overview

This document describes the implementation of Task 1.3 from the Graph RAG architecture: Neo4j schema and manager for entity and relationship storage.

## Files Created

### 1. src/storage/__init__.py
Package initialization file that exports all schema models and the Neo4jManager class.

### 2. src/storage/schemas.py
Pydantic models for all entity types, relationships, and chunks.

#### Entity Types (12 total)
1. **SYSTEM** - Top-level satellite systems
2. **SUBSYSTEM** - Components within systems
3. **COMPONENT** - Individual parts
4. **PARAMETER** - Measurable values
5. **PROCEDURE** - Operational procedures
6. **PROCEDURE_STEP** - Individual steps within procedures
7. **CONCEPT** - Technical concepts
8. **DOCUMENT** - Source documents
9. **STANDARD** - Referenced standards
10. **ANOMALY** - Known issues or failure modes
11. **TABLE** - Tables with structured data
12. **FIGURE** - Diagrams, charts, schematics

#### Base Entity Model
All entities inherit from the `Entity` base class which includes:
- `id`: Unique identifier (UUID)
- `canonical_name`: Normalized entity name (auto-lowercased, spaces to underscores)
- `entity_type`: EntityType enum
- `aliases`: List of alternative names
- `description`: Entity description
- `abbreviations`: Common abbreviations
- `confidence_score`: Extraction confidence (0.0-1.0)
- `extraction_method`: SPACY, LLM, MANUAL, or MERGED
- `status`: DRAFT, UNDER_REVIEW, APPROVED, or REJECTED
- `first_seen`: First extraction timestamp
- `last_updated`: Last update timestamp
- `mention_count`: Number of mentions across documents
- `source_documents`: List of document IDs
- `properties`: Additional flexible properties

Each entity type extends the base with specific fields:
- **Component**: part_number, manufacturer
- **Parameter**: unit, min_value, max_value, nominal_value
- **Procedure**: procedure_type, steps, duration_minutes, prerequisites
- **ProcedureStep**: parent_procedure, step_number, step_text, warnings, checks
- **Document**: filename, title, version, date, author, page_count, checksum
- **Standard**: standard_body, standard_number, version, publication_date
- **Anomaly**: severity, affected_components, root_cause, mitigation_procedures
- **Table**: table_number, caption, page_number, column_headers, structured_data
- **Figure**: figure_number, caption, page_number, figure_type, image_path

#### Relationship Model
The `Relationship` class includes:
- `id`: Unique identifier
- `type`: RelationshipType enum (30+ types)
- `source_entity_id`: Source entity ID
- `target_entity_id`: Target entity ID
- `description`: Relationship description
- `confidence_score`: Extraction confidence
- `extraction_method`: How relationship was extracted
- `bidirectional`: Whether relationship goes both ways
- `created_at`, `last_updated`: Timestamps
- `status`: Curation status
- `provenance`: List of RelationshipProvenance objects
- `confirmation_count`: Number of sources confirming relationship
- `properties`: Additional flexible properties

#### Relationship Provenance
Critical feature for tracking WHERE relationships were found:
- `document_id`: Document where relationship appears
- `section`: Section within document
- `page_number`: Page number
- `chunk_id`: Specific chunk ID
- `extracted_text`: Text supporting the relationship
- `confidence_score`: Confidence of this specific extraction

Multiple provenance records indicate the relationship was found in multiple places, increasing confidence.

#### Relationship Types (30+ types)
Organized into categories:

**Structural:**
- PART_OF, CONTAINS, DEPENDS_ON

**Functional:**
- CONTROLS, MONITORS, PROVIDES_POWER_TO, SENDS_DATA_TO

**Procedural:**
- REFERENCES, PRECEDES, REQUIRES_CHECK, AFFECTS

**Semantic:**
- IMPLEMENTS, SIMILAR_TO, CAUSED_BY, MITIGATED_BY

**Table/Figure:**
- REFERENCES_TABLE, REFERENCES_FIGURE, DEFINED_IN_TABLE, SHOWN_IN_FIGURE, CONTAINS_TABLE, CONTAINS_FIGURE

**Document:**
- CROSS_REFERENCES, MENTIONED_IN, PARENT_CHUNK, CHILD_CHUNK

### 3. src/storage/neo4j_manager.py
Comprehensive Neo4j database manager with all required functionality.

#### Neo4jManager Class

**Initialization:**
```python
from src.utils.config import DatabaseConfig
from src.storage import Neo4jManager

config = DatabaseConfig()  # Loads from environment
manager = Neo4jManager(config)
manager.connect()
```

**Connection Management:**
- `connect()`: Establish connection with connection pooling (max 50 connections)
- `close()`: Close connection
- `session()`: Context manager for sessions
- `health_check()`: Verify connection health

**Schema Operations:**
- `create_schema()`: Creates all constraints and indexes
  - Uniqueness constraints on entity IDs for all 12 entity types
  - Uniqueness constraint on Chunk IDs
  - Indexes on canonical_name for all entity types
  - Index on entity_type for cross-type queries
  - Index on status for filtering by curation status
  - Full-text search index on canonical_name, aliases, and description
  - Index on document_id for chunks
  - Index on relationship types
- `drop_schema()`: Remove all constraints and indexes (use with caution)

**Entity CRUD Operations:**
- `create_entity(entity: Entity) -> str`: Create entity node
- `get_entity(entity_id: str, entity_type: Optional[EntityType]) -> Optional[Dict]`: Get entity by ID
- `get_entity_by_canonical_name(canonical_name: str, entity_type: Optional[EntityType]) -> Optional[Dict]`: Get entity by name
- `update_entity(entity_id: str, properties: Dict) -> bool`: Update entity properties
- `delete_entity(entity_id: str) -> bool`: Delete entity and all relationships
- `search_entities(query: str, entity_types: Optional[List[EntityType]], limit: int, status: Optional[EntityStatus]) -> List[Dict]`: Full-text search
- `list_entities(entity_type: Optional[EntityType], status: Optional[EntityStatus], limit: int, offset: int) -> List[Dict]`: List with filters

**Relationship CRUD Operations:**
- `create_relationship(relationship: Relationship) -> str`: Create relationship
- `get_relationship(relationship_id: str) -> Optional[Dict]`: Get relationship by ID
- `get_relationships(entity_id: str, relationship_type: Optional[RelationshipType], direction: str) -> List[Dict]`: Get relationships for entity
  - `direction`: 'outgoing', 'incoming', or 'both'
- `update_relationship(relationship_id: str, properties: Dict) -> bool`: Update relationship
- `delete_relationship(relationship_id: str) -> bool`: Delete relationship

**Chunk Operations:**
- `create_chunk(chunk: Chunk) -> str`: Create chunk node
- `get_chunk(chunk_id: str) -> Optional[Dict]`: Get chunk by ID
- `get_chunks_by_document(document_id: str, level: Optional[int]) -> List[Dict]`: Get all chunks for a document

**Graph Traversal Operations:**
- `find_path(source_id: str, target_id: str, max_depth: int, relationship_types: Optional[List[RelationshipType]]) -> List[Dict]`: Find paths between entities
- `traverse_relationships(entity_id: str, relationship_types: List[RelationshipType], max_depth: int, direction: str) -> List[Dict]`: Traverse from an entity

**Utility Methods:**
- `get_statistics() -> Dict`: Get database statistics (entity counts by type, relationship counts by type, etc.)
- `clear_database()`: Delete all data (use with extreme caution!)
- `execute_cypher(query: str, parameters: Optional[Dict]) -> List[Dict]`: Execute custom Cypher queries

## Key Features Implemented

### 1. Connection Pooling
The Neo4j driver is configured with a connection pool (max 50 connections) for efficient concurrent access.

### 2. Full-Text Search
A full-text index is created on `canonical_name`, `aliases`, and `description` fields across all entity types, enabling fast text search.

### 3. Relationship Provenance
Every relationship can track multiple sources (documents, pages, chunks) where it was found, enabling:
- Source verification
- Confidence calculation based on multiple confirmations
- Conflict detection when sources disagree
- Efficient curation workflow

### 4. Flexible Properties
Both entities and relationships have a `properties` dict for additional custom fields without schema changes.

### 5. Comprehensive Indexing
Indexes created on:
- Entity IDs (uniqueness constraints)
- Canonical names (fast lookup)
- Entity types (cross-type queries)
- Status (filtering by curation state)
- Document IDs for chunks (document queries)
- Relationship types (relationship queries)

### 6. Error Handling
- Comprehensive logging using Python's logging module
- Descriptive error messages
- Graceful handling of missing entities/relationships
- Connection health checks

### 7. Type Safety
- Full type hints throughout
- Pydantic validation for all models
- Enum types for entity types, relationship types, status, etc.

## Usage Examples

### Creating Entities
```python
from src.storage import System, EntityType, Neo4jManager
from src.utils.config import DatabaseConfig

# Initialize manager
config = DatabaseConfig()
manager = Neo4jManager(config)
manager.connect()

# Create schema
manager.create_schema()

# Create entity
system = System(
    canonical_name="power_subsystem",
    entity_type=EntityType.SYSTEM,
    aliases=["Power System", "EPS"],
    description="Electrical Power Subsystem",
    confidence_score=0.95,
)

entity_id = manager.create_entity(system)
print(f"Created entity: {entity_id}")
```

### Creating Relationships with Provenance
```python
from src.storage import Relationship, RelationshipType, RelationshipProvenance

# Create relationship with provenance
provenance = RelationshipProvenance(
    document_id="doc-001",
    section="3.2",
    page_number=15,
    chunk_id="chunk-001",
    extracted_text="Battery provides power to the main bus",
    confidence_score=0.9,
)

relationship = Relationship(
    type=RelationshipType.PROVIDES_POWER_TO,
    source_entity_id=battery_id,
    target_entity_id=system_id,
    description="Battery powers the electrical power subsystem",
    confidence_score=0.9,
    provenance=[provenance],
)

rel_id = manager.create_relationship(relationship)
```

### Full-Text Search
```python
# Search for entities
results = manager.search_entities(
    query="power battery",
    entity_types=[EntityType.COMPONENT, EntityType.SYSTEM],
    limit=10,
    status=EntityStatus.APPROVED,
)

for result in results:
    print(f"{result['canonical_name']} (score: {result['search_score']})")
```

### Graph Traversal
```python
# Find all components that depend on a system
components = manager.traverse_relationships(
    entity_id=system_id,
    relationship_types=[RelationshipType.DEPENDS_ON],
    max_depth=2,
    direction="incoming",
)

for component in components:
    print(f"{component['canonical_name']} at depth {component['depth']}")
```

### Get Statistics
```python
stats = manager.get_statistics()
print(f"Total entities: {stats['total_entities']}")
print(f"Total relationships: {stats['total_relationships']}")
print(f"Systems: {stats['entities_by_type']['SYSTEM']}")
print(f"Components: {stats['entities_by_type']['COMPONENT']}")
```

## Testing

Run the test script:
```bash
python test_neo4j_schema.py
```

This validates:
- All Pydantic models can be instantiated
- Models correctly convert to Neo4j dictionaries
- Neo4jManager has all required methods
- No import errors

For integration testing with a real Neo4j instance:
```bash
# Start Neo4j (if using Docker)
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/ragagent2024 neo4j:latest

# Set environment variable
export NEO4J_PASSWORD=ragagent2024

# Run integration tests (to be created in tests/ directory)
pytest tests/test_neo4j_integration.py
```

## Next Steps

This implementation completes Task 1.3. The next tasks in the pipeline are:

1. **Task 1.4**: Qdrant vector database schema (for storing embeddings)
2. **Task 1.5**: PDF parsing with Docling
3. **Task 1.8**: Hierarchical chunking
4. **Task 1.10**: Complete ingestion pipeline

The Neo4j schema and manager are now ready to:
- Store entities extracted from documents
- Store relationships with provenance tracking
- Support the entity curation workflow (Phase 3)
- Enable graph-based retrieval (Phase 4)

## Configuration

Neo4j connection is configured via environment variables or .env file:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

These are loaded through `src/utils/config.py` using the `DatabaseConfig` Pydantic model.

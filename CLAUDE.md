# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Graph RAG (Retrieval-Augmented Generation) system for processing technical documents. It combines vector similarity search with knowledge graph traversal using Neo4j (graph) and Qdrant (vectors). The system extracts entities and relationships using both spaCy NER and LLM extraction, with entity deduplication and manual curation capabilities.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
uv sync

# Install spaCy language model (required for entity extraction)
uv run spacy download en_core_web_lg

# Start infrastructure services (Neo4j + Qdrant)
docker-compose up -d

# Verify services
docker-compose ps
# Neo4j: http://localhost:7474 (neo4j/ragagent2024)
# Qdrant: http://localhost:6333/dashboard
```

### Database Management
```bash
# Initialize Neo4j constraints/indexes and Qdrant collections
uv run ragagent-setup

# Recreate Qdrant collections (after embedding model changes)
uv run ragagent-setup --recreate-qdrant
```

### Document Ingestion
```bash
# Ingest PDFs from a directory
uv run ragagent-ingest --directory data/raw --batch-size 2

# Ingest specific files with verbose logging
uv run ragagent-ingest file1.pdf file2.pdf --verbose

# Force re-ingestion (bypass checksum-based skip)
uv run ragagent-ingest --directory data/raw --force-reingest

# Dry run to preview what would be processed
uv run ragagent-ingest --directory data/raw --dry-run

# Include text files (.txt, .md)
uv run ragagent-ingest notes.md --include-text
```

### Incremental Updates

Handle document additions, modifications, and deletions without full re-ingestion:

```bash
# Scan for changes and update system
uv run ragagent-update data/raw

# Dry run to see what would change
uv run ragagent-update data/raw --dry-run

# Limit to specific file extensions
uv run ragagent-update data/raw --extensions .pdf .txt
```

The update system will:
- Detect new files and ingest them
- Detect modified files (by content checksum) and re-ingest them
- Detect deleted files and remove them from the knowledge graph and vector store
- Preserve approved entities and relationships where possible

### Entity Discovery and Curation
```bash
# Generate entity discovery report
uv run ragagent-discover --min-confidence 0.7 --max-candidates 500

# Discovery with semantic merge suggestions
uv run ragagent-discover --enable-semantic-merge

# Review entity candidates (Typer-based CLI)
uv run ragagent-review

# Common review commands:
uv run ragagent-review queue --status pending --entity-type SYSTEM --limit 30
uv run ragagent-review show "<name or id>"
uv run ragagent-review approve "<id>"
uv run ragagent-review reject "<id>"
uv run ragagent-review merge "<source-id>" "<target-id>"
uv run ragagent-review batch-approve --min-confidence 0.9 --dry-run

# Normalization table operations
uv run ragagent-review normalization queue
uv run ragagent-review normalization show "<canonical>"
uv run ragagent-review normalization search "<term>"

# Launch interactive TUI (Textual-based interface)
uv run ragagent-review-interactive

# Features:
# - Three-panel layout: Candidate list | Details | Context
# - Keyboard shortcuts for fast navigation (press ? for help)
# - Batch operations (approve/reject multiple candidates)
# - Entity search and merge preview
# - Neighborhood issue detection and resolution
# - Session tracking with undo capability
```

**Session Tracking**: The interactive TUI tracks approval/rejection sessions with:
- Timestamped decision history
- Undo capability for recent actions
- Session statistics (approved/rejected/skipped counts)

### Relationship Curation

Review and curate relationship candidates extracted from documents:

```bash
# Review relationship candidates
uv run ragagent-review relationship queue --status pending --min-confidence 0.7
uv run ragagent-review relationship show "<id or key>"
uv run ragagent-review relationship approve "<id or key>"
uv run ragagent-review relationship reject "<id or key>" --reason "too vague"

# Filter by relationship type
uv run ragagent-review relationship queue --rel-type DEPENDS_ON --status pending

# Batch operations for high-confidence relationships
uv run ragagent-review relationship batch-approve --min-confidence 0.8 --dry-run
uv run ragagent-review relationship batch-approve --min-confidence 0.8 --no-dry-run
```

**Relationship Validation**: The system automatically validates relationships during extraction:
- Confidence threshold filtering (default: 0.5)
- Entity existence validation with fuzzy matching
- IS_A semantic checks (rejects adjective targets)
- Maximum entity name length enforcement

### Retrieval and Querying

Query the knowledge graph using hybrid retrieval:

```bash
# Interactive query loop
uv run ragagent-query

# Single query and exit
uv run ragagent-query --query "What are the core components of the power system?"

# Verbose output (shows query analysis and retrieved chunks)
uv run ragagent-query -v -q "How does the battery health check procedure work?"

# Export results to JSON
uv run ragagent-query -q "Thermal control dependencies" --export results.json
```

The query system uses hybrid retrieval combining:
- Vector similarity search (Qdrant)
- Graph traversal (Neo4j)
- LLM-based response generation with source citations

### Testing and Code Quality
```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_ingestion/test_pdf_parser.py

# Run single test
uv run pytest tests/test_ingestion/test_pdf_parser.py::test_parse_basic_pdf

# Code formatting
uv run black src/ tests/

# Linting
uv run ruff check src/ tests/
```

### Docker Management
```bash
# View logs
docker-compose logs -f neo4j
docker-compose logs -f qdrant

# Restart services
docker-compose restart neo4j
docker-compose restart qdrant

# Stop all services
docker-compose down

# Stop and remove volumes (DESTRUCTIVE - deletes all data)
docker-compose down -v
```

## Architecture

### Pipeline Architecture

The system uses a **stage-based pipeline pattern** (`src/pipeline/base.py`):
- `PipelineStage`: Abstract base class for individual processing stages
- `PipelineContext`: Pydantic model carrying state between stages (file path, parsed document, chunks, embeddings, stats)
- `Pipeline`: Orchestrator that runs stages sequentially, with early exit on skip/error

### Pipeline Flow
The ingestion pipeline (`src/pipeline/ingestion_pipeline.py`) orchestrates:
1. **PDF Parsing** (Docling with OCR) or text file parsing
2. **Text Cleaning** (remove headers/footers, normalize whitespace)
3. **Optional Text Rewriting** (LLM-based, disabled by default)
4. **Hierarchical Chunking** (document → section → subsection → paragraph)
5. **Entity Extraction**:
   - spaCy NER (pattern-based + trained model)
   - LLM extraction (Anthropic or OpenAI)
   - Parallel execution when both enabled
6. **Relationship Extraction** (LLM-based)
7. **Entity Merging** (cross-extractor deduplication)
8. **Acronym Resolution** (context-aware expansion)
9. **Semantic Deduplication** (embedding-based similarity)
10. **Embedding Generation** (BAAI/bge-small-en-v1.5 by default)
11. **Dual Storage**:
    - Neo4j: Document nodes, Chunk hierarchy, EntityCandidate/RelationshipCandidate for curation
    - Qdrant: Chunk vectors with metadata for semantic search

### Key Data Models (`src/storage/schemas.py`)

**Entity Types**: SYSTEM, SUBSYSTEM, COMPONENT, PARAMETER, PROCEDURE, PROCEDURE_STEP, CONCEPT, DOCUMENT, STANDARD, ANOMALY, TABLE, FIGURE

**Relationship Types**: PART_OF, CONTAINS, DEPENDS_ON, CONTROLS, MONITORS, PROVIDES_POWER_TO, SENDS_DATA_TO, REFERENCES, PRECEDES, REQUIRES_CHECK, AFFECTS, IMPLEMENTS, SIMILAR_TO, CAUSED_BY, MITIGATED_BY, REFERENCES_TABLE, REFERENCES_FIGURE, etc.

**Curation Flow**:
- Extraction creates `EntityCandidate` and `RelationshipCandidate` nodes in Neo4j
- Each candidate has `status` (PENDING/APPROVED/REJECTED/INACTIVE/MERGED_INTO_ENTITY), `candidate_key` (deterministic ID), provenance tracking
- Review interface (`ragagent-review` or `ragagent-review-interactive`) allows manual approval/rejection/merging
- Approved candidates can be promoted to actual Entity nodes

### Module Organization

```
src/
├── ingestion/           # PDF parsing, text cleaning, chunking
│   ├── pdf_parser.py         # Docling-based PDF parsing
│   ├── text_file_parser.py   # Lightweight text/markdown parsing
│   ├── text_cleaner.py       # Pattern-based cleaning
│   ├── text_rewriter.py      # Optional LLM rewriting
│   ├── chunker.py            # Hierarchical chunking
│   └── metadata_extractor.py
│
├── extraction/          # Entity and relationship extraction
│   ├── spacy_extractor.py    # Pattern + NER extraction
│   ├── llm_extractor.py      # LLM-based extraction
│   ├── entity_merger.py      # Cross-extractor merging
│   ├── dependency_extractor.py  # Syntactic relationship extraction
│   ├── pattern_extractor.py     # Rule-based relationships
│   ├── relationship_validator.py # Post-extraction filtering
│   └── models.py             # Pydantic schemas for extraction
│
├── normalization/       # Entity deduplication and normalization
│   ├── string_normalizer.py  # Text normalization
│   ├── fuzzy_matcher.py      # Fuzzy string matching
│   ├── acronym_resolver.py   # Context-aware acronym expansion
│   ├── entity_deduplicator.py # Embedding-based deduplication
│   └── normalization_table.py # Canonical name management
│
├── storage/             # Database managers
│   ├── neo4j_manager.py      # Graph database operations
│   ├── qdrant_manager.py     # Vector database operations
│   └── schemas.py            # Pydantic models for entities/relationships
│
├── curation/            # Manual review interface
│   ├── review_interface.py   # Typer-based CLI
│   ├── entity_approval.py    # Approval/rejection logic
│   ├── batch_operations.py   # Batch processing utilities
│   └── interactive/          # Textual-based TUI
│       ├── app.py                # Main TUI application
│       ├── widgets/              # Custom widgets (command palette, context panel, etc.)
│       ├── keybindings.py        # Keyboard shortcuts
│       ├── session_tracker.py    # Session state and undo
│       └── preferences.py        # User preferences
│
├── retrieval/           # Query parsing and hybrid retrieval
│   ├── query_parser.py       # Natural language query analysis
│   ├── vector_retriever.py   # Qdrant-based semantic search
│   ├── graph_retriever.py    # Neo4j graph traversal
│   ├── hybrid_retriever.py   # Combined retrieval strategy
│   ├── reranker.py           # Result reranking
│   ├── response_generator.py # LLM answer generation
│   ├── research_agent.py     # Iterative deep research
│   └── models.py             # Result schemas
│
├── pipeline/            # Orchestration
│   ├── base.py               # Pipeline/PipelineStage abstractions
│   ├── ingestion_pipeline.py # Main document processing pipeline
│   ├── discovery_pipeline.py # Entity discovery and analysis
│   ├── update_pipeline.py    # Incremental updates
│   └── stages/               # Individual pipeline stages
│       ├── parsing.py, checkpoint.py, cleaning.py, chunking.py
│       ├── extraction.py, embedding.py, storage.py
│
└── utils/               # Shared utilities
    ├── config.py             # Configuration management
    ├── embeddings.py         # Embedding generation (fastembed)
    ├── llm_client.py         # LLM provider abstraction
    ├── candidate_keys.py     # Deterministic key generation
    ├── progress.py           # Progress tracking
    └── logger.py             # Logging setup
```

### Checkpointing and Resume Logic

The ingestion pipeline uses **deterministic document IDs** (checksum-based) and tracks ingestion status:
- Each document has `ingestion_status` (ingesting/completed/failed) and `checksum`
- On ingestion attempt:
  1. If document exists with matching checksum and status=completed → **skip** (unless `--force-reingest`)
  2. If document exists but status≠completed or checksum differs → **cleanup old chunks**, then re-ingest
  3. On failure → rollback partial chunks, mark status=failed
- Chunk storage is idempotent (upsert by chunk_id in both Neo4j and Qdrant)

### Configuration System

Configuration is hierarchical:
1. **Base**: `config/config.yaml` (checked into git)
2. **Environment**: `.env` file (secrets, database URLs)
3. **Override**: `--config` CLI flag

Key config files:
- `config/config.yaml`: Main settings (chunking, extraction, normalization)
- `config/cleaning_patterns.yaml`: Text cleaning patterns (headers/footers/noise)
- `config/extraction_prompts.yaml`: LLM prompts for entity/relationship extraction
- `config/entity_patterns.jsonl`: spaCy pattern rules
- `config/acronym_overrides.yaml`: Manual acronym mappings
- `config/normalization_rules.yaml`: Entity normalization rules

Access config via: `from src.utils.config import load_config; config = load_config()`

**Component-Specific Overrides**: Use double underscores in `.env` to configure specific components:
```bash
# Override extraction LLM
EXTRACTION__LLM__PROVIDER=openai
EXTRACTION__LLM__MODEL=gpt-4.0
EXTRACTION__LLM__TEMPERATURE=0.0

# Override text rewriting LLM
INGESTION__TEXT_REWRITING__LLM__PROVIDER=anthropic
INGESTION__TEXT_REWRITING__LLM__MODEL=claude-3-5-sonnet-20241022
```

### Extraction Strategy

**Dual Extraction (spaCy + LLM)**:
- **spaCy**: Fast, pattern-based, good for known entity types (SYSTEM, COMPONENT, etc.)
- **LLM**: Context-aware, better for descriptions and relationships
- **Merging**: `EntityMerger` cross-references entities by normalized names, combines confidence scores, resolves type conflicts

**Parallel Execution**: When both extractors are enabled, they run concurrently using `ThreadPoolExecutor`

**Entity Candidates**: Merged entities are stored as `EntityCandidate` nodes with:
- `candidate_key`: Deterministic ID (type:normalized_name)
- `canonical_name`: Primary name
- `aliases`: Alternative names (including acronym expansions)
- `conflicting_types`: If extractors disagree on type
- `provenance_events`: List of where/when/how entity was observed
- `dedup_suggestions`: Embedding-based merge candidates

### Embedding Management

Embeddings are generated using `fastembed` (default: BAAI/bge-small-en-v1.5, 384 dimensions).

**Important**: If you change the embedding model or dimension in `.env`:
```bash
# Must recreate Qdrant collections to match new dimension
uv run ragagent-setup --recreate-qdrant
```

Embedding cache is stored in memory during pipeline execution, cleared on close.

### Neo4j Schema

**Constraints**:
- Unique `id` for all entity types (SYSTEM, SUBSYSTEM, etc.)
- Unique `id` and `candidate_key` for EntityCandidate
- Unique `id` and `candidate_key` for RelationshipCandidate
- Unique `id` for Chunk nodes

**Indexes**:
- `canonical_name` (text search)
- `entity_type` (filtering)
- `status` (curation workflow)
- Full-text search on entity/candidate properties

**Key Relationships**:
- `(Chunk)-[:PART_OF]->(Document)`
- `(Chunk)-[:PARENT_CHUNK]->(Chunk)` (hierarchy)
- `(EntityCandidate)-[:OBSERVED_IN]->(Chunk)` (provenance)
- Custom relationships between entities (CONTAINS, DEPENDS_ON, etc.)

## Development Guidelines

### Code Style and Conventions

**Python Standards**:
- Python 3.12+, 4-space indents, max line length 100 (Black/Ruff enforced)
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Type hints required (`disallow_untyped_defs=true`)
- Imports sorted by Ruff (E, W, F, I, N, UP rules enabled)

**Code Organization**:
- Keep side effects thin and injectable
- Isolate retrieval and ingestion logic to enable mocking in tests
- Add new features inside existing domain modules (src/ingestion, src/extraction, etc.) rather than creating new top-level packages
- Extend CLI helpers in scripts/ rather than duplicating logic
- Import style: Use `from src.module import Class` (not relative imports)
- Pydantic models for everything: Use Pydantic BaseModel, not dataclasses

**Testing**:
- Place tests in `tests/`; mirror `src/` layout for new modules
- Filenames `test_*.py`, functions `test_*`
- Mock Neo4j/Qdrant/LLMs in unit tests
- Target coverage on touched code paths
- Add regression tests for bug fixes

**Commit Guidelines**:
- Use short, imperative commit messages (e.g., `add qdrant schema helper`, `fix ingestion batch logging`)
- Keep each commit focused
- Before PRs: run `pytest`, `black`, `ruff`
- Call out required migrations or data backfills in PR descriptions

### Adding New Entity Types
1. Add to `extraction.entity_types` in `config/config.yaml`
2. Add patterns to `config/entity_patterns.jsonl` (spaCy)
3. Update extraction prompts in `config/extraction_prompts.yaml` (LLM)
4. Update `EntityType` enum in `src/storage/schemas.py`
5. Re-run discovery: `uv run ragagent-discover`

### Adding New Relationship Types
1. Add to `extraction.relationship_types` in `config/config.yaml`
2. Update extraction prompts in `config/extraction_prompts.yaml`
3. Update `RelationshipType` enum in `src/storage/schemas.py`

### Modifying Extraction Prompts
Edit `config/extraction_prompts.yaml`:
- `entity_extraction`: Main entity extraction prompt
- `relationship_extraction`: Main relationship extraction prompt
- `few_shot_examples`: Add domain-specific examples
- `table_extraction`/`figure_extraction`: Specialized prompts

### Testing with Mock Services
Use fixtures in `tests/conftest.py`:
```python
def test_ingestion_pipeline(mock_neo4j, mock_qdrant, config):
    pipeline = IngestionPipeline(config)
    # Test without live Neo4j/Qdrant
```

### Logging
Logs are written to `logs/ragagent2.log` with rotation (100MB, 5 backups).

Verbosity:
- Default: INFO level
- Verbose: `--verbose` flag or `LOG_LEVEL=DEBUG` in `.env`
- External HTTP logs (httpx, openai, docling) are suppressed unless DEBUG

## Troubleshooting

### Neo4j Connection Issues
```bash
# Check Neo4j status
docker-compose ps neo4j
docker-compose logs neo4j

# Test connection
curl http://localhost:7474

# Restart
docker-compose restart neo4j
```

### Qdrant Connection Issues
```bash
# Check Qdrant health
curl http://localhost:6333/health

# View collections
curl http://localhost:6333/collections

# Restart
docker-compose restart qdrant
```

### LLM Endpoint Issues
```bash
# Test OpenAI-compatible endpoint
curl "${OPENAI_BASE_URL:-https://api.openai.com/v1}/models"

# Check model availability
# Ensure OPENAI_MODEL or ANTHROPIC_MODEL in .env matches available models
```

### Memory Issues
If encountering OOM:
- Reduce `pipeline.batch_size` in config (default: 10)
- Reduce `pipeline.max_workers` (default: 4)
- Reduce `embedding_batch_size` in config (default: 32)
- Increase Docker Desktop memory limits
- Use smaller LLM models (e.g., gpt-4.1-mini instead of gpt-4)

### Embedding Dimension Mismatch
If you see Qdrant errors about dimension mismatch:
```bash
# Recreate collections with new embedding dimension
uv run ragagent-setup --recreate-qdrant
```

## Important Notes

- **Uses `uv` package manager**: Always prefix commands with `uv run` to ensure correct environment
- **Checksum-based idempotency**: Re-ingesting same file (same content) is a no-op unless `--force-reingest`
- **Entity curation is manual**: Extracted entities start as candidates; use `ragagent-review` or `ragagent-review-interactive` to approve/reject
- **Text rewriting is disabled by default**: Enable in config if needed (adds latency)
- **Neo4j APOC required**: Docker Compose config includes APOC plugin; don't use vanilla Neo4j
- **Python 3.12+ required**: Uses modern type hints and features
- **Context7 integration**: Always use context7 MCP tools when you need code generation, setup or configuration steps, or library/API documentation. This means you should automatically use the Context7 MCP tools to resolve library id and get library docs without me having to explicitly ask.

## Reference Documentation

- Architecture details: `plans/graph-rag-architecture.md`
- Phase breakdowns: `plans/phase-*.md`
- Neo4j schema: `docs/neo4j_schema_implementation.md`
- Qdrant operations: `docs/qdrant_manager_guide.md`
- Enhancements: `plans/enhancements-summary.md`
- Testing guide: `docs/TESTING.md`

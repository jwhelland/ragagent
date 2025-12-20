# Graph RAG System for Satellite Technical Documents

A comprehensive Graph RAG (Retrieval-Augmented Generation) system designed to process and query 100-500 satellite technical documents and standard operating procedures. The system combines vector similarity search with knowledge graph traversal to enable complex queries across technical documentation.

## Features

- **PDF Ingestion**: Parse PDFs using Docling with OCR support
- **Hierarchical Chunking**: Multi-level document representation (document → section → subsection → paragraph)
- **Entity Extraction**: Dual extraction using spaCy NER + LLM (Anthropic or OpenAI)
- **Knowledge Graph**: Neo4j graph database for entities and relationships
- **Vector Search**: Qdrant vector database for semantic search
- **Hybrid Retrieval**: Combined vector and graph search strategies
- **Entity Curation**: Manual review interface for entity approval and merging
- **Incremental Updates**: Handle document modifications without full reprocessing
- **Relationship Provenance**: Track where relationships were found with source citations

## Architecture

The system is built in 6 phases:
1. **Foundation** (Weeks 1-2): PDF parsing, chunking, embedding, basic storage
2. **Entity Extraction** (Weeks 3-4): spaCy + LLM extraction, entity candidates
3. **Normalization & Curation** (Weeks 5-6): Entity deduplication, manual review
4. **Retrieval System** (Weeks 7-8): Hybrid search, query processing, response generation
5. **Incremental Updates** (Week 9): Document change detection and differential updates
6. **Discovery & Polish** (Week 10): Entity discovery, testing, deployment

## Technology Stack

- **Python 3.12+**
- **Neo4j 5.x** - Graph database
- **Qdrant** - Vector database
- **Docling** - PDF parsing with OCR
- **spaCy** - NLP and NER
- **Anthropic/OpenAI** - LLM integration (configurable)
- **FastEmbed/Sentence Transformers** - Embeddings
- **Pydantic** - Configuration and validation
- **Typer** - CLI interface

## Prerequisites

- Python 3.12.1 or higher
- Docker and Docker Compose (for Neo4j and Qdrant)
- Anthropic (optional, for local LLM) or OpenAI API key

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ragagent2
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
uv sync

# Install spaCy transformer model
uv run spacy download en_core_web_lg
```

### 3. Start Infrastructure Services

```bash
# Start Neo4j and Qdrant using Docker Compose
docker-compose up -d

# Verify services are running
docker-compose ps

# Check Neo4j: http://localhost:7474 (username: neo4j, password: ragagent2024)
# Check Qdrant: http://localhost:6333/dashboard
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and configure your settings
# - LLM provider (openai or anthropic)
# - API keys (if using OpenAI or Anthropic)
# - Embedding model
# - Data paths
```

### 5. Configure OpenAI-Compatible Endpoint

```bash
# Optional: point to a self-hosted OpenAI-compatible server
export OPENAI_BASE_URL=http://localhost:8000/v1

# Set your API key (required for api.openai.com; local servers may accept a dummy value)
export OPENAI_API_KEY=sk-...
```

### 6. Initialize Databases

```bash
# Initialize Neo4j schema and Qdrant collections
uv run ragagent-setup

# Recreate Qdrant collections after embedding/model changes
uv run ragagent-setup --recreate-qdrant
```

## Configuration

Configuration is managed through:
- `config/config.yaml` - Main configuration file
- `.env` - Environment variables (API keys, database connections)

### Key Configuration Sections

**Ingestion**:
- PDF parser settings (OCR, table/figure extraction)
- Text cleaning patterns
- Chunking strategy and sizes

**Extraction**:
- spaCy model and patterns
- LLM provider and model selection
- Entity and relationship types

**Normalization**:
- Fuzzy matching thresholds
- Embedding similarity thresholds
- Acronym resolution settings

**Retrieval**:
- Vector search parameters (top-k, min score)
- Graph traversal settings (max depth)
- Hybrid search strategy
- Reranking weights

## Usage

All CLI helpers live in `scripts/` and are exposed as console commands. Run them with `uv run …`
so dependencies and environment are consistent.

### Setup Databases (Neo4j + Qdrant)

```bash
# Create constraints/indexes and collections
uv run ragagent-setup

# Recreate Qdrant collections after embedding/model changes
uv run ragagent-setup --recreate-qdrant
```

### Ingest Documents

```bash
# Ingest every PDF under data/raw (recursive)
uv run ragagent-ingest --directory data/raw --batch-size 2

# Ingest explicit files with a specific config and verbose logging
uv run ragagent-ingest docs/file1.pdf docs/file2.pdf -c config/config.yaml --verbose

# See what would be processed without running the pipeline
uv run ragagent-ingest --directory data/raw --dry-run
```

Key flags: `--directory` (or pass paths), `--batch-size`, `--config`, `--verbose`,
`--dry-run`, `--force-reingest` (bypass checkpoint skip and reprocess a completed doc).

### Run Entity Discovery

```bash
# Generate a discovery report with confidence filtering
uv run ragagent-discover --min-confidence 0.7 --max-candidates 500

# Limit to certain statuses/types and skip visualization
uv run ragagent-discover --status approved --status pending --candidate-type SYSTEM --no-viz

# Enable semantic merge suggestions (requires embedding model)
uv run ragagent-discover --enable-semantic-merge --max-merge-suggestions 50
```

Useful flags: `--output-dir`, `--status` (repeatable), `--candidate-type` (repeatable),
`--min-cooccurrence`, `--max-edges`, `--max-clusters`, `--verbose`.

### Review and Curate Entities

```bash
uv run ragagent-review
```

Launches the Typer-based review CLI. Common workflow:
- Browse candidates: `uv run ragagent-review queue --status pending --entity-type SYSTEM --limit 30`
- Inspect details: `uv run ragagent-review show "<name or id>"`
- Approve/reject one: `uv run ragagent-review approve "<id>"` or `... reject "<id>"`
- Fix metadata: `uv run ragagent-review edit "<id>" --name "New Name" --type SYSTEM --confidence 0.92`
- Merge duplicates: `uv run ragagent-review merge "<source-id>" "<target-id>"`
- Batch approve: `uv run ragagent-review batch-approve --min-confidence 0.9 --dry-run --preview-limit 10`
- Normalization table tools: `uv run ragagent-review normalization queue`, `... normalization show "<canonical>"`, `... normalization search "<term>"`, `... normalization stats`

Flags you may need: `--config` to point at a different config file, `--table-path` for an alternate normalization table, `--verbose` via env `RICH_COLOR_SYSTEM=standard` if your terminal needs it.

## Project Structure

```
ragagent2/
├── src/
│   ├── ingestion/          # PDF parsing, cleaning, chunking
│   ├── extraction/         # Entity and relationship extraction
│   ├── normalization/      # Entity deduplication and normalization
│   ├── storage/            # Neo4j and Qdrant managers
│   ├── retrieval/          # Query parsing and hybrid retrieval
│   ├── curation/           # Manual review interface
│   ├── pipeline/           # Orchestration pipelines
│   └── utils/              # Configuration, logging, embeddings
├── scripts/                # CLI entry points
├── tests/                  # Test suites
├── config/                 # Configuration files
│   ├── config.yaml
│   ├── cleaning_patterns.yaml
│   ├── entity_patterns.jsonl
│   └── extraction_prompts.yaml
├── data/                   # Data directories
│   ├── raw/               # Input PDFs
│   ├── processed/         # Parsed documents
│   ├── entities/          # Entity candidates
│   └── normalization/     # Normalization tables
├── plans/                  # Architecture and planning docs
├── docker-compose.yml      # Infrastructure setup
├── pyproject.toml         # Python dependencies
└── README.md
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_ingestion/test_pdf_parser.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Adding New Entity Types

1. Add entity type to `config/config.yaml` under `extraction.entity_types`
2. Add patterns to `config/entity_patterns.jsonl` (for spaCy)
3. Update extraction prompts in `config/extraction_prompts.yaml`
4. Run discovery to find entities of the new type

### Customizing Extraction Prompts

Edit `config/extraction_prompts.yaml` to customize:
- Entity extraction prompts
- Relationship extraction prompts
- Few-shot examples
- Table and figure extraction

## Monitoring and Logging

Logs are written to `logs/ragagent2.log` with rotation (100MB, 5 backups).

```bash
# View logs
tail -f logs/ragagent2.log

# Check Docker logs
docker-compose logs -f neo4j
docker-compose logs -f qdrant
```

## Troubleshooting

### Neo4j Connection Issues

```bash
# Check Neo4j is running
docker-compose ps neo4j

# View Neo4j logs
docker-compose logs neo4j

# Restart Neo4j
docker-compose restart neo4j
```

### Qdrant Connection Issues

```bash
# Check Qdrant is running
docker-compose ps qdrant

# Check Qdrant health
curl http://localhost:6333/health
```

### LLM Endpoint Issues

```bash
# Check the OpenAI-compatible endpoint is reachable
curl "${OPENAI_BASE_URL:-https://api.openai.com/v1}/models"

# Ensure the configured model is available on the endpoint
# (endpoint-specific; adjust path if needed)
```

### Memory Issues

If encountering out-of-memory errors:
- Reduce `batch_size` in config
- Reduce `max_workers` for parallel processing
- Increase Docker memory limits in Docker Desktop settings
- Use smaller LLM models (e.g., llama3.1:8b instead of 70b)

## Performance Optimization

- **Embedding Generation**: Adjust `embedding_batch_size` based on available GPU memory
- **Parallel Processing**: Tune `max_workers` for your CPU
- **Vector Search**: Adjust `top_k` for faster but less comprehensive results
- **Graph Traversal**: Limit `max_depth` for faster queries
- **Caching**: Enable query caching for frequently accessed data

## Backup and Recovery

### Backup

```bash
# Backup Neo4j
docker-compose exec neo4j neo4j-admin database dump neo4j --to-path=/backups

# Backup Qdrant
curl -X POST http://localhost:6333/collections/document_chunks/snapshots

# Backup configuration and data
tar -czf ragagent2-backup.tar.gz config/ data/normalization/
```

### Restore

```bash
# Restore Neo4j
docker-compose exec neo4j neo4j-admin database load neo4j --from-path=/backups

# Restore Qdrant snapshots via API
# See Qdrant documentation for snapshot restoration
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## Documentation

- [Architecture](plans/graph-rag-architecture.md) - Detailed system architecture
- [Developer Tasks](plans/developer-tasks.md) - Implementation task breakdown
- [Enhancements](plans/enhancements-summary.md) - Key enhancements and their impact

## Roadmap

See detailed milestones in `plans/phase-*-*.md`.

## Acknowledgments

Built using:
- Neo4j for knowledge graph storage
- Qdrant for vector similarity search
- Docling for PDF parsing
- spaCy for NER
- OpenAI/Anthropic for LLM capabilities

# RAG Agent (OpenAI + Qdrant + Neo4j)

A production-ready Retrieval-Augmented Generation system with strict citation tracking, hybrid vector+graph retrieval, and comprehensive PDF ingestion pipeline.

## Overview

- **Goal**: Python RAG agent using OpenAI SDK for LLM orchestration
- **Storage**: Hybrid retrieval via Qdrant (1024-d vectors) and Neo4j (knowledge graph)
- **Ingestion**: English PDFs parsed with Docling, EasyOCR fallback for scanned documents, preserves tables
- **Deployment**: Docker Compose stack with all services included
- **Grounding**: Strict citation requirements with verification (document ID, page, table IDs)

## Quick Start

### Prerequisites

- Docker and Docker Compose (v2+)
- Python ≥3.13.1 (for local development)
- OpenAI API key

### 1. Clone and Configure

```bash
# Clone repository
git clone <repository-url>
cd ragagent

# Copy environment template and add your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 2. Start Services

```bash
# Start entire stack (app, Qdrant, Neo4j, embeddings, worker)
docker compose up -d

# Check service health
curl http://localhost:8000/health
# Should return: {"status":"healthy"}
```

**Service Ports:**
- FastAPI app: http://localhost:8000
- Qdrant UI: http://localhost:6333/dashboard
- Neo4j Browser: http://localhost:7474 (user: neo4j, pass: neo4j)

### 3. Ingest Documents

```bash
# Place PDFs in data/input_pdfs/
mkdir -p data/input_pdfs
# Copy your PDFs here

# Run ingestion pipeline
docker compose exec worker python -m ragagent.ingest.cli data/input_pdfs --out data/processed
or
make ingest

# Monitor progress in logs
docker compose logs -f worker
or 
make logs-worker
```

### 4. Query the System

```bash
# Example chat request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 8}'
```

Or use the provided Makefile commands (see below).

## Data Ingestion Pipeline

The ingestion pipeline turns raw PDFs into chunked, enriched, and indexed content that powers retrieval.

- **1. Discover PDFs**
  - Recursively scans the input directory (`data/input_pdfs/` in Docker, or any path you pass to `ragagent.ingest.cli`) for `*.pdf`.
  - Computes a SHA-256 checksum per file and tracks them in a manifest (`data/manifest.json`) so already-processed files are skipped.

- **2. Parse and Extract Content**
  - Tries **Docling** first for layout-aware extraction (pages, flowing text, tables with Markdown rendering).
  - If Docling is unavailable or yields empty content, falls back to **PyMuPDF + OCR**:
    - Extracts text with PyMuPDF.
    - If a page has no text, renders it as an image and runs **EasyOCR** (if installed) using the languages in `OCR_LANGUAGES`.
  - Writes a per-document artifact to `data/processed/<doc_id>/extraction.json` (overridable via `--out`).

- **3. Chunking and Metadata Enrichment**
  - Splits page text into semantic chunks using the project’s chunker (`src/ragagent/chunking/splitter.py`).
  - Emits additional chunks for tables (as Markdown where available), with stable `table_id`-based chunk IDs.
  - Attaches rich metadata to every chunk (document ID, page, SHA-256, source path, optional `table_id`).
  - Runs a lightweight NLP pass to attach detected entities and keyphrases for downstream graph linking.

- **4. Embedding and Vector Indexing**
  - Batches chunk texts and calls the configured embeddings service via `EmbeddingsClient` (`EMBEDDINGS_ENDPOINT`).
  - Uses exponential backoff and retries on transient failures.
  - Upserts embeddings and metadata into Qdrant (`QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_VECTOR_SIZE`, `QDRANT_DISTANCE`).

- **5. Graph Enrichment**
  - Registers each document and its sections in Neo4j (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`).
  - Creates “episode” nodes for chunks with their body text and provenance (document, page, table).
  - Upserts entity nodes and links them to relevant chunks, enabling graph-augmented retrieval.

- **6. Tracing, Logging, and Idempotency**
  - All major steps (extract, embed, upsert to vector store/graph) are wrapped in OpenTelemetry spans.
  - Structured logs include keys such as `doc_id`, `page`, `chunk_id`, and `sha256` for troubleshooting.
  - Re-running ingestion is safe: already-processed files (by SHA-256) are skipped unless you delete or reset the manifest.

### Ingestion Entry Points and Options

You can run ingestion either from Docker (recommended) or directly from Python:

- **Docker worker (default path configuration):**
  ```bash
  docker compose exec worker python -m ragagent.ingest.cli data/input_pdfs --out data/processed
  ```

- **Local Python (without Docker worker):**
  ```bash
  uv pip install -e .
  python -m ragagent.ingest.cli path/to/pdfs --out data/processed --manifest data/manifest.json
  ```

Key levers:
- `input_dir` (positional): Where PDFs are read from (recursive).
- `--out`: Directory for extracted artifacts and per-document JSON (default: `data/processed`).
- `--manifest`: JSON manifest used for idempotency (default: `data/manifest.json`).
- `OCR_LANGUAGES`: Comma-separated languages for EasyOCR (e.g. `en`, `en,de`).
- `EMBEDDINGS_ENDPOINT`, `QDRANT_*`, `NEO4J_*`: Control where embeddings, vectors, and graph data are written.

## Architecture

### Data Flow

```
User Query → FastAPI (/chat)
  ↓
Vector Retrieval (Qdrant top-k=8)
  ↓
Graph Expansion (Neo4j)
  ↓
Context Assembly (merge, dedupe, assign [S1], [S2]... tags)
  ↓
LLM Generation (OpenAI with citation instructions)
  ↓
Verification (token overlap + optional LLM check)
  ↓
Response (answer + citations + verification report)
```

### Key Components

- **Vector Store**: Qdrant with 1024-d embeddings (intfloat/e5-large-v2)
- **Graph Store**: Neo4j for entity/relationship tracking
- **Embeddings**: Local Hugging Face TEI service (CPU-optimized)
- **Ingestion**: Docling for PDF extraction, EasyOCR for scanned pages
- **Agent**: OpenAI SDK orchestration with strict citation system

## Makefile Commands

The project includes a Makefile for common operations:

```bash
make   # get list of available commands
```

## Configuration

All configuration via `.env` file. Key variables:

```bash
# Required
OPENAI_API_KEY=sk-...

# Service URLs (auto-configured in Docker)
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
EMBEDDINGS_ENDPOINT=http://localhost:8080/embed

# Qdrant settings
QDRANT_COLLECTION=documents
QDRANT_VECTOR_SIZE=1024
QDRANT_DISTANCE=Cosine

# Neo4j credentials
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j

# OCR settings
OCR_ENGINE=easyocr
OCR_LANGUAGES=en

# Logging
LOG_LEVEL=INFO
APP_ENV=development
```

See `.env.example` for full configuration options.

## Troubleshooting

### Service Fails to Start

**Symptom**: Docker Compose exits immediately or services show as unhealthy

**Solutions**:
```bash
# Check logs for specific service
docker compose logs app
docker compose logs qdrant
docker compose logs neo4j

# Common issues:
# 1. Port conflicts - check if ports 8000, 6333, 7474, 7687 are available
lsof -i :8000

# 2. Missing .env file
cp .env.example .env
# Add OPENAI_API_KEY

# 3. Docker resource limits - increase Docker Desktop memory (recommend ≥8GB)
```

### Qdrant Collection Not Found

**Symptom**: `CollectionNotFound` error when querying

**Solutions**:
```bash
# Check if collection exists
curl http://localhost:6333/collections/documents

# If missing, collection should auto-create on first ingestion
# Or manually reinitialize:
docker compose restart qdrant-init

# Verify collection was created
curl http://localhost:6333/collections/documents | jq
```

### Neo4j Connection Refused

**Symptom**: `ServiceUnavailable` or connection timeout errors

**Solutions**:
```bash
# Check Neo4j is running and ready
docker compose logs neo4j | grep "Started"

# Neo4j takes 20-30 seconds to fully start
# Wait and retry

# Test connection
docker compose exec neo4j cypher-shell -u neo4j -p neo4j "RETURN 1;"

# If authentication fails, reset password:
docker compose down neo4j
docker volume rm ragagent_neo4j_data
docker compose up -d neo4j
```

### Ingestion Fails

**Symptom**: Ingestion errors with PDFs, extraction failures

**Solutions**:
```bash
# Check worker logs for specific errors
docker compose logs -f worker

# Common issues:
# 1. Missing embeddings service
docker compose ps embeddings
# Should show "Up" - if not, restart:
docker compose restart embeddings

# 2. PDF extraction errors
# - Ensure PDFs are not corrupted (try opening in viewer)
# - For scanned PDFs, EasyOCR fallback runs automatically
# - Check OCR_LANGUAGES in .env matches document language

# 3. Out of memory during processing
# Increase Docker Desktop memory allocation or process smaller batches:
docker compose exec worker python -m ragagent.ingest.cli data/input_pdfs --batch-size 5
```

### Query Returns No Results

**Symptom**: Queries return empty context or "no relevant documents found"

**Solutions**:
```bash
# 1. Verify documents were ingested
curl http://localhost:6333/collections/documents | jq '.result.points_count'
# Should show > 0

# 2. Check Neo4j has document nodes
docker compose exec neo4j cypher-shell -u neo4j -p neo4j \
  "MATCH (d:Document) RETURN count(d);"

# 3. Test vector search directly
curl -X POST http://localhost:6333/collections/documents/points/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [...], "limit": 5}'

# 4. Check embedding service is responding
curl http://localhost:8080/health
```

### Citation Verification Failures

**Symptom**: Verification status shows "weak" or "unsupported" citations

**Solutions**:
- **Expected behavior**: Strict verification is working correctly
- Review `verification_report` in API response to see which citations failed
- Check if LLM is inventing facts not in retrieved context
- Consider adjusting citation instructions in `src/ragagent/agent/prompts.py`
- Enable LLM verification for more accurate checks (slower):
  ```bash
  # In evaluation
  python -m ragagent.eval.runner eval/questions.jsonl --enable-llm-verifier
  ```

### High Memory Usage

**Symptom**: Services consuming excessive memory

**Solutions**:
```bash
# Check resource usage
docker stats

# Neo4j memory settings (adjust in docker-compose.yml):
# NEO4J_server_memory_pagecache_size=8G  # Reduce if needed
# NEO4J_server_memory_heap_max__size=4G  # Reduce if needed

# For embeddings service, GPU acceleration helps:
# Uncomment GPU sections in docker-compose.yml (requires NVIDIA runtime)

# Restart services after changes
docker compose down
docker compose up -d
```

### Performance is Slow

**Symptom**: Queries take >5 seconds to complete

**Solutions**:
1. **Check retrieval latency**: Enable tracing (OpenTelemetry spans in logs)
2. **Tune Qdrant HNSW**: Adjust `ef` parameter in `src/ragagent/vectorstore/qdrant_store.py:search()`
   - Higher `ef` = more accurate but slower (default: 128)
   - Lower `ef` = faster but less accurate
3. **Reduce context size**: Lower `top_k` in query (default: 8)
4. **Enable GPU**: For embeddings and OCR (see docker-compose.yml)
5. **Add re-ranking**: Implement cross-encoder re-ranking (future work)

### Fresh Start Required

**Symptom**: Need to completely reset environment

**Solutions**:
```bash
# Nuclear option - removes all data
docker compose down -v

# Restart from scratch
docker compose up -d

# Re-run ingestion
docker compose exec worker python -m ragagent.ingest.cli data/input_pdfs
```

## Backup and Restore

### Qdrant Backup

Qdrant stores vectors in named volumes. To backup:

```bash
# 1. Create backup using Qdrant's snapshot API
curl -X POST 'http://localhost:6333/collections/documents/snapshots'

# Response will include snapshot name, e.g., "documents-2025-11-20-12-30-45.snapshot"

# 2. Copy snapshot from container
docker compose cp qdrant:/qdrant/storage/documents/snapshots/documents-2025-11-20-12-30-45.snapshot ./backups/

# Or backup entire volume
docker run --rm -v ragagent_qdrant_data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/qdrant-backup-$(date +%Y%m%d-%H%M%S).tar.gz -C /data .
```

### Qdrant Restore

```bash
# Option 1: Restore from snapshot (preferred)
# 1. Copy snapshot to container
docker compose cp ./backups/documents-snapshot.snapshot qdrant:/qdrant/storage/snapshots/

# 2. Restore via API
curl -X PUT 'http://localhost:6333/collections/documents/snapshots/documents-snapshot.snapshot' \
  -H 'Content-Type: application/json' \
  -d '{"location":"file:///qdrant/storage/snapshots/documents-snapshot.snapshot"}'

# Option 2: Restore entire volume
docker compose down qdrant
docker volume rm ragagent_qdrant_data
docker volume create ragagent_qdrant_data
docker run --rm -v ragagent_qdrant_data:/data -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/qdrant-backup-20251120-123045.tar.gz -C /data
docker compose up -d qdrant
```

### Neo4j Backup

Neo4j stores graph data in named volumes. To backup:

```bash
# Option 1: Export via cypher-shell (logical backup)
docker compose exec neo4j cypher-shell -u neo4j -p neo4j \
  "CALL apoc.export.cypher.all('/var/lib/neo4j/backups/backup-$(date +%Y%m%d).cypher', {format:'cypher-shell'})"

# Copy from container
docker compose cp neo4j:/var/lib/neo4j/backups/backup-20251120.cypher ./backups/

# Option 2: Full volume backup (physical backup)
docker compose stop neo4j
docker run --rm -v ragagent_neo4j_data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/neo4j-backup-$(date +%Y%m%d-%H%M%S).tar.gz -C /data .
docker compose start neo4j
```

### Neo4j Restore

```bash
# Option 1: Import from cypher dump
# 1. Clear existing data
docker compose exec neo4j cypher-shell -u neo4j -p neo4j \
  "MATCH (n) DETACH DELETE n;"

# 2. Copy backup to container
docker compose cp ./backups/backup-20251120.cypher neo4j:/var/lib/neo4j/backups/

# 3. Import
docker compose exec neo4j cypher-shell -u neo4j -p neo4j \
  < ./backups/backup-20251120.cypher

# Option 2: Restore entire volume
docker compose down neo4j
docker volume rm ragagent_neo4j_data
docker volume create ragagent_neo4j_data
docker run --rm -v ragagent_neo4j_data:/data -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/neo4j-backup-20251120-123045.tar.gz -C /data
docker compose up -d neo4j
```

### Backup Strategy Recommendations

**For Development:**
- Snapshot before major changes (ingestion, schema updates)
- Keep 3-5 recent snapshots locally

**For Production:**
- **Frequency**: Daily automated backups (3 AM UTC)
- **Retention**: 7 daily, 4 weekly, 12 monthly
- **Storage**: Off-site S3/GCS bucket with versioning
- **Verification**: Weekly restore test to staging environment
- **Monitoring**: Alert on backup failures

**Automation Example** (crontab):
```bash
# Daily backup at 3 AM
0 3 * * * /path/to/scripts/backup-qdrant.sh && /path/to/scripts/backup-neo4j.sh && \
  aws s3 sync /path/to/backups/ s3://my-rag-backups/$(date +\%Y-\%m-\%d)/
```

### Disaster Recovery

**RTO (Recovery Time Objective)**: ~15 minutes
**RPO (Recovery Point Objective)**: Last backup (24 hours for daily backups)

**Recovery Steps:**
1. Provision new environment (`docker compose up -d`)
2. Restore Qdrant snapshot (5 min)
3. Restore Neo4j volume (5 min)
4. Verify data integrity (`make test` or sample queries)
5. Resume operations

## Development

For detailed development documentation, architecture deep-dive, and code contribution guidelines, see `CLAUDE.md`.

### Running Locally (without Docker)

```bash
# Install dependencies
uv pip install -e .

# Start external services only
docker compose up -d qdrant neo4j embeddings

# Run app locally with hot reload
uvicorn ragagent.app:app --reload --port 8000

# Run tests
pytest -q
```

## Deployment Scope

**Single-Tenant Architecture**: This system is designed for single-tenant deployment. Key implications:

- No authentication/authorization on API endpoints by default
- Shared Qdrant collection and Neo4j graph per instance
- No per-user data isolation or access controls
- Rate limiting and quotas not implemented

For multi-tenant deployments, see **Optional Hardening** section below.

## Project Structure

```
ragagent/
├── src/ragagent/
│   ├── agent/          # RetrievalAgent, prompts, verification
│   ├── retrieval/      # Vector/graph retrievers, context assembly
│   ├── ingest/         # PDF ingestion pipeline
│   ├── vectorstore/    # Qdrant client
│   ├── graph/          # Neo4j integration
│   ├── embeddings/     # TEI embeddings client
│   └── eval/           # Evaluation harness
├── docker/             # Docker init scripts, configs
├── eval/               # Evaluation datasets
├── data/               # Input PDFs, processed outputs, manifests
├── tests/              # Test suite
└── tasks/              # Implementation phase checklists
```

## Resources

- **Plan**: `rag_agent_plan.md` - Architecture and design decisions
- **Developer Guide**: `CLAUDE.md` - Detailed code documentation
- **Tasks**: `tasks/` - Implementation checklists (Phases A-F)

## Optional Hardening

For production deployments or multi-tenant scenarios, consider these hardening measures:

### Authentication & Authorization

**Current State**: No auth - open API endpoints

**Hardening Options:**

1. **API Key Authentication** (Simple)
```python
# Add to src/ragagent/app.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Apply to endpoints
@app.post("/chat", dependencies=[Depends(verify_api_key)])
```

2. **OAuth2/JWT** (Multi-tenant)
```python
# Use FastAPI's OAuth2PasswordBearer
# Integrate with Auth0, Keycloak, or custom identity provider
# Add user_id to request context for data isolation
```

3. **mTLS** (Internal services)
```yaml
# Configure in docker-compose.yml for service-to-service auth
# Require client certificates for inter-service communication
```

### Rate Limiting

**Current State**: No rate limits

**Implementation Options:**

1. **slowapi** (Application-level)
```python
# Add to pyproject.toml: slowapi = "^0.1.9"
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(...):
    ...
```

2. **nginx/traefik** (Reverse proxy)
```nginx
# nginx.conf
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/m;

location /chat {
    limit_req zone=api_limit burst=5 nodelay;
    proxy_pass http://app:8000;
}
```

### Quotas & Cost Controls

**Current State**: Unlimited usage per user

**Hardening Options:**

1. **Token/Query Quotas**
```python
# Track per-user usage in Redis
# Enforce daily/monthly limits
# Return 429 when quota exceeded
```

2. **Context Size Limits**
```python
# Already configurable in ContextAssembler
# Enforce stricter limits: max_chunks=5, max_chars_per_chunk=500
# Prevents excessive LLM costs
```

3. **Async Job Queues**
```python
# Use Celery/RQ for long-running ingestion
# Prevents resource exhaustion from concurrent uploads
```

### Monitoring & Observability

**Current State**: Basic logging, OpenTelemetry instrumentation

**Hardening Options:**

1. **Prometheus + Grafana Stack**
```yaml
# Add to docker-compose.yml
prometheus:
  image: prom/prometheus
  volumes:
    - ./docker/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
```

2. **Custom Metrics**
```python
from prometheus_client import Counter, Histogram

query_counter = Counter('rag_queries_total', 'Total queries')
citation_failures = Counter('citation_verification_failures', 'Failed citations')
latency_histogram = Histogram('rag_query_latency_seconds', 'Query latency')
```

3. **Alerting Rules**
```yaml
# prometheus rules
groups:
  - name: rag_alerts
    rules:
      - alert: HighCitationFailureRate
        expr: rate(citation_verification_failures[5m]) > 0.1
        annotations:
          summary: "Citation failure rate > 10%"
      - alert: SlowQueries
        expr: histogram_quantile(0.95, rag_query_latency_seconds) > 5
        annotations:
          summary: "p95 latency > 5s"
```

### Network Security

**Current State**: All services exposed on localhost

**Hardening Options:**

1. **Internal Networks**
```yaml
# docker-compose.yml - isolate services
networks:
  frontend:  # app only
  backend:   # app, qdrant, neo4j, embeddings

app:
  networks: [frontend, backend]
qdrant:
  networks: [backend]  # not exposed to frontend
```

2. **TLS/HTTPS**
```yaml
# Terminate TLS at reverse proxy (nginx/traefik)
# Use Let's Encrypt for certificates
traefik:
  command:
    - "--certificatesresolvers.le.acme.email=admin@example.com"
    - "--certificatesresolvers.le.acme.tlschallenge=true"
```

3. **Secrets Management**
```bash
# Use Docker secrets or external vaults
# Never commit .env to version control (already in .gitignore)
docker secret create openai_api_key ./openai_key.txt
```

### Data Isolation (Multi-tenant)

**Current State**: Single shared collection/graph

**Hardening Options:**

1. **Qdrant Collections per Tenant**
```python
# Create tenant-specific collections: documents_tenant_{id}
# Filter queries by tenant_id in metadata
collection_name = f"documents_tenant_{user.tenant_id}"
```

2. **Neo4j Tenant Labels**
```cypher
# Add tenant_id property to all nodes
CREATE (d:Document {tenant_id: $tenant_id, ...})

# Query with tenant filter
MATCH (d:Document {tenant_id: $tenant_id})
```

3. **Database per Tenant** (High isolation)
```python
# Spin up dedicated Qdrant/Neo4j instances per tenant
# Higher resource cost but complete isolation
```

### Compliance & Privacy

**For regulated industries (HIPAA, GDPR, etc.):**

1. **Data Encryption at Rest**: Enable volume encryption
2. **Audit Logging**: Log all queries with user attribution
3. **Data Retention**: Implement automatic purging of old data
4. **Right to Deletion**: Add endpoints to remove user data from stores
5. **PII Redaction**: Scrub sensitive data during ingestion

### Hardening Checklist

- [ ] Enable authentication (API key minimum)
- [ ] Implement rate limiting (10 req/min per user)
- [ ] Add monitoring dashboard (Grafana + Prometheus)
- [ ] Set up alerting for failures (PagerDuty/Slack)
- [ ] Configure TLS/HTTPS for public access
- [ ] Isolate services on internal networks
- [ ] Implement backup automation (daily snapshots)
- [ ] Add health check monitoring (uptime checks)
- [ ] Enable audit logging for compliance
- [ ] Document incident response procedures

## Support

For issues, questions, or contributions, please refer to project documentation or open an issue.

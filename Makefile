.PHONY: help up down restart logs logs-app logs-worker logs-neo4j logs-qdrant \
        status ingest eval test clean backup-qdrant backup-neo4j restore health \
        dev install build shell-worker shell-app

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "RAG Agent - Available Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# Docker Compose Commands

up: ## Start all services in detached mode
	docker compose up -d
	@echo "✓ Services started. Check health with: make health"

down: ## Stop all services
	docker compose down
	@echo "✓ Services stopped"

restart: ## Restart all services
	docker compose restart
	@echo "✓ Services restarted"

status: ## Show status of all services
	docker compose ps

logs: ## View logs from all services (follow mode)
	docker compose logs -f

logs-app: ## View logs from app service only
	docker compose logs -f app

logs-worker: ## View logs from worker service only
	docker compose logs -f worker

logs-neo4j: ## View logs from neo4j service only
	docker compose logs -f neo4j

logs-qdrant: ## View logs from qdrant service only
	docker compose logs -f qdrant

health: ## Check health of all services
	@echo "Checking service health..."
	@echo ""
	@echo "FastAPI App:"
	@curl -s http://localhost:8000/health | jq '.' || echo "  ✗ Not responding"
	@echo ""
	@echo "Qdrant:"
	@curl -s http://localhost:6333/collections/documents | jq '.result.status' || echo "  ✗ Not responding"
	@echo ""
	@echo "Neo4j:"
	@docker compose exec -T neo4j cypher-shell -u neo4j -p neo4j "RETURN 'healthy' as status;" 2>&1 | grep -q "healthy" && echo "  ✓ Healthy" || echo "  ✗ Not responding"
	@echo ""
	@echo "Embeddings Service:"
	@curl -s http://localhost:8080/health | jq '.' || echo "  ✗ Not responding"

clean: ## Stop all services and remove volumes (WARNING: deletes all data)
	@echo "⚠️  WARNING: This will delete all data (Qdrant vectors, Neo4j graph, manifests)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v; \
		echo "✓ Services stopped and volumes removed"; \
	else \
		echo "Cancelled"; \
	fi

# Application Commands

ingest: ## Run ingestion pipeline on data/input_pdfs
	@if [ ! -d "data/input_pdfs" ]; then \
		mkdir -p data/input_pdfs; \
		echo "Created data/input_pdfs directory - add PDFs here first"; \
		exit 1; \
	fi
	@echo "Running ingestion pipeline..."
	docker compose exec worker python -m ragagent.ingest.cli data/input_pdfs --out data/processed --manifest data/manifest.json
	@echo "✓ Ingestion complete"

eval: ## Run evaluation suite
	@if [ ! -f "eval/questions.jsonl" ]; then \
		echo "Error: eval/questions.jsonl not found"; \
		exit 1; \
	fi
	@echo "Running evaluation suite..."
	docker compose exec app python -m ragagent.eval.runner eval/questions.jsonl --verbose
	@echo "✓ Evaluation complete"

eval-sample: ## Run evaluation on first 10 questions (quick test)
	@echo "Running evaluation on first 10 questions..."
	docker compose exec app python -m ragagent.eval.runner eval/questions.jsonl --limit 10 --verbose

test: ## Run test suite
	@echo "Running test suite..."
	docker compose exec app pytest -q
	@echo "✓ Tests complete"

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	docker compose exec app pytest --cov=ragagent --cov-report=term-missing

# Development Commands

dev: ## Run app locally with hot reload (requires: make dev-services first)
	uvicorn ragagent.app:app --reload --port 8000

dev-services: ## Start only external services (for local development)
	docker compose up -d qdrant neo4j embeddings qdrant-init neo4j-init
	@echo "✓ External services started. Run 'make dev' in another terminal"

install: ## Install Python dependencies locally
	uv pip install -e .
	@echo "✓ Dependencies installed"

build: ## Build Docker images without starting services
	docker compose build

# Utility Commands

shell-worker: ## Open bash shell in worker container
	docker compose exec worker bash

shell-app: ## Open bash shell in app container
	docker compose exec app bash

shell-neo4j: ## Open cypher-shell in neo4j container
	docker compose exec neo4j cypher-shell -u neo4j -p neo4j

# Backup Commands

backup-qdrant: ## Create Qdrant snapshot backup
	@mkdir -p backups
	@echo "Creating Qdrant snapshot..."
	@SNAPSHOT=$$(curl -s -X POST 'http://localhost:6333/collections/documents/snapshots' | jq -r '.result.name'); \
	if [ "$$SNAPSHOT" = "null" ] || [ -z "$$SNAPSHOT" ]; then \
		echo "✗ Failed to create snapshot"; \
		exit 1; \
	fi; \
	echo "Snapshot created: $$SNAPSHOT"; \
	sleep 2; \
	docker compose cp qdrant:/qdrant/storage/documents/snapshots/$$SNAPSHOT ./backups/; \
	echo "✓ Backup saved to: backups/$$SNAPSHOT"

backup-neo4j: ## Create Neo4j backup (requires APOC)
	@mkdir -p backups
	@echo "Creating Neo4j backup..."
	@BACKUP_FILE="backup-$$(date +%Y%m%d-%H%M%S).cypher"; \
	docker compose exec -T neo4j cypher-shell -u neo4j -p neo4j \
		"CALL apoc.export.cypher.all('/tmp/$$BACKUP_FILE', {format:'cypher-shell'});" && \
	docker compose cp neo4j:/tmp/$$BACKUP_FILE ./backups/$$BACKUP_FILE && \
	echo "✓ Backup saved to: backups/$$BACKUP_FILE"

backup-all: backup-qdrant backup-neo4j ## Create backups of both Qdrant and Neo4j

restore-qdrant: ## Restore Qdrant from snapshot (usage: make restore-qdrant SNAPSHOT=filename.snapshot)
	@if [ -z "$(SNAPSHOT)" ]; then \
		echo "Error: SNAPSHOT parameter required"; \
		echo "Usage: make restore-qdrant SNAPSHOT=documents-2025-11-20-12-30-45.snapshot"; \
		exit 1; \
	fi
	@if [ ! -f "backups/$(SNAPSHOT)" ]; then \
		echo "Error: backups/$(SNAPSHOT) not found"; \
		exit 1; \
	fi
	@echo "Restoring Qdrant from $(SNAPSHOT)..."
	docker compose cp backups/$(SNAPSHOT) qdrant:/qdrant/storage/snapshots/$(SNAPSHOT)
	curl -X PUT 'http://localhost:6333/collections/documents/snapshots/$(SNAPSHOT)' \
		-H 'Content-Type: application/json' \
		-d '{"location":"file:///qdrant/storage/snapshots/$(SNAPSHOT)"}'
	@echo "✓ Restore complete"

restore-neo4j: ## Restore Neo4j from backup (usage: make restore-neo4j BACKUP=backup-20251120-123045.cypher)
	@if [ -z "$(BACKUP)" ]; then \
		echo "Error: BACKUP parameter required"; \
		echo "Usage: make restore-neo4j BACKUP=backup-20251120-123045.cypher"; \
		exit 1; \
	fi
	@if [ ! -f "backups/$(BACKUP)" ]; then \
		echo "Error: backups/$(BACKUP) not found"; \
		exit 1; \
	fi
	@echo "⚠️  This will clear existing data. Continue? [y/N]"; \
	read -r confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "Clearing Neo4j data..."; \
		docker compose exec -T neo4j cypher-shell -u neo4j -p neo4j "MATCH (n) DETACH DELETE n;" && \
		echo "Copying backup to container..."; \
		docker compose cp backups/$(BACKUP) neo4j:/tmp/$(BACKUP) && \
		echo "Restoring data..."; \
		docker compose exec -T neo4j cypher-shell -u neo4j -p neo4j < backups/$(BACKUP) && \
		echo "✓ Restore complete"; \
	else \
		echo "Cancelled"; \
	fi

# Query Commands

query: ## Send a test query (usage: make query Q="What are the main findings?")
	@if [ -z "$(Q)" ]; then \
		echo "Error: Q parameter required"; \
		echo "Usage: make query Q=\"What are the main findings?\""; \
		exit 1; \
	fi
	@echo "Querying: $(Q)"
	@curl -s -X POST http://localhost:8000/chat \
		-H "Content-Type: application/json" \
		-d "{\"question\": \"$(Q)\", \"top_k\": 8}" | jq '.'

# Stats Commands

stats: ## Show database statistics
	@echo "=== Database Statistics ==="
	@echo ""
	@echo "Qdrant Collection:"
	@curl -s http://localhost:6333/collections/documents | jq '.result | {status, points_count, vectors_count}'
	@echo ""
	@echo "Neo4j Nodes:"
	@docker compose exec -T neo4j cypher-shell -u neo4j -p neo4j \
		"MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC;" 2>/dev/null | grep -v "^$$"
	@echo ""

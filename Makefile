.PHONY: help sync fmt lint test test-fast \
	services-up services-down services-logs services-reset \
	setup-db ingest update query tui review backfill-relationships \
	clean-normalization clean-curation clean-logs clean-cache clean-all reset-all

SHELL := /bin/bash

# Prefer docker-compose if installed; override with `make DC="docker compose" ...`
DC ?= docker-compose
UV ?= uv

# Common paths
NORM_TABLE ?= data/normalization/normalization_table.json
UNDO_STACK ?= data/curation/undo_stack.json

# Defaults for scripts
INPUT ?= data/raw
BATCH_SIZE ?= 10

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install/sync dependencies (uv)
	$(UV) sync

fmt: ## Format code (black)
	$(UV) run --offline black src/ tests/ scripts/

lint: ## Lint code (ruff)
	$(UV) run --offline ruff check src/ tests/ scripts/

test: ## Run test suite
	$(UV) run --offline pytest

test-fast: ## Run a fast unit-test subset (no coverage config changes)
	$(UV) run --offline pytest -q

services-up: ## Start Neo4j + Qdrant
	$(DC) up -d neo4j qdrant

services-down: ## Stop Neo4j + Qdrant (keeps volumes)
	$(DC) down

services-logs: ## Tail service logs
	$(DC) logs -f --tail=200

services-reset: ## Stop services and delete volumes (DESTRUCTIVE)
	$(DC) down -v --remove-orphans

setup-db: ## Initialize Neo4j/Qdrant schemas
	$(UV) run --offline python scripts/setup_databases.py

ingest: ## Ingest PDFs from data/raw (override INPUT/BATCH_SIZE)
	$(UV) run --offline python scripts/ingest_documents.py --input $(INPUT) --batch-size $(BATCH_SIZE)

update: ## Update documents (see scripts/update_documents.py)
	$(UV) run --offline python scripts/update_documents.py

query: ## Run query CLI (see scripts/query_system.py)
	$(UV) run --offline python scripts/query_system.py

tui: ## Launch interactive curation TUI
	$(UV) run --offline python scripts/review_entities_interactive.py

review: ## Launch non-interactive review CLI
	$(UV) run --offline python scripts/review_entities.py

backfill-relationships: ## Promote RelationshipCandidates into graph edges
	$(UV) run --offline python scripts/backfill_relationships.py

backfill-embeddings: ## Backfill Qdrant entity embeddings from Neo4j entities
	$(UV) run --offline python scripts/backfill_entity_embeddings.py

cleanup-relationships: ## Bulk promote/reject pending relationships (see scripts/cleanup_relationships.py)
	$(UV) run --offline python scripts/cleanup_relationships.py --help

reset-databases: ## Delete ALL data from Neo4j and Qdrant (prompts for confirmation)
	$(UV) run --offline python scripts/reset_databases.py

clean-normalization: ## Delete normalization table JSON (recommended when resetting Neo4j)
	rm -f "$(NORM_TABLE)"

clean-curation: ## Delete curation undo stack
	rm -f "$(UNDO_STACK)"

clean-logs: ## Remove local logs (non-destructive to data/)
	rm -rf logs/*

clean-cache: ## Remove local Python/tool caches (non-destructive)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov .uv_cache

clean-all: clean-normalization clean-curation clean-logs clean-cache ## Clean local artifacts (DESTRUCTIVE)
	@true

reset-all: services-reset clean-all services-up sync setup-db  ## Full reset: wipe volumes + local artifacts, then start fresh (DESTRUCTIVE)
	@echo "Reset complete. Next: re-ingest with \`make ingest INPUT=data/raw BATCH_SIZE=10\`"

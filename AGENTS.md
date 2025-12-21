# Repository Guidelines

## Project Structure & Modules
- Core Python lives in `src/` (ingestion, extraction, normalization, storage, retrieval, curation, pipeline, utils). Add new features inside these domains instead of new top-level packages.
- CLI helpers are in `scripts/` (e.g., `scripts/setup_databases.py`, `scripts/ingest_documents.py`); extend them rather than duplicating logic.
- Configuration is in `config/` (`config.yaml`, cleaning/prompt files); data inputs/outputs stay under `data/` (`raw/`, `processed/`, `entities/`, `normalization/`). Tests live in `tests/` mirroring `src/`.
- Architecture and planning notes belong in `plans/`; keep design intent there when changing flows or schemas.

## Build, Test, and Development Commands
- Setup: `uv sync` (uses `uv.lock`), then run tools via `uv run ...`.
- Services for integration work: `docker-compose up -d neo4j qdrant`.
- Test suite: `uv run pytest` (default `-v --cov=src --cov-report=term-missing --cov-report=html` → `htmlcov/`).
- Code quality: `uv run black src/ tests/`, `uv run ruff check src/ tests/`.
- Operational scripts: `uv run python scripts/setup_databases.py` to initialize stores; `uv run python scripts/ingest_documents.py --input data/raw/ --batch-size 10` to ingest PDFs.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indents, max line length 100 (Black/Ruff). Keep imports sorted by Ruff.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Type hints expected (`disallow_untyped_defs=true`); prefer `pydantic` models for structured config/entities and centralize config access in `src/utils`.
- Keep side effects thin and injectable; isolate retrieval and ingestion logic to enable mocking in tests.
- Use pydantic models over dataclasses

## Testing Guidelines
- Place tests in `tests/`; filenames `test_*.py`, functions `test_*`. Mirror `src/` layout for new modules.
- Mock Neo4j/Qdrant/LLMs in unit tests; run live-service checks only when explicitly intended.
- Target coverage on touched code paths; add regression tests for bug fixes or changes in chunking, extraction, or query pipelines.

## Commit & Pull Request Guidelines
- Use short, imperative commit messages (e.g., `add qdrant schema helper`, `fix ingestion batch logging`); keep each commit focused.
- Before PRs: run `pytest`, `black`, `ruff`; summarize change scope, risk/impact, and config updates (`config/*.yaml`) in the description.
- Link issues/tasks; attach screenshots or log snippets when CLI behavior changes. Call out required migrations or data backfills.

## Security & Configuration Tips
- Never commit secrets; rely on `.env` and `config/config.yaml` overrides. Share example values, not real keys.
- Keep large artifacts under `data/`; avoid committing raw PDFs, generated embeddings, or logs in `logs/`.
- When altering schemas or collection layouts, document the change in `plans/` and update setup scripts accordingly.

# Plan: Refactoring & Semantic Improvements

This plan outlines critical improvements to the Graph RAG system, focusing on fixing semantic entity resolution, refactoring the ingestion pipeline, and ensuring robust incremental updates.

**Priority Order:**
1.  **Semantic Entity Resolution** (Critical - functionality gap)
2.  **Ingestion Pipeline Refactoring** (High - architectural debt)
3.  **Robust Incremental Updates** (Medium - data integrity)

---

## 1. Semantic Entity Resolution
**Goal:** Enable vector-based entity linking to bridge the gap between natural language queries and graph nodes.

### Task 1.1: Enable Entity Embedding on Approval
Modify `EntityCurationService` to generate embeddings for entities upon approval/creation and store them in Qdrant.

-   **File:** `src/curation/entity_approval.py`
-   **Changes:**
    -   Inject `EmbeddingGenerator` and `QdrantManager` into `EntityCurationService`.
    -   In `approve_candidate`, `create_entity`, `merge_candidates`, and `merge_candidate_into_entity`:
        -   Generate embedding for the entity (using canonical name + description + aliases).
        -   Call `qdrant_manager.upsert_entities`.
    -   *Note:* Ensure this is async or fast enough not to block the TUI.

### Task 1.2: Hybrid Entity Resolution in Retrieval
Update `GraphRetriever` to use vector search for resolving entities when exact matches fail.

-   **File:** `src/retrieval/graph_retriever.py`
-   **Changes:**
    -   Inject `QdrantManager` into `GraphRetriever`.
    -   In `_resolve_entities`:
        -   Keep existing logic (Exact Match -> Full-text Search).
        -   If confidence is low or no matches found:
            -   Generate embedding for the mention text.
            -   Call `qdrant_manager.search_entities`.
            -   Add high-scoring vector matches to the `resolved` list with a lower confidence than exact matches (e.g., 0.85).

### Task 1.3: Backfill Script
Create a script to embed all existing approved entities in Neo4j and push them to Qdrant.

-   **Script:** `scripts/backfill_entity_embeddings.py`
-   **Logic:**
    -   Fetch all `APPROVED` entities from Neo4j.
    -   Generate embeddings batch-wise.
    -   Upsert to Qdrant.

---

## 2. Ingestion Pipeline Refactoring
**Goal:** Decompose the monolithic `IngestionPipeline` into a modular, testable pipeline of stages.

### Task 2.1: Define Pipeline Infrastructure
Create the base abstractions for the pipeline.

-   **Files:** `src/pipeline/base.py`
-   **Classes:**
    -   `PipelineContext`: Holds the document, metadata, intermediate artifacts (chunks, entities).
    -   `PipelineStage`: Abstract base class with `run(context: PipelineContext) -> PipelineContext`.
    -   `Pipeline`: Orchestrator that executes a list of stages.

### Task 2.2: Implement Concrete Stages
Move logic from `IngestionPipeline` into discrete stages.

-   **Directory:** `src/pipeline/stages/`
-   **Stages:**
    -   `PDFParsingStage`: Wraps `Docling` / `PDFParser`.
    -   `TextCleaningStage`: Wraps `TextCleaner`.
    -   `ChunkingStage`: Wraps `HierarchicalChunker`.
    -   `EmbeddingStage`: Wraps `EmbeddingGenerator` for chunks.
    -   `ExtractionStage`: Wraps `LLMExtractor` / `SpacyExtractor`.
    -   `GraphStorageStage`: Writes to Neo4j.
    -   `VectorStorageStage`: Writes to Qdrant.

### Task 2.3: Update IngestionPipeline
Refactor `IngestionPipeline` to be a thin wrapper around the new `Pipeline` class to maintain backward compatibility with CLI scripts.

-   **File:** `src/pipeline/ingestion_pipeline.py`
-   **Changes:**
    -   Initialize the `Pipeline` with the standard set of stages in `__init__`.
    -   `process_document` simply calls `pipeline.run()`.

---

## 3. Robust Incremental Updates [DONE]
**Goal:** Ensure document identity allows for file renames without data duplication or loss.

### Task 3.1: Content-Based Hashing [DONE]
Switch from purely file-path based identity to content-based identity.

-   **File:** `src/pipeline/update_pipeline.py`
-   **Changes:**
    -   Refine `_compute_checksum` to be the "Identity" of the file.
    -   In `detect_changes`:
        -   First pass: Map all on-disk files by `checksum`.
        -   Second pass: Map all DB documents by `checksum`.
        -   Match logic:
            -   If `checksum` matches but `path` is different -> **RENAME** (Update `file_path` in DB, do not re-ingest).
            -   If `path` matches but `checksum` is different -> **MODIFIED** (Re-ingest).
            -   If neither matches -> **NEW**.
            -   If DB doc has no checksum match on disk -> **DELETED**.

### Task 3.2: Verify Update Logic [DONE]
Add a test case specifically for the "Rename" scenario to ensure it updates the path metadata without triggering a costly re-extraction.

---

## Execution Plan

### Step 1: Fix Semantic Resolution (Immediate)
1.  Implement Task 1.1 (`EntityCurationService` updates).
2.  Implement Task 1.2 (`GraphRetriever` updates).
3.  Run Task 1.3 (Backfill) to align dev environments.
4.  Verify with a query like "What is the CPU?" when only "Central Processing Unit" exists in the graph.

### Step 2: Refactor Ingestion (Next)
1.  Create `src/pipeline/base.py`.
2.  Create `src/pipeline/stages/`.
3.  Port logic stage-by-stage (tests can be migrated iteratively).
4.  Swap `IngestionPipeline` implementation.

### Step 3: Hardening Updates (Final)
1.  Update `detect_changes` logic for Rename detection.
2.  Add tests.

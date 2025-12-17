# Phase 1: Foundation (Weeks 1-2)

## Overview

This phase establishes the foundational infrastructure for the Graph RAG system, including development environment setup, database schemas, PDF parsing, text processing, and the basic ingestion pipeline.

**Timeline:** Weeks 1-2  
**Architecture Reference:** [`graph-rag-architecture.md`](graph-rag-architecture.md)
**Status:** Phase complete ✅

## Completion Checklist
- [x] 1.1 Development Environment Setup
- [x] 1.2 Configuration Management System
- [x] 1.3 Database Schema Implementation - Neo4j
- [x] 1.4 Database Schema Implementation - Qdrant
- [x] 1.5 PDF Parsing with Docling
- [x] 1.6 Text Cleaning and Preprocessing
- [x] 1.7 Optional LLM Text Rewriting
- [x] 1.8 Hierarchical Chunking Implementation
- [x] 1.9 Embedding Generation
- [x] 1.10 Basic Ingestion Pipeline

---

## Task 1.1: Development Environment Setup
**Priority:** Critical  
**Dependencies:** None
**Status:** ✅ Completed

**Description:**
Set up the complete development environment including databases, LLM infrastructure, and project structure.

**Steps:**
1. Create Docker Compose configuration for Neo4j and Qdrant
2. (Skip as already installed) Install and configure Ollama with Llama 3.1 model
3. Set up Python virtual environment with required dependencies
4. Create project directory structure as specified in architecture
5. Initialize Git repository with appropriate `.gitignore`
6. Create `.env.example` with all required environment variables

**Deliverables:**
- [`docker-compose.yml`] with Neo4j and Qdrant services
- [`pyproject.toml`] with all dependencies listed
- Complete project directory structure
- [`.env.example`] template file
- [`README.md`] with setup instructions

**Acceptance Criteria:**
- Neo4j accessible at localhost:7474 with authentication
- Qdrant accessible at localhost:6333
- Ollama running and responding to test prompts
- All dependencies installable via pip
- Project structure matches architecture document

---

## Task 1.2: Configuration Management System
**Priority:** High  
**Dependencies:** Task 1.1
**Status:** ✅ Completed

**Description:**
Implement centralized configuration management using Pydantic for validation and environment variables.

**Steps:**
1. Create [`src/utils/config.py`] with Pydantic models for configuration
2. Implement config loader that reads from YAML and environment variables
3. Create [`config/config.yaml`] with default values
4. Add configuration validation on startup
5. Implement configuration override mechanism for testing

**Deliverables:**
- [`src/utils/config.py`] with `Config` class
- [`config/config.yaml`] with all configuration sections
- Unit tests for configuration loading and validation

**Acceptance Criteria:**
- Configuration loads from YAML file
- Environment variables override YAML values
- Invalid configurations raise descriptive errors
- All sections from architecture are represented

---

## Task 1.3: Database Schema Implementation - Neo4j
**Priority:** Critical  
**Dependencies:** Task 1.1
**Status:** ✅ Completed

**Description:**
Create Neo4j graph database schema including node labels, relationship types, properties, and indexes.

**Steps:**
1. Create [`src/storage/schemas.py`] with Pydantic models for all entity types
2. Create [`scripts/setup_databases.py`] script for database initialization
3. Implement Cypher queries to create constraints and indexes
4. Define node labels for all entity types (System, Subsystem, Component, etc.)
5. Define relationship types with properties
6. Create utility functions for schema validation

**Deliverables:**
- [`src/storage/schemas.py`] with data models
- [`scripts/setup_databases.py`] initialization script
- [`src/storage/neo4j_manager.py`] with connection and basic CRUD operations
- Documentation of schema in docstrings

**Acceptance Criteria:**
- All node labels created with required properties
- All relationship types defined
- Indexes created on: [`canonical_name`], [`entity_type`], [`id`]
- Full-text search index created for entities
- Script can be run multiple times idempotently

---

## Task 1.4: Database Schema Implementation - Qdrant
**Priority:** Critical  
**Dependencies:** Task 1.1
**Status:** ✅ Completed

**Description:**
Set up Qdrant vector database collections for chunks and entities with appropriate configurations.

**Steps:**
1. Create [`src/storage/qdrant_manager.py`] with Qdrant client wrapper
2. Implement collection creation for [`document_chunks`] and [`entities`]
3. Define payload schemas with proper indexing
4. Configure HNSW parameters for optimal performance
5. Implement vector upsert and search operations
6. Add health check and collection info methods

**Deliverables:**
- [`src/storage/qdrant_manager.py`] with `QdrantManager` class
- Collection initialization in [`scripts/setup_databases.py`]
- Payload schema definitions
- Basic search and upsert methods

**Acceptance Criteria:**
- Two collections created: [`document_chunks`] and [`entities`]
- Proper vector dimensions configured (768 for BGE embeddings)
- Payload indexes on: [`document_id`], [`entity_ids`], [`entity_type`]
- Cosine similarity as distance metric
- Connection pooling implemented

---

## Task 1.5: PDF Parsing with Docling
**Priority:** High  
**Dependencies:** Task 1.1, Task 1.2
**Status:** ✅ Completed

**Description:**
Implement PDF parsing using Docling with OCR support, extracting text, structure, and metadata.

**Steps:**
1. Create [`src/ingestion/pdf_parser.py`] with `PDFParser` class
2. Implement Docling integration with OCR enabled
3. Extract document structure (sections, subsections, paragraphs)
4. Extract metadata (title, date, version, page count)
5. Handle tables and figures separately
6. Implement error handling for corrupt/malformed PDFs
7. Add progress logging for large documents

**Deliverables:**
- [`src/ingestion/pdf_parser.py`] with parsing logic
- [`src/ingestion/metadata_extractor.py`] for metadata extraction
- Test suite with sample PDFs
- Parsed document data structure (JSON schema)

**Acceptance Criteria:**
- Parses PDFs with and without OCR
- Preserves document hierarchy and structure
- Extracts metadata accurately
- Handles multi-column layouts
- Returns structured output matching schema
- Graceful error handling with informative messages

---

## Task 1.6: Text Cleaning and Preprocessing
**Priority:** High  
**Dependencies:** Task 1.5
**Status:** ✅ Completed

**Description:**
Implement text cleaning module to remove noise, headers, footers, and other unwanted content using regex patterns.

**Steps:**
1. Create [`src/ingestion/text_cleaner.py`] with `TextCleaner` class
2. Implement regex-based pattern matching for common noise
3. Create pattern categories: headers, footers, page_numbers, watermarks, OCR errors
4. Implement whitespace normalization
5. Add preservation logic for code blocks and equations
6. Create [`config/cleaning_patterns.yaml`] with default patterns
7. Add pattern enable/disable configuration
8. Implement custom pattern support

**Deliverables:**
- [`src/ingestion/text_cleaner.py`] with cleaning logic
- [`config/cleaning_patterns.yaml`] with regex patterns
- Configuration integration in [`config/config.yaml`]
- Unit tests with various noise scenarios

**Acceptance Criteria:**
- Removes headers and footers accurately
- Strips page numbers and metadata
- Preserves technical content (code, equations)
- Configurable patterns per document type
- Fast processing (>10 pages/second)
- Handles edge cases without data loss
- Whitelist patterns protect important content

---

## Task 1.7: Optional LLM Text Rewriting
**Priority:** Medium  
**Dependencies:** Task 1.6, Task 2.2
**Status:** ✅ Completed

**Description:**
Implement optional LLM-based text rewriting to improve readability and parsing while preserving information.

**Steps:**
1. Create [`src/ingestion/text_rewriter.py`] with `TextRewriter` class
2. Implement separate LLM integration for text rewriting (can use different provider/model)
3. Add configuration for rewriting-specific LLM provider and model
4. Create rewriting prompt template in [`config/rewriting_prompt.yaml`]
5. Implement chunk-level rewriting (section/subsection)
6. Add preservation checks for technical terms and numbers
7. Implement original text preservation for comparison
8. Add quality validation (ensure information preserved)
9. Create enable/disable configuration
10. Add selective document targeting

**Deliverables:**
- [`src/ingestion/text_rewriter.py`] with rewriting logic
- [`config/rewriting_prompt.yaml`] with prompt template
- Configuration for enabling/disabling feature
- Comparison report showing original vs. rewritten
- Quality validation checks

**Acceptance Criteria:**
- Preserves all technical terms and numbers
- Maintains relationships between concepts
- Improves readability demonstrably
- Optional feature (disabled by default)
- Uses separate LLM configuration (provider and model)
- Can use different LLM than extraction pipeline
- Processing time acceptable (<10s per section)
- Can be applied selectively to documents
- Original text preserved for audit

---

## Task 1.8: Hierarchical Chunking Implementation
**Priority:** High  
**Dependencies:** Task 1.7
**Status:** ✅ Completed

**Description:**
Implement hierarchical document chunking strategy with four levels: document, section, subsection, paragraph.

**Steps:**
1. Create [`src/ingestion/chunker.py`] with `HierarchicalChunker` class
2. Implement level 1 (document) chunking with full metadata
3. Implement level 2 (section) chunking based on document structure
4. Implement level 3 (subsection) chunking
5. Implement level 4 (paragraph) chunking with token limits
6. Maintain parent-child relationships between chunks
7. Add hierarchy path tracking (e.g., "1.2.3")
8. Implement token counting and size limits

**Deliverables:**
- [`src/ingestion/chunker.py`] with chunking logic
- Chunk data structure with all required fields
- Test suite with various document structures
- Configuration for chunk sizes and overlap

**Acceptance Criteria:**
- Generates chunks at all four hierarchy levels
- Parent-child relationships correctly maintained
- Each chunk has unique ID and references parent
- Token counts accurate for each chunk
- Configurable chunk sizes and overlap
- Preserves page number references

---

## Task 1.9: Embedding Generation
**Priority:** High  
**Dependencies:** Task 1.1, Task 1.2
**Status:** ✅ Completed

**Description:**
Implement embedding generation for document chunks and entities using FastEmbed or Sentence Transformers.

**Steps:**
1. Create [`src/utils/embeddings.py`] with `EmbeddingGenerator` class
2. Integrate FastEmbed or Sentence Transformers
3. Implement batched embedding generation for efficiency
4. Add caching mechanism for embeddings
5. Handle long text truncation
6. Implement retry logic for API-based embeddings (optional)
7. Add embedding normalization if needed

**Deliverables:**
- [`src/utils/embeddings.py`] with embedding generation
- Support for multiple embedding models via configuration
- Batch processing with configurable batch size
- Error handling and logging

**Acceptance Criteria:**
- Generates 768-dimensional embeddings (BGE model)
- Batched processing for efficiency
- Consistent embeddings for same input
- Handles text longer than model max length
- Configurable model selection
- Performance: >100 embeddings/second

---

## Task 1.10: Basic Ingestion Pipeline
**Priority:** High  
**Dependencies:** Task 1.3, Task 1.4, Task 1.5, Task 1.6, Task 1.7, Task 1.8, Task 1.9
**Status:** ✅ Completed

**Description:**
Create end-to-end ingestion pipeline that processes PDFs and stores chunks in both databases.

**Steps:**
1. Create [`src/pipeline/ingestion_pipeline.py`] with `IngestionPipeline` class
2. Implement pipeline orchestration: parse → chunk → embed → store
3. Add document tracking and status management
4. Implement batch processing for multiple documents
5. Add progress tracking and logging
6. Store chunks in Qdrant with embeddings
7. Store document metadata and chunk references in Neo4j
8. Add error handling and rollback mechanism

**Deliverables:**
- [`src/pipeline/ingestion_pipeline.py`] with complete pipeline
- [`scripts/ingest_documents.py`] CLI script for batch ingestion
- Document status tracking in both databases
- Comprehensive logging

**Acceptance Criteria:**
- Processes single PDF end-to-end successfully
- Batches multiple PDFs efficiently
- Stores all chunks in Qdrant with embeddings
- Creates document and chunk nodes in Neo4j
- Handles failures gracefully with rollback
- Provides progress feedback
- Can resume interrupted ingestion

---

## Phase 1 Summary

**Key Deliverables:**
- Complete development environment
- Neo4j and Qdrant schemas implemented
- PDF parsing and text processing pipeline
- Hierarchical chunking system
- Embedding generation
- End-to-end ingestion pipeline

**Success Metrics:**
- Environment setup complete and documented
- Database schemas validated
- Can process PDFs with structure preservation
- Chunks stored with embeddings in vector DB
- Document graph created in Neo4j

**Next Phase:** Phase 2 - Entity Extraction

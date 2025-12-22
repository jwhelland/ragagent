# Phase 2: Entity Extraction (Weeks 3-4)

## Overview

This phase implements the entity and relationship extraction system using both spaCy NER and LLM-based extraction, merging results, and storing candidates for curation.

**Timeline:** Weeks 3-4  
**Architecture Reference:** [`graph-rag-architecture.md`](graph-rag-architecture.md)  
**Previous Phase:** [Phase 1 - Foundation](phase-1-foundation.md)

---

## Task 2.1: spaCy NER Pipeline Setup
**Priority:** High  
**Dependencies:** Task 1.10

**Description:**
Set up spaCy NER pipeline with domain-specific patterns for technical terminology.

**Steps:**
1. Create [`src/extraction/spacy_extractor.py`] with `SpacyExtractor` class
2. Load transformer-based English model (`en_core_web_lg`)
3. Create custom entity patterns for technical domain
4. Add custom pipeline component for domain terms
5. Implement entity extraction from chunks
6. Add confidence scoring based on context
7. Store patterns in [`config/entity_patterns.jsonl`]

**Deliverables:**
- [`src/extraction/spacy_extractor.py`] with extraction logic
- [`config/entity_patterns.jsonl`] with domain patterns
- Entity extraction returning structured format
- Confidence scoring implementation

**Acceptance Criteria:**
- Extracts standard entities (ORG, PRODUCT, etc.)
- Recognizes domain-specific terms
- Confidence scores between 0-1
- Fast processing (>1000 tokens/second)
- Configurable pattern matching rules
- Returns entities with spans and context

---

## Task 2.2: LLM Integration (Anthropic and OpenAI)
**Priority:** High  
**Dependencies:** Task 1.1, Task 1.2

**Description:**
Implement unified LLM integration that works with either Anthropic or OpenAI API based on configuration.

**Steps:**
1. Create [`src/extraction/llm_extractor.py`] with `LLMExtractor` class
3. Implement OpenAI/Anthropic client integration
4. Add configuration-based provider selection
5. Create extraction prompts for entities
6. Create extraction prompts for relationships
7. Implement structured output parsing (JSON)
8. Add retry logic for failed requests
9. Implement timeout handling
10. Add prompt templates in [`config/extraction_prompts.yaml`]

**Deliverables:**
- [`src/extraction/llm_extractor.py`] with OpenAI or Anthropic integration
- [`config/extraction_prompts.yaml`] with prompt templates
- Structured output parsing for entities/relationships
- Configuration-based provider selection
- Error handling and retry logic

**Acceptance Criteria:**
- Connects to Anthropic or OpenAI based on configuration
- Provider switchable via config file
- Extracts entities with types and descriptions
- Extracts relationships with source/target/type
- Returns structured JSON output
- Handles timeouts and retries
- Configurable temperature and max tokens
- Same interface regardless of provider
- Average processing time <5 seconds per chunk for Anthropic, <2 seconds for OpenAI

---

## Task 2.3: Entity and Relationship Prompt Engineering
**Priority:** High  
**Dependencies:** Task 2.2

**Description:**
Design and test prompts for accurate entity and relationship extraction from technical documents.

**Steps:**
1. Create entity extraction prompt with examples
2. Create relationship extraction prompt with examples
3. Define output JSON schema for entities
4. Define output JSON schema for relationships
5. Test prompts on sample documents
6. Iterate based on extraction quality
7. Document prompt design decisions
8. Store final prompts in [`config/extraction_prompts.yaml`]

**Deliverables:**
- [`config/extraction_prompts.yaml`] with optimized prompts
- JSON schemas for entity and relationship output
- Test results showing extraction quality
- Documentation of prompt engineering process

**Acceptance Criteria:**
- Prompts extract relevant entities consistently
- Prompts identify correct relationship types
- Output follows defined JSON schema
- Few-shot examples improve accuracy
- Works well with both API LLMs
- Documented prompt versions and changes

---

## Task 2.4: Entity Merger
**Priority:** High  
**Dependencies:** Task 2.1, Task 2.2

**Description:**
Merge entity candidates from spaCy and LLM extractions, resolving conflicts and combining confidence scores.

**Implementation Notes:**
- Matching is chunk-scoped and based on normalized surface forms plus LLM aliases.
- Type conflicts are resolved via weighted confidence voting across extractors; non-winning labels are retained as `conflicting_types`.
- Confidence uses probabilistic OR (`1 - Π(1 - conf_i)`) with a small cross-source confirmation bonus.

**Steps:**
1. Create [`src/extraction/entity_merger.py`] with merger logic
2. Implement entity matching between spaCy and LLM results
3. Implement conflict resolution (type mismatches)
4. Combine confidence scores from multiple sources
5. Deduplicate within same chunk
6. Preserve source provenance (which extractor found it)
7. Create merged entity candidate structure

**Deliverables:**
- [`src/extraction/entity_merger.py`] with merging logic
- Conflict resolution strategy documented
- Confidence score combination formula
- Test suite with edge cases

**Acceptance Criteria:**
- Merges entities found by both extractors
- Resolves type conflicts intelligently
- Combined confidence higher for confirmed entities
- Preserves extraction source metadata
- Handles cases where extractors disagree
- No duplicate entities in output

---

## Task 2.5: Entity Candidate Database
**Priority:** High  
**Dependencies:** Task 2.4

**Description:**
Create storage and management system for entity candidates before curation.

**Steps:**
1. Design entity candidate schema in Neo4j
2. Implement candidate storage in [`src/storage/neo4j_manager.py`]
3. Create candidate query methods (by confidence, type, frequency)
4. Implement candidate statistics calculation
5. Add candidate status tracking (pending, approved, rejected)
6. Create merge suggestion identification
7. Store extraction provenance

**Deliverables:**
- Candidate node type in Neo4j schema
- CRUD operations for candidates in [`src/storage/neo4j_manager.py`]
- Query methods for candidate retrieval
- Statistics calculation methods

**Acceptance Criteria:**
- Candidates stored separately from production entities
- Queryable by confidence, type, status
- Tracks mention count and documents
- Identifies similar candidates for merging
- Maintains extraction history
- Supports bulk operations

---

## Task 2.6: Extraction Pipeline Orchestration
**Priority:** High  
**Dependencies:** Task 2.1, Task 2.2, Task 2.4, Task 2.5

**Description:**
Orchestrate the complete extraction pipeline: classify → extract (spaCy + LLM) → merge → store candidates.

**Steps:**
1. Extend [`src/pipeline/ingestion_pipeline.py`] with extraction phase
2. Parallel execution of spaCy and LLM extraction
3. Entity merging after both extractors complete
4. Store entity and relationship candidates
5. Add extraction metrics and logging
6. Implement error handling per chunk
7. Create extraction summary report

**Deliverables:**
- Complete extraction pipeline in [`src/pipeline/ingestion_pipeline.py`]
- Extraction metrics and reporting
- Error handling and retry logic
- Progress tracking

**Acceptance Criteria:**
- Processes all chunks through extraction
- Uses configured LLM provider (Anthropic or OpenAI)
- Merges results from both extractors
- Stores all candidates with metadata
- Reports extraction statistics
- Handles extraction failures gracefully
- Can process documents in parallel

---

## Phase 2 Summary

**Key Deliverables:**
- spaCy NER pipeline with domain patterns
- Flexible LLM integration (Anthropic/OpenAI)
- Optimized extraction prompts
- Entity merger combining multiple sources
- Entity candidate database
- Complete extraction pipeline

**Success Metrics:**
- Extracts entities from all document chunks
- Hybrid approach (spaCy + LLM) improves coverage
- Entity candidates stored with provenance
- Extraction quality meets >80% accuracy target
- Pipeline processes documents efficiently

**Next Phase:** [Phase 3 - Normalization & Curation](phase-3-normalization-curation.md)

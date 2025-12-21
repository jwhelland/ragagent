# Phase 3: Normalization & Curation (Weeks 5-6)

## Overview

This phase implements entity normalization, deduplication, and a curation system for reviewing and approving entity candidates. It includes string normalization, fuzzy matching, acronym resolution, and a CLI-based review interface.

**Timeline:** Weeks 5-6  
**Architecture Reference:** [`graph-rag-architecture.md`](graph-rag-architecture.md)  
**Previous Phase:** [Phase 2 - Entity Extraction](phase-2-entity-extraction.md)

---

## Task 3.1: String Normalization
**Priority:** High  
**Dependencies:** Task 2.5

**Description:**
Implement string normalization for entity canonical names and text matching.

**Steps:**
1. Create [`src/normalization/string_normalizer.py`] with normalizer
2. Implement lowercase conversion with special handling for technical terms
3. Remove extra whitespace and normalize punctuation
4. Handle special characters in technical terminology
5. Create normalization rules configuration
6. Add reversibility for display purposes
7. Document normalization decisions

**Deliverables:**
- [`src/normalization/string_normalizer.py`] with normalization
- Configuration for normalization rules
- Unit tests with edge cases
- Documentation of normalization strategy

**Acceptance Criteria:**
- Normalizes entity names consistently
- Preserves technical meaning (e.g., "C++" not corrupted)
- Handles acronyms appropriately
- Configurable normalization rules
- Fast processing (>10000 strings/second)
- Reversible for display purposes

---

## Task 3.2: Fuzzy String Matching
**Priority:** High  
**Dependencies:** Task 3.1

**Description:**
Implement fuzzy matching to identify entity variants and typos using RapidFuzz.

**Steps:**
1. Create [`src/normalization/fuzzy_matcher.py`] with matching logic
2. Integrate RapidFuzz library
3. Implement similarity scoring with configurable threshold
4. Add batch matching for efficiency
5. Handle short strings (higher threshold needed)
6. Create match candidate pairs
7. Add confidence scoring based on match quality

**Deliverables:**
- [`src/normalization/fuzzy_matcher.py`] with fuzzy matching
- Configurable similarity thresholds
- Match candidate generation
- Performance optimization for large entity sets

**Acceptance Criteria:**
- Identifies variants with >90% similarity
- Configurable threshold per entity type
- Fast batch processing
- Returns match confidence scores
- Handles Unicode and special characters
- No false positives for very different terms

---

## Task 3.3: Acronym Resolution System
**Priority:** Medium  
**Dependencies:** Task 2.5

**Description:**
Build system to identify acronym definitions and resolve acronyms to full forms.

**Steps:**
1. Create [`src/normalization/acronym_resolver.py`] with resolver
2. Implement pattern matching for definitions (e.g., "Natural Language Processing (NLP)")
3. Build acronym dictionary from corpus
4. Implement context-based disambiguation
5. Create manual acronym override configuration
6. Add acronym extraction from chunks
7. Store acronym mappings

**Deliverables:**
- [`src/normalization/acronym_resolver.py`] with resolution logic
- Acronym dictionary extraction from corpus
- [`config/acronym_overrides.yaml`] for manual mappings
- Context-based disambiguation

**Acceptance Criteria:**
- Extracts acronym definitions from text
- Builds comprehensive acronym dictionary
- Resolves acronyms to full forms
- Handles multi-meaning acronyms with context
- Manual override capability
- Stores both forms as entity properties

---

## Task 3.4: Entity Deduplication with Embeddings
**Priority:** High  
**Dependencies:** Task 1.9, Task 2.5, Task 3.2

**Description:**
Use embedding similarity to identify semantically similar entities for merging.

**Steps:**
1. Create [`src/normalization/entity_deduplicator.py`] with deduplication
2. Generate embeddings for entity descriptions
3. Implement cosine similarity comparison
4. Cluster similar entities using DBSCAN or hierarchical clustering
5. Identify merge candidates within clusters
6. Score merge suggestions by multiple factors
7. Rank merge candidates by confidence

**Deliverables:**
- [`src/normalization/entity_deduplicator.py`] with deduplication
- Embedding-based similarity scoring
- Clustering implementation
- Merge suggestion ranking

**Acceptance Criteria:**
- Identifies semantically similar entities
- Configurable similarity threshold (default 0.85)
- Clusters entities for review
- Ranks suggestions by confidence
- Considers entity type in matching
- Performance: process 1000 entities in <1 minute

---

## Task 3.5: Normalization Table Implementation
**Priority:** High  
**Dependencies:** Task 3.1, Task 3.2, Task 3.3, Task 3.4

**Description:**
Create normalization table that maps raw text mentions to canonical entity IDs.

**Steps:**
1. Create [`src/normalization/normalization_table.py`] with table manager
2. Design normalization table schema
3. Implement CRUD operations for mappings
4. Add lookup methods (raw text → canonical ID)
5. Support bulk operations
6. Implement versioning for normalization changes
7. Export/import functionality for manual editing

**Deliverables:**
- [`src/normalization/normalization_table.py`] with table management
- Normalization table schema
- Lookup and update operations
- Export/import for manual editing

**Acceptance Criteria:**
- Stores raw text → canonical entity mappings
- Fast lookup (<1ms per query)
- Supports bulk updates
- Tracks normalization method and confidence
- Exportable to JSON/CSV for review
- Can handle 10000+ mappings

---

## Task 3.6: CLI Review Interface - Core
**Priority:** High  
**Dependencies:** Task 2.5, Task 3.5

**Description:**
Build CLI-based interface for reviewing and curating entity candidates.

**Steps:**
1. Create [`src/curation/review_interface.py`] with CLI interface using Typer
2. Implement review queue display
3. Add filtering by confidence, type, status
4. Implement sorting options
5. Display entity details with context
6. Add navigation commands
7. Implement search functionality
8. Create help system

**Deliverables:**
- [`src/curation/review_interface.py`] with CLI interface
- [`scripts/review_entities.py`] entry point script
- User-friendly command system
- Help documentation

**Acceptance Criteria:**
- Interactive CLI with intuitive commands
- Display entities with relevant context
- Filter and sort capabilities
- Search by name or description
- Clear visual presentation
- Responsive (<100ms for commands)
- Help available for all commands

---

## Task 3.7: Entity Curation Operations
**Priority:** High  
**Dependencies:** Task 3.6

**Description:**
Implement operations for approving, merging, rejecting, and editing entities.

**Steps:**
1. Create [`src/curation/entity_approval.py`] with curation logic
2. Implement approve operation (move to production graph)
3. Implement merge operation (combine multiple candidates)
4. Implement reject operation (mark as noise)
5. Implement edit operation (modify properties)
6. Add undo functionality
7. Create audit trail for all operations
8. Update normalization table on approval/merge

**Deliverables:**
- [`src/curation/entity_approval.py`] with curation operations
- Approve, merge, reject, edit functionality
- Audit trail implementation
- Undo capability

**Acceptance Criteria:**
- Approve moves candidate to production with all relationships
- Merge combines properties and updates references
- Reject marks candidate but preserves for review
- Edit updates properties correctly
- All operations logged in audit trail
- Undo works for recent operations
- Normalization table updated automatically

---

## Task 3.8: Batch Curation Operations
**Priority:** Medium  
**Dependencies:** Task 3.7

**Description:**
Implement batch operations for efficient curation of large numbers of entities.

**Steps:**
1. Create [`src/curation/batch_operations.py`] with batch logic
2. Implement batch approve by confidence threshold
3. Implement batch merge for similar entities
4. Add preview before executing batch operations
5. Create operation templates for common patterns
6. Add progress tracking for large batches
7. Implement rollback for batch operations

**Deliverables:**
- [`src/curation/batch_operations.py`] with batch operations
- Preview before execution
- Progress tracking
- Rollback capability

**Acceptance Criteria:**
- Batch approve all entities above threshold
- Batch merge clusters of similar entities
- Preview shows affected entities before execution
- Progress bar for large operations
- Rollback if issues detected
- Dry-run mode for testing

---

## Task 3.9: Discovery Pipeline
**Priority:** High  
**Dependencies:** Task 3.4, Task 3.5

**Description:**
Create pipeline that analyzes entire corpus and generates entity discovery report.

**Steps:**
1. Create [`src/pipeline/discovery_pipeline.py`] with discovery logic
2. Implement corpus-wide entity statistics
3. Calculate co-occurrence matrices
4. Identify entity clusters
5. Suggest new entity types based on patterns
6. Generate merge suggestions
7. Create discovery report
8. Add visualization of entity relationships

**Deliverables:**
- [`src/pipeline/discovery_pipeline.py`] with discovery
- [`scripts/run_discovery.py`] entry point
- Discovery report generator
- Entity statistics and visualizations

**Acceptance Criteria:**
- Analyzes all entity candidates
- Generates frequency statistics
- Identifies co-occurrence patterns
- Clusters similar entities
- Suggests entity types not in schema
- Produces comprehensive report
- Can run on subsets for testing

---

## Phase 3 Summary

**Key Deliverables:**
- String normalization system
- Fuzzy matching for variant detection
- Acronym resolution
- Embedding-based deduplication
- Normalization table for entity mapping
- CLI review interface
- Curation operations (approve, merge, reject, edit)
- Batch curation capabilities
- Discovery pipeline with analytics

**Success Metrics:**
- Entity normalization reduces duplicates significantly
- CLI interface enables efficient manual review
- Curation operations maintain data quality
- Discovery pipeline identifies merge opportunities
- Normalization table improves entity resolution

**Next Phase:** [Phase 3.5 - Enhanced Review Interface](phase-3.5-enhanced-review.md)

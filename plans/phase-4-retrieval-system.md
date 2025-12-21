# Phase 4: Retrieval System (Weeks 7-8)

## Overview

This phase implements the retrieval system that enables querying the Graph RAG system. It includes query parsing, vector retrieval, graph traversal, hybrid search, reranking, and response generation.

**Timeline:** Weeks 7.5-9.5
**Architecture Reference:** [`graph-rag-architecture.md`](graph-rag-architecture.md)
**Previous Phase:** [Phase 3.5 - Enhanced Review Interface](phase-3.5-enhanced-review.md)

---

## Task 4.1: Query Parser and Intent Detection
**Priority:** High  
**Dependencies:** Task 1.9

**Description:**
Build query parser to understand user intent and extract key components.

**Steps:**
1. Create [`src/retrieval/query_parser.py`] with parser
2. Implement entity mention extraction from queries
3. Classify query intent (semantic, structural, procedural, hybrid)
4. Extract query parameters (constraints, filters)
5. Implement query expansion (synonyms, acronyms)
6. Add query validation
7. Store query history for analysis

**Deliverables:**
- [`src/retrieval/query_parser.py`] with parsing logic
- Intent classification implementation
- Entity extraction from queries
- Query expansion logic

**Acceptance Criteria:**
- Extracts entity mentions from natural language
- Classifies query intent accurately (>80%)
- Identifies query constraints
- Expands queries with synonyms
- Handles various query phrasings
- Fast parsing (<50ms per query)

---

## Task 4.2: Vector Retriever
**Priority:** High  
**Dependencies:** Task 1.4, Task 1.9, Task 4.1

**Description:**
Implement semantic search using vector similarity in Qdrant.

**Steps:**
1. Create [`src/retrieval/vector_retriever.py`] with retriever
2. Implement query embedding generation
3. Implement vector search in Qdrant
4. Add filtering by metadata (document, section, entity)
5. Implement pagination for results
6. Add score normalization
7. Extract entity IDs from retrieved chunks

**Deliverables:**
- [`src/retrieval/vector_retriever.py`] with vector search
- Filtering and pagination
- Result scoring and ranking
- Entity extraction from results

**Acceptance Criteria:**
- Searches vector store efficiently
- Configurable top-k results (default 20)
- Filters work on metadata fields
- Returns relevance scores
- Extracts related entity IDs
- Query time <100ms for most queries
- Supports MMR for diversity (optional)

---

## Task 4.3: Graph Retriever - Cypher Queries
**Priority:** High  
**Dependencies:** Task 1.3, Task 4.1

**Description:**
Implement graph-based retrieval using Cypher queries for relationship traversal.

**Steps:**
1. Create [`src/retrieval/graph_retriever.py`] with retriever
2. Implement entity resolution from query mentions
3. Create query templates for common patterns:
   - Multi-hop relationships (DEPENDS_ON, PART_OF)
   - Hierarchical queries (CONTAINS paths)
   - Sequential queries (PRECEDES chains)
   - Procedural queries (REFERENCES)
4. Implement path finding algorithms
5. Add depth limiting for traversals
6. Extract relevant chunks from graph results
7. Score results by graph distance and relevance

**Deliverables:**
- [`src/retrieval/graph_retriever.py`] with graph queries
- Query templates for common patterns
- Entity resolution logic
- Path scoring algorithms

**Acceptance Criteria:**
- Resolves query entities to graph nodes
- Executes multi-hop traversals correctly
- Supports all relationship types
- Configurable max depth (default 3)
- Returns paths with confidence scores
- Query time <200ms for simple queries
- Handles multiple starting entities

---

## Task 4.4: Hybrid Retriever
**Priority:** High  
**Dependencies:** Task 4.2, Task 4.3

**Description:**
Combine vector and graph retrieval strategies for comprehensive results.

**Steps:**
1. Create [`src/retrieval/hybrid_retriever.py`] with hybrid logic
2. Implement parallel execution of vector and graph retrieval
3. Create result merging strategy
4. Implement score fusion (weighted average or learned)
5. Add diversity ranking
6. Handle partial results (only one method succeeds)
7. Implement strategy selection based on query intent

**Deliverables:**
- [`src/retrieval/hybrid_retriever.py`] with hybrid search
- Parallel execution implementation
- Score fusion algorithms
- Strategy selection logic

**Acceptance Criteria:**
- Executes both retrievers in parallel
- Merges results intelligently
- Weighted scoring configurable
- Maintains result diversity
- Falls back gracefully if one retriever fails
- Selects optimal strategy for query type
- Total time <250ms for hybrid queries

---

## Task 4.5: Reranking System
**Priority:** High  
**Dependencies:** Task 4.4

**Description:**
Implement reranking to improve result ordering using multiple signals.

**Steps:**
1. Create [`src/retrieval/reranker.py`] with reranking logic
2. Implement scoring based on:
   - Vector similarity score
   - Graph relevance (distance, centrality)
   - Entity coverage (how many query entities)
   - Confidence scores of entities/relationships
   - Source diversity (multiple documents)
   - Chunk hierarchy level
3. Implement weighted score combination
4. Add result deduplication
5. Create configurable scoring weights

**Deliverables:**
- [`src/retrieval/reranker.py`] with reranking
- Multi-signal scoring implementation
- Configurable weights in config
- Result deduplication

**Acceptance Criteria:**
- Combines multiple relevance signals
- Configurable weights for each signal
- Deduplicates near-identical results
- Improves ranking over base retrieval
- Fast processing (<50ms for 50 results)
- Preserves top relevant results

---

## Task 4.6: Response Generation
**Priority:** Medium  
**Dependencies:** Task 4.5

**Description:**
Generate natural language responses from retrieved context using LLM.

**Steps:**
1. Extend [`src/retrieval/hybrid_retriever.py`] with response generation
2. Implement context formatting from retrieved chunks
3. Create response generation prompt
4. Integrate with LLM (local or API)
5. Add source citations in responses
6. Implement streaming for long responses (optional)
7. Add fallback for low-confidence results

**Deliverables:**
- Response generation in retrieval module
- Context formatting logic
- Response prompts
- Citation generation

**Acceptance Criteria:**
- Generates coherent responses from context
- Includes source citations
- Handles insufficient context gracefully
- Supports API LLMs
- Response time <5 seconds
- References specific chunks in sources

---

## Task 4.7: Query Interface CLI
**Priority:** Medium  
**Dependencies:** Task 4.6

**Description:**
Create CLI interface for querying the system interactively.

**Steps:**
1. Create [`scripts/query_system.py`] with query interface
2. Implement interactive query loop.  Consider using textual or better packages for interactive cli.
3. Add query history
4. Display results with formatting
5. Show entity graph visualizations (ASCII)
6. Add verbose mode for debugging
7. Support query parameters (top-k, strategy, etc.)
8. Implement result export (JSON, Markdown)

**Deliverables:**
- [`scripts/query_system.py`] interactive CLI
- Query history
- Result formatting
- Export capabilities

**Acceptance Criteria:**
- Interactive query interface
- Displays formatted results
- Shows confidence scores
- Includes source citations
- History of recent queries
- Export results to files
- Configurable retrieval parameters

---

## Task 4.8: Query Strategy Optimization
**Priority:** Low  
**Dependencies:** Task 4.4, Task 4.7

**Description:**
Analyze query patterns and optimize strategy selection.

**Steps:**
1. Implement query logging with results
2. Analyze query patterns by intent
3. Tune strategy selection rules
4. Optimize scoring weights based on feedback
5. Create query performance metrics
6. Add caching for common queries
7. Document optimization findings

**Deliverables:**
- Query analytics implementation
- Optimized strategy selection
- Performance metrics
- Query caching

**Acceptance Criteria:**
- Logs all queries with results and timing
- Analyzes query patterns
- Optimized weights improve relevance
- Caching speeds up repeated queries
- Metrics track query performance
- Documentation of optimization process

---

## Phase 4 Summary

**Key Deliverables:**
- Query parser with intent detection
- Vector-based semantic retrieval
- Graph-based relationship traversal
- Hybrid retrieval combining both approaches
- Reranking system with multiple signals
- Response generation with citations
- Interactive CLI query interface
- Query optimization and caching

**Success Metrics:**
- Query response time <2 seconds for simple queries
- Hybrid retrieval improves result quality
- Reranking enhances result relevance
- Response generation provides coherent answers
- CLI interface enables efficient querying
- System handles various query types effectively

**Next Phase:** [Phase 5 - Incremental Updates](phase-5-incremental-updates.md)
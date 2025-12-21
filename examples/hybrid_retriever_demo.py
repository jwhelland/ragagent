"""Demo script for HybridRetriever (Task 4.4).

This script demonstrates how to use the HybridRetriever to combine
vector-based semantic search with graph-based relationship traversal.

Usage:
    uv run python examples/hybrid_retriever_demo.py
"""

from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalStrategy
from src.retrieval.query_parser import QueryParser
from src.storage.neo4j_manager import Neo4jManager
from src.utils.config import Config


def demo_automatic_strategy():
    """Demonstrate hybrid retrieval with automatic strategy selection."""
    print("\n=== Demo 1: Automatic Strategy Selection ===\n")

    # Load configuration
    config = Config.from_yaml()

    # Initialize managers and retriever
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    hybrid_retriever = HybridRetriever(config=config, neo4j_manager=neo4j)

    # Parse a query (automatic strategy selection)
    query_parser = QueryParser(config=config)

    # Test different query types
    queries = [
        "What is the thermal control system?",  # Semantic query
        "What are the components of the power system?",  # Structural query
        "How to perform a battery health check?",  # Procedural query
    ]

    for query_text in queries:
        print(f"\nQuery: {query_text}")
        parsed_query = query_parser.parse(query_text)

        print(f"  Intent: {parsed_query.intent.value}")
        print(f"  Requires graph: {parsed_query.requires_graph_traversal}")
        print(f"  Entity mentions: {len(parsed_query.entity_mentions)}")

        # Perform hybrid retrieval (strategy selected automatically)
        result = hybrid_retriever.retrieve(parsed_query, top_k=5)

        print(f"  Strategy used: {result.strategy_used.value}")
        print(f"  Results: {len(result.chunks)}")
        print(f"  Vector success: {result.vector_success}")
        print(f"  Graph success: {result.graph_success}")
        print(f"  Retrieval time: {result.retrieval_time_ms:.2f}ms")

        # Show top result
        if result.chunks:
            top_chunk = result.chunks[0]
            print(f"\n  Top result (Rank {top_chunk.rank}):")
            print(f"    Final score: {top_chunk.final_score:.3f}")
            print(
                f"    Vector score: {top_chunk.vector_score:.3f if top_chunk.vector_score else 'N/A'}"
            )
            print(
                f"    Graph score: {top_chunk.graph_score:.3f if top_chunk.graph_score else 'N/A'}"
            )
            print(f"    Entity coverage: {top_chunk.entity_coverage_score:.3f}")
            print(f"    Source: {top_chunk.source}")
            print(f"    Content preview: {top_chunk.content[:150]}...")

        print("-" * 60)

    neo4j.close()


def demo_parallel_hybrid():
    """Demonstrate parallel hybrid retrieval."""
    print("\n=== Demo 2: Parallel Hybrid Retrieval ===\n")

    config = Config.from_yaml()
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    hybrid_retriever = HybridRetriever(config=config, neo4j_manager=neo4j)
    query_parser = QueryParser(config=config)

    # Query that benefits from both vector and graph search
    parsed_query = query_parser.parse(
        "How does the battery system depend on other power components?"
    )

    print(f"Query: {parsed_query.original_text}")
    print(f"Intent: {parsed_query.intent.value}")

    # Force parallel hybrid strategy
    result = hybrid_retriever.retrieve(
        parsed_query,
        strategy=RetrievalStrategy.HYBRID_PARALLEL,
        top_k=10,
    )

    print(f"\nStrategy: {result.strategy_used.value}")
    print(f"Total results: {result.total_results}")
    print(f"Vector results: {result.vector_results}")
    print(f"Graph results: {result.graph_results}")
    print(f"Merged results: {result.merged_results}")
    print("\nTiming breakdown:")
    print(f"  Vector time: {result.vector_time_ms:.2f}ms")
    print(f"  Graph time: {result.graph_time_ms:.2f}ms")
    print(f"  Merge time: {result.merge_time_ms:.2f}ms")
    print(f"  Total time: {result.retrieval_time_ms:.2f}ms")

    # Show score distribution
    print(f"\nTop {len(result.chunks)} results:")
    for i, chunk in enumerate(result.chunks[:5], 1):
        print(f"\n  {i}. Chunk {chunk.chunk_id[:12]}...")
        print(f"     Final score: {chunk.final_score:.3f}")
        print(
            f"     Vector: {chunk.vector_score:.3f if chunk.vector_score else 'N/A'}, "
            f"Graph: {chunk.graph_score:.3f if chunk.graph_score else 'N/A'}"
        )
        print(f"     Source: {chunk.source}")
        print(f"     Entities: {len(chunk.entity_ids)}")

    neo4j.close()


def demo_score_fusion():
    """Demonstrate score fusion and reranking."""
    print("\n=== Demo 3: Score Fusion and Reranking ===\n")

    config = Config.from_yaml()
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    hybrid_retriever = HybridRetriever(config=config, neo4j_manager=neo4j)
    query_parser = QueryParser(config=config)

    parsed_query = query_parser.parse("Power distribution system components and relationships")

    print(f"Query: {parsed_query.original_text}")

    # Check reranking configuration
    reranking_config = config.retrieval.reranking
    print(f"\nReranking enabled: {reranking_config.enabled}")
    print("Score weights:")
    for signal, weight in reranking_config.weights.items():
        print(f"  {signal}: {weight}")

    # Perform retrieval with reranking
    result = hybrid_retriever.retrieve(parsed_query, top_k=10)

    print(f"\nResults: {len(result.chunks)}")
    print(f"Reranking applied: {result.reranking_enabled}")

    # Show score breakdown for top results
    print("\nScore breakdown for top 5 results:")
    for i, chunk in enumerate(result.chunks[:5], 1):
        print(f"\n  {i}. Final score: {chunk.final_score:.4f}")
        print("     Components:")
        print(
            f"       - Vector similarity: {chunk.vector_score:.4f if chunk.vector_score else 0.0}"
        )
        print(f"       - Graph relevance: {chunk.graph_score:.4f if chunk.graph_score else 0.0}")
        print(f"       - Entity coverage: {chunk.entity_coverage_score:.4f}")
        print(f"       - Confidence: {chunk.confidence_score:.4f}")
        print(f"       - Diversity: {chunk.diversity_score:.4f}")

    neo4j.close()


def demo_strategy_comparison():
    """Compare different retrieval strategies on the same query."""
    print("\n=== Demo 4: Strategy Comparison ===\n")

    config = Config.from_yaml()
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    hybrid_retriever = HybridRetriever(config=config, neo4j_manager=neo4j)
    query_parser = QueryParser(config=config)

    parsed_query = query_parser.parse("Satellite thermal management system")

    strategies = [
        RetrievalStrategy.VECTOR_ONLY,
        RetrievalStrategy.GRAPH_ONLY,
        RetrievalStrategy.HYBRID_PARALLEL,
    ]

    print(f"Query: {parsed_query.original_text}\n")
    print("Comparing strategies:")

    results = {}
    for strategy in strategies:
        result = hybrid_retriever.retrieve(parsed_query, strategy=strategy, top_k=5)
        results[strategy] = result

        print(f"\n  {strategy.value}:")
        print(f"    Results: {len(result.chunks)}")
        print(f"    Time: {result.retrieval_time_ms:.2f}ms")
        print(f"    Vector success: {result.vector_success}")
        print(f"    Graph success: {result.graph_success}")
        if result.chunks:
            print(f"    Top score: {result.chunks[0].final_score:.3f}")

    # Compare top results
    print("\n\nTop result from each strategy:")
    for strategy, result in results.items():
        if result.chunks:
            top = result.chunks[0]
            print(f"\n  {strategy.value}:")
            print(f"    Score: {top.final_score:.3f}")
            print(f"    Content: {top.content[:100]}...")

    neo4j.close()


def demo_diversity_ranking():
    """Demonstrate diversity-aware ranking."""
    print("\n=== Demo 5: Diversity Ranking ===\n")

    config = Config.from_yaml()
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    # Enable diversity in config (if not already)
    if config.retrieval.reranking.weights.get("diversity", 0.0) == 0.0:
        print("Note: Diversity weight is 0.0 in config. Setting to 0.1 for demo.")
        config.retrieval.reranking.weights["diversity"] = 0.1

    hybrid_retriever = HybridRetriever(config=config, neo4j_manager=neo4j)
    query_parser = QueryParser(config=config)

    parsed_query = query_parser.parse("Power system architecture")

    print(f"Query: {parsed_query.original_text}")

    result = hybrid_retriever.retrieve(parsed_query, top_k=10)

    print(f"\nResults with diversity ranking: {len(result.chunks)}")
    print("\nDiversity scores:")
    for i, chunk in enumerate(result.chunks[:8], 1):
        print(
            f"  {i}. Diversity: {chunk.diversity_score:.3f}, "
            f"Final: {chunk.final_score:.3f}, "
            f"Document: {chunk.document_id[:20]}..."
        )

    # Check document diversity
    unique_docs = len(result.get_document_ids())
    print(f"\nUnique documents in top 10: {unique_docs}/{len(result.chunks)}")

    neo4j.close()


def demo_graceful_fallback():
    """Demonstrate graceful fallback when one retriever fails."""
    print("\n=== Demo 6: Graceful Fallback ===\n")

    config = Config.from_yaml()
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    hybrid_retriever = HybridRetriever(config=config, neo4j_manager=neo4j)
    query_parser = QueryParser(config=config)

    # Query without entities (graph retrieval may return no results)
    parsed_query = query_parser.parse("general information about satellite operations")

    print(f"Query: {parsed_query.original_text}")
    print(f"Entity mentions: {len(parsed_query.entity_mentions)}")

    result = hybrid_retriever.retrieve(parsed_query, strategy=RetrievalStrategy.HYBRID_PARALLEL)

    print(f"\nStrategy: {result.strategy_used.value}")
    print(f"Vector success: {result.vector_success}")
    print(f"Graph success: {result.graph_success}")
    print(f"Results: {len(result.chunks)}")

    if result.vector_success and not result.graph_success:
        print("\nNote: Graph retrieval returned no results, but vector retrieval succeeded.")
        print("Hybrid retriever gracefully fell back to vector-only results.")
    elif result.graph_success and not result.vector_success:
        print("\nNote: Vector retrieval failed, but graph retrieval succeeded.")
        print("Hybrid retriever gracefully fell back to graph-only results.")
    elif result.vector_success and result.graph_success:
        print("\nBoth retrievers succeeded. Results merged successfully.")

    neo4j.close()


def demo_statistics():
    """Demonstrate getting hybrid retrieval statistics."""
    print("\n=== Demo 7: Retrieval Statistics ===\n")

    config = Config.from_yaml()
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    hybrid_retriever = HybridRetriever(config=config, neo4j_manager=neo4j)

    stats = hybrid_retriever.get_statistics()

    print("Hybrid Retriever Statistics:\n")

    # Hybrid config
    hybrid_config = stats.get("hybrid_config", {})
    print("Hybrid Configuration:")
    print(f"  Enabled: {hybrid_config.get('enabled', False)}")
    print(f"  Parallel execution: {hybrid_config.get('parallel_execution', False)}")
    print(f"  Strategy selection: {hybrid_config.get('strategy_selection', 'auto')}")

    # Reranking config
    reranking_config = stats.get("reranking_config", {})
    print("\nReranking Configuration:")
    print(f"  Enabled: {reranking_config.get('enabled', False)}")
    print(f"  Max results: {reranking_config.get('max_results', 10)}")
    print("  Weights:")
    for signal, weight in reranking_config.get("weights", {}).items():
        print(f"    {signal}: {weight}")

    # Vector retriever stats
    vector_stats = stats.get("vector_retriever", {})
    print("\nVector Retriever:")
    print(f"  Total chunks: {vector_stats.get('total_chunks', 0)}")
    print(f"  Embedding model: {vector_stats.get('embedding_model', 'N/A')}")
    print(f"  Top-k default: {vector_stats.get('top_k_default', 20)}")

    # Graph retriever stats
    graph_stats = stats.get("graph_retriever", {})
    print("\nGraph Retriever:")
    print(f"  Total entities: {graph_stats.get('total_entities', 0)}")
    print(f"  Total relationships: {graph_stats.get('total_relationships', 0)}")

    neo4j.close()


if __name__ == "__main__":
    print("=" * 60)
    print("HybridRetriever Demo - Task 4.4")
    print("Hybrid Retrieval: Vector + Graph Search")
    print("=" * 60)

    try:
        # Run demos
        demo_automatic_strategy()
        demo_parallel_hybrid()
        demo_score_fusion()
        demo_strategy_comparison()
        demo_diversity_ranking()
        demo_graceful_fallback()
        demo_statistics()

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()
        print("\nNote: This demo requires:")
        print("  1. Neo4j running (docker-compose up -d)")
        print("  2. Qdrant running (docker-compose up -d)")
        print("  3. Data ingested (uv run ragagent-ingest)")
        print("  4. spaCy model installed (uv run spacy download en_core_web_lg)")

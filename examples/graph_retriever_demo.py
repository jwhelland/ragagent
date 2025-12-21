"""Demo script for GraphRetriever (Task 4.3).

This script demonstrates how to use the GraphRetriever to perform
graph-based retrieval using relationship traversal in Neo4j.

Usage:
    uv run python examples/graph_retriever_demo.py
"""

from datetime import datetime

from src.retrieval.graph_retriever import GraphRetriever, TraversalStrategy
from src.retrieval.query_parser import QueryParser, EntityMention, ParsedQuery, QueryIntent
from src.storage.neo4j_manager import Neo4jManager
from src.storage.schemas import EntityType, RelationshipType
from src.utils.config import Config


def demo_basic_retrieval():
    """Demonstrate basic graph retrieval with entity resolution."""
    print("\n=== Demo 1: Basic Graph Retrieval ===\n")

    # Load configuration
    config = Config.from_yaml()

    # Initialize Neo4j and GraphRetriever
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    graph_retriever = GraphRetriever(config=config, neo4j_manager=neo4j)

    # Parse a query
    query_parser = QueryParser(config=config)
    parsed_query = query_parser.parse("What are the components of the Electrical Power System?")

    print(f"Query: {parsed_query.original_text}")
    print(f"Intent: {parsed_query.intent.value}")
    print(f"Requires graph traversal: {parsed_query.requires_graph_traversal}")
    print(f"Max depth: {parsed_query.max_depth}")
    print(f"Entity mentions: {len(parsed_query.entity_mentions)}")

    # Perform graph retrieval
    result = graph_retriever.retrieve(parsed_query)

    print(f"\nResolved entities: {len(result.resolved_entities)}")
    for entity in result.resolved_entities:
        print(f"  - {entity.canonical_name} ({entity.entity_type.value})")
        print(f"    Confidence: {entity.confidence:.2f}, Method: {entity.match_method}")

    print(f"\nPaths found: {len(result.paths)}")
    for i, path in enumerate(result.paths[:5], 1):  # Show first 5 paths
        print(f"  Path {i}:")
        print(f"    Length: {path.length}, Score: {path.score:.2f}")
        print(f"    Start: {path.start_entity_id}")
        print(f"    End: {path.end_entity_id}")

    print(f"\nTotal chunks: {len(result.chunk_ids)}")
    print(f"Total entities: {len(result.entity_ids)}")
    print(f"Strategy: {result.strategy_used.value}")
    print(f"Retrieval time: {result.retrieval_time_ms:.2f}ms")

    neo4j.close()


def demo_custom_strategy():
    """Demonstrate graph retrieval with custom traversal strategy."""
    print("\n=== Demo 2: Custom Traversal Strategy ===\n")

    config = Config.from_yaml()
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    graph_retriever = GraphRetriever(config=config, neo4j_manager=neo4j)

    # Create a custom parsed query for hierarchical traversal
    parsed_query = ParsedQuery(
        query_id="demo_query_hierarchical",
        original_text="Show me the subsystems in the power system",
        normalized_text="show me the subsystems in the power system",
        intent=QueryIntent.STRUCTURAL,
        intent_confidence=0.9,
        entity_mentions=[
            EntityMention(
                text="power system",
                normalized="power_system",
                entity_type=EntityType.SYSTEM,
                start_char=30,
                end_char=42,
                confidence=0.85,
            )
        ],
        relationship_types=[RelationshipType.CONTAINS],
        constraints=[],
        expanded_terms={},
        keywords=["subsystems", "power", "system"],
        requires_graph_traversal=True,
        max_depth=2,
        timestamp=datetime.now(),
        metadata={},
    )

    # Perform retrieval with hierarchical strategy
    result = graph_retriever.retrieve(
        parsed_query,
        strategy=TraversalStrategy.HIERARCHICAL,
        max_depth=2,
    )

    print(f"Query: {parsed_query.original_text}")
    print(f"Strategy: {result.strategy_used.value}")
    print(f"Max depth: {result.max_depth}")
    print(f"\nResolved entities: {len(result.resolved_entities)}")
    print(f"Paths found: {len(result.paths)}")
    print(f"Retrieval time: {result.retrieval_time_ms:.2f}ms")

    neo4j.close()


def demo_multi_entity_path():
    """Demonstrate shortest path finding between multiple entities."""
    print("\n=== Demo 3: Shortest Path Between Entities ===\n")

    config = Config.from_yaml()
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    graph_retriever = GraphRetriever(config=config, neo4j_manager=neo4j)

    # Create query with multiple entities
    parsed_query = ParsedQuery(
        query_id="demo_query_path",
        original_text="How is the battery connected to the power distribution unit?",
        normalized_text="how is the battery connected to the power distribution unit",
        intent=QueryIntent.STRUCTURAL,
        intent_confidence=0.85,
        entity_mentions=[
            EntityMention(
                text="battery",
                normalized="battery",
                entity_type=EntityType.COMPONENT,
                start_char=11,
                end_char=18,
                confidence=0.9,
            ),
            EntityMention(
                text="power distribution unit",
                normalized="power_distribution_unit",
                entity_type=EntityType.COMPONENT,
                start_char=36,
                end_char=59,
                confidence=0.9,
            ),
        ],
        relationship_types=[],
        constraints=[],
        expanded_terms={},
        keywords=["battery", "connected", "power", "distribution", "unit"],
        requires_graph_traversal=True,
        max_depth=3,
        timestamp=datetime.now(),
        metadata={},
    )

    # Perform retrieval with shortest path strategy
    result = graph_retriever.retrieve(
        parsed_query,
        strategy=TraversalStrategy.SHORTEST_PATH,
        relationship_types=[
            RelationshipType.PART_OF,
            RelationshipType.CONTAINS,
            RelationshipType.DEPENDS_ON,
            RelationshipType.PROVIDES_POWER_TO,
        ],
    )

    print(f"Query: {parsed_query.original_text}")
    print(f"Strategy: {result.strategy_used.value}")
    print(f"\nResolved entities: {len(result.resolved_entities)}")
    for entity in result.resolved_entities:
        print(f"  - {entity.canonical_name} ({entity.entity_type.value})")

    print(f"\nPaths found: {len(result.paths)}")
    for i, path in enumerate(result.paths, 1):
        print(f"\n  Path {i}:")
        print(f"    Length: {path.length} hops")
        print(f"    Score: {path.score:.2f}")
        print(f"    Confidence: {path.confidence:.2f}")
        print(f"    Nodes: {len(path.nodes)}")
        print(f"    Relationships: {len(path.relationships)}")

    neo4j.close()


def demo_chunk_extraction():
    """Demonstrate extracting chunks from graph results."""
    print("\n=== Demo 4: Chunk Extraction from Graph Results ===\n")

    config = Config.from_yaml()
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    graph_retriever = GraphRetriever(config=config, neo4j_manager=neo4j)
    query_parser = QueryParser(config=config)

    parsed_query = query_parser.parse("What is the thermal control system?")

    # Perform retrieval
    result = graph_retriever.retrieve(parsed_query)

    print(f"Query: {parsed_query.original_text}")
    print(f"\nEntity IDs found: {len(result.entity_ids)}")

    # Extract chunks for the entities
    if result.entity_ids:
        chunks = graph_retriever.get_chunks_for_entities(result.entity_ids)

        print(f"Chunks extracted: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3], 1):  # Show first 3 chunks
            print(f"\n  Chunk {i}:")
            print(f"    ID: {chunk.get('id', 'N/A')}")
            print(f"    Level: {chunk.get('level', 'N/A')}")
            print(f"    Section: {chunk.get('section_title', 'N/A')}")
            content = chunk.get("content", "")
            print(f"    Content preview: {content[:100]}...")
    else:
        print("No entities resolved - cannot extract chunks")

    neo4j.close()


def demo_statistics():
    """Demonstrate getting retrieval statistics."""
    print("\n=== Demo 5: Retrieval Statistics ===\n")

    config = Config.from_yaml()
    neo4j = Neo4jManager(config=config.database)
    neo4j.connect()

    graph_retriever = GraphRetriever(config=config, neo4j_manager=neo4j)

    stats = graph_retriever.get_statistics()

    print("Graph Retriever Statistics:")
    print(f"  Total entities: {stats.get('total_entities', 0)}")
    print(f"  Total relationships: {stats.get('total_relationships', 0)}")
    print(f"  Total chunks: {stats.get('total_chunks', 0)}")

    graph_config = stats.get("graph_config", {})
    print(f"\nGraph Configuration:")
    print(f"  Max depth: {graph_config.get('max_depth', 'N/A')}")
    print(f"  Relationship types: {len(graph_config.get('relationship_types', []))}")
    print(f"  Shortest path enabled: {graph_config.get('enable_shortest_path', False)}")

    neo4j.close()


if __name__ == "__main__":
    print("=" * 60)
    print("GraphRetriever Demo - Task 4.3")
    print("Graph-based Retrieval using Cypher Queries")
    print("=" * 60)

    try:
        # Run demos
        demo_basic_retrieval()
        demo_custom_strategy()
        demo_multi_entity_path()
        demo_chunk_extraction()
        demo_statistics()

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demo: {e}")
        print("\nNote: This demo requires:")
        print("  1. Neo4j running (docker-compose up -d)")
        print("  2. Data ingested (uv run ragagent-ingest)")
        print("  3. spaCy model installed (uv run spacy download en_core_web_lg)")

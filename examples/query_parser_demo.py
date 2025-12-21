"""Demo script for query parser functionality.

This script demonstrates how to use the QueryParser to parse natural language
queries and extract structured information including intent, entities, relationships,
constraints, and expanded terms.

Usage:
    uv run python examples/query_parser_demo.py
"""

from src.normalization.acronym_resolver import AcronymResolver
from src.retrieval.query_parser import QueryParser
from src.utils.config import load_config


def main() -> None:
    """Run query parser demo."""
    # Load configuration
    config = load_config()

    # Initialize acronym resolver (optional but recommended)
    acronym_resolver = AcronymResolver(config=config.normalization)
    # Try to load stored acronym mappings
    try:
        acronym_resolver.load_mappings()
        print("✓ Loaded acronym mappings")
    except FileNotFoundError:
        print("ℹ No stored acronym mappings found")

    # Initialize query parser
    parser = QueryParser(config=config, acronym_resolver=acronym_resolver)
    print("✓ Query parser initialized\n")

    # Example queries to demonstrate different capabilities
    example_queries = [
        "What is the Electrical Power System?",
        "What components are contained in the attitude control subsystem?",
        "How to perform the spacecraft startup procedure?",
        "Show entities with confidence greater than 0.8",
        "What systems depend on the battery controller?",
        "Explain how the EPS works",  # Tests acronym expansion
        "What are the steps for power system initialization?",
    ]

    print("=" * 80)
    print("QUERY PARSER DEMO")
    print("=" * 80)

    for i, query_text in enumerate(example_queries, 1):
        print(f"\n[Query {i}] {query_text}")
        print("-" * 80)

        # Parse the query
        parsed = parser.parse(query_text)

        # Display results
        print(f"Intent: {parsed.intent.value} (confidence: {parsed.intent_confidence:.2f})")
        print(f"Normalized: {parsed.normalized_text}")

        if parsed.entity_mentions:
            print(f"\nEntity Mentions ({len(parsed.entity_mentions)}):")
            for mention in parsed.entity_mentions:
                entity_type_str = (
                    f" [{mention.entity_type.value}]" if mention.entity_type else ""
                )
                print(
                    f"  • {mention.text}{entity_type_str} "
                    f"→ {mention.normalized} (confidence: {mention.confidence:.2f})"
                )

        if parsed.relationship_types:
            print(f"\nRelationship Types ({len(parsed.relationship_types)}):")
            for rel_type in parsed.relationship_types:
                print(f"  • {rel_type.value}")

        if parsed.constraints:
            print(f"\nConstraints ({len(parsed.constraints)}):")
            for constraint in parsed.constraints:
                print(
                    f"  • {constraint.field} {constraint.operator} {constraint.value}"
                )

        if parsed.expanded_terms:
            print(f"\nExpanded Terms ({len(parsed.expanded_terms)}):")
            for term, expansions in parsed.expanded_terms.items():
                print(f"  • {term} → {', '.join(expansions[:3])}")

        if parsed.keywords:
            print(f"\nKeywords: {', '.join(parsed.keywords[:10])}")

        print(f"\nGraph Traversal: {'Required' if parsed.requires_graph_traversal else 'Not required'}")
        if parsed.max_depth:
            print(f"Max Depth: {parsed.max_depth}")

        # Validate query
        is_valid, error = parser.validate_query(parsed)
        if not is_valid:
            print(f"\n⚠ Validation Error: {error}")

    # Display query statistics
    print("\n" + "=" * 80)
    print("QUERY STATISTICS")
    print("=" * 80)
    stats = parser.get_query_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()

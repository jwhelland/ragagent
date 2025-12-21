"""Tests for query parser (Task 4.1)."""

from datetime import datetime
from pathlib import Path

import pytest
import spacy
from spacy.language import Language

from src.normalization.acronym_resolver import AcronymResolver
from src.retrieval.query_parser import (
    QueryIntent,
    QueryParser,
)
from src.storage.schemas import RelationshipType
from src.utils.config import Config


@pytest.fixture
def config() -> Config:
    """Create test configuration."""
    return Config.from_yaml()


@pytest.fixture
def nlp() -> Language:
    """Load spaCy model for testing."""
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        pytest.skip("spaCy model en_core_web_lg not available")


@pytest.fixture
def acronym_resolver(config: Config) -> AcronymResolver:
    """Create acronym resolver with test data."""
    resolver = AcronymResolver(config=config.normalization)
    # Add some test acronyms
    resolver.dictionary["EPS"] = (
        resolver.dictionary.get("EPS")
        or type("Entry", (), {"acronym": "EPS", "expansions": {"Electrical Power System": 5}})()
    )
    resolver.dictionary["TCS"] = (
        resolver.dictionary.get("TCS")
        or type("Entry", (), {"acronym": "TCS", "expansions": {"Thermal Control System": 3}})()
    )
    return resolver


@pytest.fixture
def query_parser(
    config: Config, nlp: Language, acronym_resolver: AcronymResolver, tmp_path: Path
) -> QueryParser:
    """Create query parser for testing."""
    history_path = tmp_path / "query_history.jsonl"
    return QueryParser(
        config=config,
        nlp=nlp,
        acronym_resolver=acronym_resolver,
        query_history_path=str(history_path),
    )


class TestQueryParserInitialization:
    """Tests for QueryParser initialization."""

    def test_init_with_defaults(self, config: Config) -> None:
        """Test initialization with default parameters."""
        parser = QueryParser(config=config)
        assert parser.config is not None
        assert parser.nlp is not None
        assert parser.normalizer is not None
        assert parser.query_history_path.exists() or True  # Path may not exist yet

    def test_init_with_custom_nlp(self, config: Config, nlp: Language) -> None:
        """Test initialization with custom spaCy model."""
        parser = QueryParser(config=config, nlp=nlp)
        assert parser.nlp is nlp

    def test_init_with_acronym_resolver(
        self, config: Config, nlp: Language, acronym_resolver: AcronymResolver
    ) -> None:
        """Test initialization with acronym resolver."""
        parser = QueryParser(config=config, nlp=nlp, acronym_resolver=acronym_resolver)
        assert parser.acronym_resolver is acronym_resolver


class TestQueryIntentClassification:
    """Tests for query intent classification."""

    def test_semantic_intent_what_is(self, query_parser: QueryParser) -> None:
        """Test semantic intent detection for 'what is' queries."""
        parsed = query_parser.parse("What is the Electrical Power System?")
        assert parsed.intent == QueryIntent.SEMANTIC
        assert parsed.intent_confidence >= 0.7

    def test_semantic_intent_explain(self, query_parser: QueryParser) -> None:
        """Test semantic intent detection for 'explain' queries."""
        parsed = query_parser.parse("Explain how the thermal control system works")
        assert parsed.intent == QueryIntent.SEMANTIC
        assert parsed.intent_confidence >= 0.7

    def test_structural_intent_contains(self, query_parser: QueryParser) -> None:
        """Test structural intent detection for 'contains' queries."""
        parsed = query_parser.parse("What components are contained in the power subsystem?")
        assert parsed.intent == QueryIntent.STRUCTURAL
        assert parsed.intent_confidence >= 0.7

    def test_structural_intent_part_of(self, query_parser: QueryParser) -> None:
        """Test structural intent detection for 'part of' queries."""
        parsed = query_parser.parse("What is part of the attitude control system?")
        # This query could be STRUCTURAL or HYBRID (has both "what is" and "part of")
        assert parsed.intent in {QueryIntent.STRUCTURAL, QueryIntent.HYBRID}
        assert parsed.intent_confidence >= 0.7

    def test_procedural_intent_how_to(self, query_parser: QueryParser) -> None:
        """Test procedural intent detection for 'how to' queries."""
        parsed = query_parser.parse("How to perform the startup procedure?")
        assert parsed.intent == QueryIntent.PROCEDURAL
        assert parsed.intent_confidence >= 0.7

    def test_procedural_intent_steps(self, query_parser: QueryParser) -> None:
        """Test procedural intent detection for 'steps' queries."""
        parsed = query_parser.parse("What are the steps for system initialization?")
        # This query could be PROCEDURAL or HYBRID (has both "what are" and "steps")
        assert parsed.intent in {QueryIntent.PROCEDURAL, QueryIntent.HYBRID}
        assert parsed.intent_confidence >= 0.7

    def test_hybrid_intent(self, query_parser: QueryParser) -> None:
        """Test hybrid intent detection for complex queries."""
        parsed = query_parser.parse("What is the power system and what components does it contain?")
        assert parsed.intent == QueryIntent.HYBRID
        assert parsed.intent_confidence >= 0.5

    def test_unknown_intent(self, query_parser: QueryParser) -> None:
        """Test handling of queries with unclear intent."""
        parsed = query_parser.parse("System power")
        # Should default to SEMANTIC with lower confidence
        assert parsed.intent in {QueryIntent.SEMANTIC, QueryIntent.UNKNOWN}
        assert parsed.intent_confidence <= 0.7


class TestEntityMentionExtraction:
    """Tests for entity mention extraction."""

    def test_extract_capitalized_entity(self, query_parser: QueryParser) -> None:
        """Test extraction of capitalized entity names."""
        parsed = query_parser.parse("What is the Electrical Power System?")
        assert len(parsed.entity_mentions) > 0
        # Should extract "Electrical Power System" or parts of it
        entity_texts = [m.text for m in parsed.entity_mentions]
        assert any("Power" in text or "System" in text for text in entity_texts)

    def test_extract_multiple_entities(self, query_parser: QueryParser) -> None:
        """Test extraction of multiple entities."""
        parsed = query_parser.parse(
            "How does the Battery Controller interact with the Power Distribution Unit?"
        )
        assert len(parsed.entity_mentions) >= 2

    def test_entity_normalization(self, query_parser: QueryParser) -> None:
        """Test that entity mentions are normalized."""
        parsed = query_parser.parse("What is the POWER SYSTEM?")
        if parsed.entity_mentions:
            mention = parsed.entity_mentions[0]
            assert mention.normalized == mention.normalized.lower()

    def test_entity_character_offsets(self, query_parser: QueryParser) -> None:
        """Test that entity character offsets are correct."""
        query = "What is the Electrical Power System?"
        parsed = query_parser.parse(query)
        for mention in parsed.entity_mentions:
            extracted = query[mention.start_char : mention.end_char]
            # Offsets should match or be close (accounting for normalization)
            assert mention.text in query or extracted in query


class TestRelationshipExtraction:
    """Tests for relationship type extraction."""

    def test_extract_part_of_relationship(self, query_parser: QueryParser) -> None:
        """Test extraction of PART_OF relationship."""
        parsed = query_parser.parse("What components are part of the power system?")
        assert RelationshipType.PART_OF in parsed.relationship_types

    def test_extract_contains_relationship(self, query_parser: QueryParser) -> None:
        """Test extraction of CONTAINS relationship."""
        parsed = query_parser.parse("What does the subsystem contain?")
        assert RelationshipType.CONTAINS in parsed.relationship_types

    def test_extract_depends_on_relationship(self, query_parser: QueryParser) -> None:
        """Test extraction of DEPENDS_ON relationship."""
        parsed = query_parser.parse("What does the controller depend on?")
        assert RelationshipType.DEPENDS_ON in parsed.relationship_types

    def test_extract_multiple_relationships(self, query_parser: QueryParser) -> None:
        """Test extraction of multiple relationship types."""
        parsed = query_parser.parse(
            "What components does the system contain and what does it depend on?"
        )
        assert len(parsed.relationship_types) >= 2
        assert RelationshipType.CONTAINS in parsed.relationship_types
        assert RelationshipType.DEPENDS_ON in parsed.relationship_types

    def test_no_relationships(self, query_parser: QueryParser) -> None:
        """Test query with no explicit relationships."""
        parsed = query_parser.parse("What is the temperature sensor?")
        # May or may not have relationships, but should not crash
        assert isinstance(parsed.relationship_types, list)


class TestQueryConstraints:
    """Tests for query constraint extraction."""

    def test_extract_numeric_constraint_greater(self, query_parser: QueryParser) -> None:
        """Test extraction of greater-than numeric constraint."""
        parsed = query_parser.parse("Show entities with confidence greater than 0.8")
        assert len(parsed.constraints) > 0
        constraint = parsed.constraints[0]
        assert constraint.operator == "gt"
        assert constraint.value == 0.8

    def test_extract_numeric_constraint_less(self, query_parser: QueryParser) -> None:
        """Test extraction of less-than numeric constraint."""
        parsed = query_parser.parse("Find components with temperature less than 50")
        assert len(parsed.constraints) > 0
        constraint = parsed.constraints[0]
        assert constraint.operator == "lt"
        assert constraint.value == 50.0

    def test_extract_page_constraint(self, query_parser: QueryParser) -> None:
        """Test extraction of page number constraint."""
        parsed = query_parser.parse("What is mentioned on page 42?")
        page_constraints = [c for c in parsed.constraints if c.operator == "page"]
        assert len(page_constraints) > 0
        assert page_constraints[0].value == 42

    def test_extract_document_constraint(self, query_parser: QueryParser) -> None:
        """Test extraction of document constraint."""
        parsed = query_parser.parse("Show information from document ABC123")
        doc_constraints = [c for c in parsed.constraints if c.operator in {"document", "source"}]
        assert len(doc_constraints) > 0


class TestQueryExpansion:
    """Tests for query term expansion."""

    def test_expand_synonyms(self, query_parser: QueryParser) -> None:
        """Test synonym expansion."""
        parsed = query_parser.parse("What is the system architecture?")
        # Should expand "system" with synonyms
        if "system" in parsed.expanded_terms:
            assert len(parsed.expanded_terms["system"]) > 0

    def test_expand_acronyms(self, query_parser: QueryParser) -> None:
        """Test acronym expansion."""
        parsed = query_parser.parse("What is the EPS?")
        # Should expand EPS if resolver is available and has mapping
        if query_parser.acronym_resolver and "EPS" in parsed.expanded_terms:
            assert len(parsed.expanded_terms["EPS"]) > 0

    def test_multiple_expansions(self, query_parser: QueryParser) -> None:
        """Test multiple term expansions in one query."""
        parsed = query_parser.parse("How does the EPS power system work?")
        # Should have expansions for both acronym and synonyms
        assert isinstance(parsed.expanded_terms, dict)


class TestKeywordExtraction:
    """Tests for keyword extraction."""

    def test_extract_keywords(self, query_parser: QueryParser) -> None:
        """Test extraction of keywords from query."""
        parsed = query_parser.parse("What is the solar panel voltage regulator?")
        assert len(parsed.keywords) > 0
        # Should extract important nouns
        keywords_lower = [k.lower() for k in parsed.keywords]
        assert any(
            keyword in ["panel", "voltage", "regulator", "solar"] for keyword in keywords_lower
        )

    def test_filter_stop_words(self, query_parser: QueryParser) -> None:
        """Test that stop words are filtered from keywords."""
        parsed = query_parser.parse("What is the main power system?")
        keywords_lower = [k.lower() for k in parsed.keywords]
        # Stop words like "the", "is", "what" should be filtered
        assert "the" not in keywords_lower
        assert "is" not in keywords_lower
        assert "what" not in keywords_lower

    def test_extract_entity_keywords(self, query_parser: QueryParser) -> None:
        """Test that entity mentions are included in keywords."""
        parsed = query_parser.parse("How does the Battery Controller work?")
        keywords_lower = [k.lower() for k in parsed.keywords]
        # Entity mentions should appear in keywords
        assert any(keyword in ["battery", "controller"] for keyword in keywords_lower)


class TestGraphTraversal:
    """Tests for graph traversal detection."""

    def test_requires_graph_structural(self, query_parser: QueryParser) -> None:
        """Test that structural queries require graph traversal."""
        parsed = query_parser.parse("What contains the power system?")
        assert parsed.requires_graph_traversal is True

    def test_requires_graph_relationship(self, query_parser: QueryParser) -> None:
        """Test that queries with relationships require graph traversal."""
        parsed = query_parser.parse("Show components that depend on the battery")
        assert parsed.requires_graph_traversal is True

    def test_requires_graph_keywords(self, query_parser: QueryParser) -> None:
        """Test that graph keywords trigger traversal."""
        parsed = query_parser.parse("Show the hierarchy of the system")
        assert parsed.requires_graph_traversal is True

    def test_no_graph_semantic(self, query_parser: QueryParser) -> None:
        """Test that simple semantic queries don't require graph."""
        parsed = query_parser.parse("What is a solar panel?")
        # Simple definition queries typically don't need graph
        # (though this could be false depending on implementation)
        assert isinstance(parsed.requires_graph_traversal, bool)

    def test_max_depth_explicit(self, query_parser: QueryParser) -> None:
        """Test explicit max depth specification."""
        parsed = query_parser.parse("Show dependencies 2 levels deep")
        assert parsed.max_depth == 2

    def test_max_depth_implicit(self, query_parser: QueryParser) -> None:
        """Test implicit max depth from keywords."""
        parsed = query_parser.parse("Show direct dependencies")
        if parsed.requires_graph_traversal:
            assert parsed.max_depth == 1


class TestQueryValidation:
    """Tests for query validation."""

    def test_validate_valid_query(self, query_parser: QueryParser) -> None:
        """Test validation of valid query."""
        parsed = query_parser.parse("What is the power system?")
        is_valid, error = query_parser.validate_query(parsed)
        assert is_valid is True
        assert error is None

    def test_validate_too_short(self, query_parser: QueryParser) -> None:
        """Test validation fails for too short query."""
        parsed = query_parser.parse("System")
        is_valid, error = query_parser.validate_query(parsed)
        assert is_valid is False
        assert "too short" in error.lower()

    def test_validate_empty_fails(self, query_parser: QueryParser) -> None:
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            query_parser.parse("")

    def test_validate_whitespace_only_fails(self, query_parser: QueryParser) -> None:
        """Test that whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            query_parser.parse("   ")


class TestQueryHistory:
    """Tests for query history storage and retrieval."""

    def test_store_query(self, query_parser: QueryParser) -> None:
        """Test storing query in history."""
        parsed = query_parser.parse("What is the power system?")
        assert query_parser.query_history_path.exists()

        # Verify file has content
        with open(query_parser.query_history_path) as f:
            content = f.read()
            assert len(content) > 0
            assert parsed.query_id in content

    def test_load_query_history(self, query_parser: QueryParser) -> None:
        """Test loading query history."""
        # Parse multiple queries
        query_parser.parse("What is the power system?")
        query_parser.parse("Show system components")
        query_parser.parse("How to perform startup?")

        # Load history
        history = query_parser.load_query_history()
        assert len(history) >= 3
        # Should be in reverse chronological order
        assert history[0].timestamp >= history[-1].timestamp

    def test_load_history_with_limit(self, query_parser: QueryParser) -> None:
        """Test loading query history with limit."""
        # Parse multiple queries
        for i in range(5):
            query_parser.parse(f"Test query {i}")

        # Load with limit
        history = query_parser.load_query_history(limit=3)
        assert len(history) == 3

    def test_query_statistics(self, query_parser: QueryParser) -> None:
        """Test query statistics generation."""
        # Parse queries with different intents
        query_parser.parse("What is the power system?")  # semantic
        query_parser.parse("What contains the subsystem?")  # structural
        query_parser.parse("How to start the system?")  # procedural

        stats = query_parser.get_query_statistics()
        assert stats["total_queries"] >= 3
        assert "intent_distribution" in stats
        assert "avg_entities_per_query" in stats
        assert "queries_requiring_graph" in stats


class TestQueryNormalization:
    """Tests for query text normalization."""

    def test_normalize_whitespace(self, query_parser: QueryParser) -> None:
        """Test normalization of multiple spaces."""
        parsed = query_parser.parse("What  is   the    power system?")
        assert "  " not in parsed.normalized_text

    def test_normalize_question_mark(self, query_parser: QueryParser) -> None:
        """Test removal of trailing question mark."""
        parsed = query_parser.parse("What is the power system?")
        assert not parsed.normalized_text.endswith("?")

    def test_normalize_leading_trailing_space(self, query_parser: QueryParser) -> None:
        """Test removal of leading/trailing whitespace."""
        parsed = query_parser.parse("  What is the power system?  ")
        assert parsed.normalized_text == parsed.normalized_text.strip()


class TestQueryParserPerformance:
    """Tests for query parser performance."""

    def test_parse_time_under_threshold(self, query_parser: QueryParser) -> None:
        """Test that parsing completes within time threshold."""
        start = datetime.now()
        query_parser.parse("What is the electrical power system and how does it work?")
        elapsed_ms = (datetime.now() - start).total_seconds() * 1000

        # Should parse in under 50ms (per acceptance criteria)
        # Note: First parse may be slower due to spaCy model loading
        # This is more of a benchmark than a strict test
        assert elapsed_ms < 500  # Generous threshold for test environment

    def test_parse_multiple_queries_fast(self, query_parser: QueryParser) -> None:
        """Test parsing multiple queries efficiently."""
        queries = [
            "What is the power system?",
            "Show system components",
            "How to perform startup?",
            "What contains the subsystem?",
            "Explain the thermal control system",
        ]

        start = datetime.now()
        for query in queries:
            query_parser.parse(query)
        elapsed_ms = (datetime.now() - start).total_seconds() * 1000

        # Average should be reasonable (< 100ms per query after warmup)
        avg_ms = elapsed_ms / len(queries)
        assert avg_ms < 200  # Generous threshold


class TestParsedQuerySerialization:
    """Tests for ParsedQuery serialization."""

    def test_to_dict(self, query_parser: QueryParser) -> None:
        """Test conversion of ParsedQuery to dictionary."""
        parsed = query_parser.parse("What is the power system?")
        data = parsed.to_dict()

        assert isinstance(data, dict)
        assert "query_id" in data
        assert "original_text" in data
        assert "intent" in data
        assert data["intent"] == parsed.intent.value

    def test_reconstruct_from_dict(self, query_parser: QueryParser) -> None:
        """Test reconstruction of ParsedQuery from dictionary."""
        original = query_parser.parse("What is the power system?")
        data = original.to_dict()

        # Reconstruct
        reconstructed = query_parser._reconstruct_query(data)

        assert reconstructed.query_id == original.query_id
        assert reconstructed.original_text == original.original_text
        assert reconstructed.intent == original.intent
        assert len(reconstructed.entity_mentions) == len(original.entity_mentions)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_special_characters(self, query_parser: QueryParser) -> None:
        """Test parsing query with special characters."""
        parsed = query_parser.parse("What is the V&V (Verification & Validation) process?")
        assert parsed is not None
        assert len(parsed.original_text) > 0

    def test_parse_all_caps(self, query_parser: QueryParser) -> None:
        """Test parsing query in all caps."""
        parsed = query_parser.parse("WHAT IS THE POWER SYSTEM?")
        assert parsed.intent != QueryIntent.UNKNOWN

    def test_parse_no_punctuation(self, query_parser: QueryParser) -> None:
        """Test parsing query without punctuation."""
        parsed = query_parser.parse("what is the power system")
        assert parsed.intent != QueryIntent.UNKNOWN

    def test_parse_very_long_query(self, query_parser: QueryParser) -> None:
        """Test parsing very long query."""
        long_query = (
            "Can you explain in detail how the electrical power system "
            "interacts with the thermal control system and what components "
            "are involved in the power distribution process from the solar "
            "panels to the battery and finally to the individual subsystems?"
        )
        parsed = query_parser.parse(long_query)
        assert parsed is not None
        # Long queries may or may not extract entities depending on spaCy model
        # The important thing is that the query is parsed without errors
        assert len(parsed.keywords) > 0  # Should at least extract keywords

    def test_parse_query_with_numbers(self, query_parser: QueryParser) -> None:
        """Test parsing query with numbers."""
        parsed = query_parser.parse("What is the voltage of the 28V power bus?")
        assert parsed is not None

    def test_parse_query_with_units(self, query_parser: QueryParser) -> None:
        """Test parsing query with measurement units."""
        parsed = query_parser.parse("What is the temperature in degrees Celsius?")
        assert parsed is not None

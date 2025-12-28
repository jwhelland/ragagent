"""Unit tests for the discovery pipeline core analysis (Task 3.9)."""

from __future__ import annotations

from src.curation.batch_operations import ApprovedEntityLookup
from src.pipeline.discovery_pipeline import (
    AutoResolvedEntity,
    DiscoveryCandidate,
    DiscoveryParameters,
    DiscoveryPipeline,
    cluster_cooccurrence_graph,
    compute_cooccurrence_edges,
    compute_entity_type_suggestions,
    generate_fuzzy_merge_suggestions,
)
from src.utils.config import NormalizationConfig


def test_cooccurrence_edges_count_and_pmi() -> None:
    candidates = [
        DiscoveryCandidate(
            candidate_key="A",
            canonical_name="Alpha",
            candidate_type="SYSTEM",
            chunk_ids=["c1", "c2"],
        ),
        DiscoveryCandidate(
            candidate_key="B",
            canonical_name="Beta",
            candidate_type="SYSTEM",
            chunk_ids=["c1"],
        ),
        DiscoveryCandidate(
            candidate_key="C",
            canonical_name="Gamma",
            candidate_type="SYSTEM",
            chunk_ids=["c2"],
        ),
    ]

    edges, chunk_freq, total_chunks = compute_cooccurrence_edges(
        candidates,
        min_cooccurrence=1,
        max_edges=10,
        max_entities_per_chunk=50,
    )

    assert total_chunks == 2
    assert chunk_freq["A"] == 2
    assert chunk_freq["B"] == 1
    assert chunk_freq["C"] == 1

    by_pair = {(edge.left_key, edge.right_key): edge for edge in edges}
    assert by_pair[("A", "B")].count == 1
    assert by_pair[("A", "C")].count == 1

    # PMI should be finite in this setup (non-zero probabilities).
    assert by_pair[("A", "B")].pmi > float("-inf")


def test_cooccurrence_clusters_connected_components() -> None:
    candidates = [
        DiscoveryCandidate(
            candidate_key="A",
            canonical_name="Alpha",
            candidate_type="SYSTEM",
            chunk_ids=["c1"],
        ),
        DiscoveryCandidate(
            candidate_key="B",
            canonical_name="Beta",
            candidate_type="SYSTEM",
            chunk_ids=["c1"],
        ),
        DiscoveryCandidate(
            candidate_key="C",
            canonical_name="Gamma",
            candidate_type="SYSTEM",
            chunk_ids=["c2"],
        ),
        DiscoveryCandidate(
            candidate_key="D",
            canonical_name="Delta",
            candidate_type="SYSTEM",
            chunk_ids=["c2"],
        ),
    ]

    edges, _, _ = compute_cooccurrence_edges(
        candidates,
        min_cooccurrence=1,
        max_edges=10,
        max_entities_per_chunk=50,
    )
    clusters = cluster_cooccurrence_graph(edges, min_edge_count=1, max_clusters=10)

    cluster_sets = [set(cluster.entity_keys) for cluster in clusters]
    assert {"A", "B"} in cluster_sets
    assert {"C", "D"} in cluster_sets


def test_entity_type_suggestions_ignores_known_types() -> None:
    candidates = [
        DiscoveryCandidate(
            candidate_key="A",
            canonical_name="Alpha",
            candidate_type="SYSTEM",
            conflicting_types=["SYSTEM", "NEW_KIND"],
        ),
        DiscoveryCandidate(
            candidate_key="B",
            canonical_name="Beta",
            candidate_type="SYSTEM",
            conflicting_types=["new_kind", "OTHER_KIND"],
        ),
    ]

    suggestions = compute_entity_type_suggestions(
        candidates,
        known_types=["SYSTEM", "COMPONENT"],
        top_k=10,
    )

    labels = {s.label for s in suggestions}
    assert "NEW_KIND" in labels
    assert "OTHER_KIND" in labels
    assert "SYSTEM" not in labels


def test_fuzzy_merge_suggestions_returns_high_confidence_pairs() -> None:
    config = NormalizationConfig(fuzzy_threshold=0.80)
    candidates = [
        DiscoveryCandidate(
            candidate_key="A",
            canonical_name="Attitude Control System",
            candidate_type="SYSTEM",
        ),
        DiscoveryCandidate(
            candidate_key="B",
            canonical_name="Attitude Control Subsystem",
            candidate_type="SYSTEM",
        ),
        DiscoveryCandidate(
            candidate_key="C",
            canonical_name="Thermal Control System",
            candidate_type="SYSTEM",
        ),
    ]

    suggestions = generate_fuzzy_merge_suggestions(
        candidates,
        config=config,
        max_suggestions=10,
        block_prefix=3,
    )

    assert any(
        suggestion.source_key in {"A", "B"} and suggestion.target_key in {"A", "B"}
        for suggestion in suggestions
    )
    assert all(suggestion.method == "fuzzy" for suggestion in suggestions)


def test_discovery_report_formatting() -> None:
    from src.pipeline.discovery_pipeline import (
        CooccurrenceEdge,
        DiscoveryMergeSuggestion,
        DiscoveryReport,
    )

    report = DiscoveryReport(
        totals={"candidates": 10, "chunks": 5},
        by_type=[{"candidate_type": "PERSON", "count": 7}, {"candidate_type": "ORG", "count": 3}],
        cooccurrence_edges=[CooccurrenceEdge(left_key="A", right_key="B", count=5, pmi=2.0)],
        merge_suggestions=[
            DiscoveryMergeSuggestion(
                method="fuzzy",
                source_key="A",
                target_key="C",
                entity_type="PERSON",
                score=0.9,
                confidence=0.8,
                reason="test",
            )
        ],
    )

    md = report.to_markdown(candidate_names={"A": "Alice", "B": "Bob", "C": "Alicia"})
    assert "# Entity Discovery Report" in md
    assert "Alice" in md
    assert "Bob" in md
    assert "Alicia" in md
    assert "PERSON" in md
    # Check ASCII bar
    assert "█" in md

    html = report.to_html(candidate_names={"A": "Alice", "B": "Bob", "C": "Alicia"})
    assert "<!DOCTYPE html>" in html
    assert "Alice" in html
    assert "Bob" in html
    assert "Alicia" in html
    assert "Co-occurrence Matrix" in html
    assert "background-color: rgba" in html


# --- Tests for Strategy 1: Pipeline-Level Pre-Resolution ---


def test_auto_resolved_entity_model() -> None:
    """Test AutoResolvedEntity model creation and serialization."""
    resolved = AutoResolvedEntity(
        candidate_key="system:battery_management_system",
        candidate_name="Battery Management System",
        candidate_type="SYSTEM",
        matched_entity_id="entity-123",
        match_reason="canonical_name",
    )

    assert resolved.candidate_key == "system:battery_management_system"
    assert resolved.candidate_name == "Battery Management System"
    assert resolved.candidate_type == "SYSTEM"
    assert resolved.matched_entity_id == "entity-123"
    assert resolved.match_reason == "canonical_name"

    # Test serialization
    data = resolved.model_dump()
    assert data["candidate_key"] == "system:battery_management_system"


def test_discovery_parameters_resolve_existing_default_false() -> None:
    """Test that resolve_existing defaults to False."""
    params = DiscoveryParameters()
    assert params.resolve_existing is False


def test_discovery_parameters_resolve_existing_can_be_set() -> None:
    """Test that resolve_existing can be set to True."""
    params = DiscoveryParameters(resolve_existing=True)
    assert params.resolve_existing is True


def test_discovery_report_with_auto_resolved_markdown() -> None:
    """Test that auto_resolved entities appear in markdown report."""
    from src.pipeline.discovery_pipeline import DiscoveryReport

    auto_resolved = [
        AutoResolvedEntity(
            candidate_key="system:bms",
            candidate_name="BMS",
            candidate_type="SYSTEM",
            matched_entity_id="entity-123",
            match_reason="alias:Battery Management System",
        ),
        AutoResolvedEntity(
            candidate_key="system:tcs",
            candidate_name="TCS",
            candidate_type="SYSTEM",
            matched_entity_id="entity-456",
            match_reason="canonical_name",
        ),
    ]

    report = DiscoveryReport(
        totals={"candidates": 5, "chunks": 3},
        auto_resolved=auto_resolved,
    )

    md = report.to_markdown()
    assert "## Auto-Resolved Entities" in md
    assert "**2** candidates were auto-resolved" in md
    assert "BMS" in md
    assert "TCS" in md
    assert "entity-123" in md
    assert "entity-456" in md
    assert "alias:Battery Management System" in md
    assert "canonical_name" in md


def test_discovery_report_with_auto_resolved_html() -> None:
    """Test that auto_resolved entities appear in HTML report."""
    from src.pipeline.discovery_pipeline import DiscoveryReport

    auto_resolved = [
        AutoResolvedEntity(
            candidate_key="system:bms",
            candidate_name="BMS",
            candidate_type="SYSTEM",
            matched_entity_id="entity-123",
            match_reason="canonical_name",
        ),
    ]

    report = DiscoveryReport(
        totals={"candidates": 5, "chunks": 3},
        auto_resolved=auto_resolved,
    )

    html = report.to_html()
    assert "<h2>Auto-Resolved Entities</h2>" in html
    assert "<strong>1</strong> candidates were auto-resolved" in html
    assert "<code>BMS</code>" in html
    assert "<code>entity-123</code>" in html


def test_discovery_report_without_auto_resolved_omits_section() -> None:
    """Test that auto_resolved section is omitted when empty."""
    from src.pipeline.discovery_pipeline import DiscoveryReport

    report = DiscoveryReport(
        totals={"candidates": 5, "chunks": 3},
        auto_resolved=[],
    )

    md = report.to_markdown()
    assert "Auto-Resolved Entities" not in md

    html = report.to_html()
    assert "Auto-Resolved Entities" not in html


def test_determine_match_reason_canonical_name() -> None:
    """Test _determine_match_reason returns 'canonical_name' when canonical name matches."""
    from unittest.mock import MagicMock

    from src.utils.config import Config

    config = MagicMock(spec=Config)
    pipeline = DiscoveryPipeline.__new__(DiscoveryPipeline)
    pipeline.config = config

    lookup = ApprovedEntityLookup()
    lookup.add_entity(
        entity_id="entity-123",
        canonical_name="Battery Management System",
        entity_type="SYSTEM",
        aliases=["BMS"],
    )

    reason = pipeline._determine_match_reason(
        canonical_name="Battery Management System",
        aliases=["BMS"],
        matched_entity_id="entity-123",
        entity_type="SYSTEM",
        lookup=lookup,
    )

    assert reason == "canonical_name"


def test_determine_match_reason_alias() -> None:
    """Test _determine_match_reason returns 'alias:...' when alias matches."""
    from unittest.mock import MagicMock

    from src.utils.config import Config

    config = MagicMock(spec=Config)
    pipeline = DiscoveryPipeline.__new__(DiscoveryPipeline)
    pipeline.config = config

    lookup = ApprovedEntityLookup()
    lookup.add_entity(
        entity_id="entity-123",
        canonical_name="Battery Management System",
        entity_type="SYSTEM",
        aliases=["BMS"],
    )

    # Candidate has different canonical name but matching alias
    reason = pipeline._determine_match_reason(
        canonical_name="Battery System",
        aliases=["BMS", "Battery Mgmt"],
        matched_entity_id="entity-123",
        entity_type="SYSTEM",
        lookup=lookup,
    )

    assert reason == "alias:BMS"


def test_determine_match_reason_unknown_fallback() -> None:
    """Test _determine_match_reason returns 'unknown' when no match reason found."""
    from unittest.mock import MagicMock

    from src.utils.config import Config

    config = MagicMock(spec=Config)
    pipeline = DiscoveryPipeline.__new__(DiscoveryPipeline)
    pipeline.config = config

    lookup = ApprovedEntityLookup()
    # Empty lookup - no matches possible
    lookup._lookup[("something else", "SYSTEM")] = "entity-123"
    lookup._entity_ids.add("entity-123")

    reason = pipeline._determine_match_reason(
        canonical_name="Battery Management System",
        aliases=["BMS"],
        matched_entity_id="entity-123",
        entity_type="SYSTEM",
        lookup=lookup,
    )

    assert reason == "unknown"


def test_load_candidates_filters_resolved_entities() -> None:
    """Test that _load_candidates filters out candidates matching approved entities."""
    from unittest.mock import MagicMock

    from src.utils.config import Config, DatabaseConfig

    # Create mock config
    config = MagicMock(spec=Config)
    config.database = DatabaseConfig()

    # Create pipeline
    pipeline = DiscoveryPipeline.__new__(DiscoveryPipeline)
    pipeline.config = config
    pipeline.neo4j_manager = MagicMock()

    # Mock get_entity_candidates to return candidates on first call, empty on subsequent calls
    test_candidates = [
        {
            "candidate_key": "system:bms",
            "canonical_name": "Battery Management System",
            "candidate_type": "SYSTEM",
            "status": "pending",
            "confidence_score": 0.9,
            "aliases": ["BMS"],
        },
        {
            "candidate_key": "system:tcs",
            "canonical_name": "Thermal Control System",
            "candidate_type": "SYSTEM",
            "status": "pending",
            "confidence_score": 0.8,
            "aliases": ["TCS"],
        },
        {
            "candidate_key": "system:power",
            "canonical_name": "Power System",
            "candidate_type": "SYSTEM",
            "status": "pending",
            "confidence_score": 0.7,
            "aliases": [],
        },
    ]
    # Return items on first call, empty list on subsequent calls to stop pagination
    pipeline.neo4j_manager.get_entity_candidates.side_effect = [test_candidates, []]

    # Create approved entity lookup with BMS
    lookup = ApprovedEntityLookup()
    lookup.add_entity(
        entity_id="entity-123",
        canonical_name="Battery Management System",
        entity_type="SYSTEM",
        aliases=["BMS"],
    )

    params = DiscoveryParameters(
        statuses=("pending",),
        resolve_existing=True,
    )

    candidates, auto_resolved = pipeline._load_candidates(params, lookup)

    # BMS should be auto-resolved, not in candidates
    assert len(candidates) == 2
    assert all(c.candidate_key != "system:bms" for c in candidates)
    assert any(c.candidate_key == "system:tcs" for c in candidates)
    assert any(c.candidate_key == "system:power" for c in candidates)

    # BMS should be in auto_resolved
    assert len(auto_resolved) == 1
    assert auto_resolved[0].candidate_key == "system:bms"
    assert auto_resolved[0].matched_entity_id == "entity-123"
    assert auto_resolved[0].match_reason == "canonical_name"


def test_load_candidates_without_lookup_returns_all() -> None:
    """Test that _load_candidates returns all candidates when no lookup provided."""
    from unittest.mock import MagicMock

    from src.utils.config import Config, DatabaseConfig

    config = MagicMock(spec=Config)
    config.database = DatabaseConfig()

    pipeline = DiscoveryPipeline.__new__(DiscoveryPipeline)
    pipeline.config = config
    pipeline.neo4j_manager = MagicMock()

    test_candidates = [
        {
            "candidate_key": "system:bms",
            "canonical_name": "Battery Management System",
            "candidate_type": "SYSTEM",
            "status": "pending",
            "confidence_score": 0.9,
            "aliases": ["BMS"],
        },
    ]
    # Return items on first call, empty list on subsequent calls to stop pagination
    pipeline.neo4j_manager.get_entity_candidates.side_effect = [test_candidates, []]

    params = DiscoveryParameters(statuses=("pending",))

    # No lookup provided (resolve_existing=False is default)
    candidates, auto_resolved = pipeline._load_candidates(params, None)

    assert len(candidates) == 1
    assert len(auto_resolved) == 0


def test_load_candidates_matches_by_alias() -> None:
    """Test that _load_candidates can match candidates by alias."""
    from unittest.mock import MagicMock

    from src.utils.config import Config, DatabaseConfig

    config = MagicMock(spec=Config)
    config.database = DatabaseConfig()

    pipeline = DiscoveryPipeline.__new__(DiscoveryPipeline)
    pipeline.config = config
    pipeline.neo4j_manager = MagicMock()

    # Candidate has alias that matches approved entity's canonical name
    test_candidates = [
        {
            "candidate_key": "system:battery_sys",
            "canonical_name": "Battery Sys",
            "candidate_type": "SYSTEM",
            "status": "pending",
            "confidence_score": 0.9,
            "aliases": ["Battery Management System"],  # This matches the approved entity
        },
    ]
    # Return items on first call, empty list on subsequent calls to stop pagination
    pipeline.neo4j_manager.get_entity_candidates.side_effect = [test_candidates, []]

    lookup = ApprovedEntityLookup()
    lookup.add_entity(
        entity_id="entity-123",
        canonical_name="Battery Management System",
        entity_type="SYSTEM",
        aliases=[],
    )

    params = DiscoveryParameters(statuses=("pending",))

    candidates, auto_resolved = pipeline._load_candidates(params, lookup)

    assert len(candidates) == 0
    assert len(auto_resolved) == 1
    assert auto_resolved[0].match_reason == "alias:Battery Management System"

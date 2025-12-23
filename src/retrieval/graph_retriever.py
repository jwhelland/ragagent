"""Graph-based retrieval using Cypher queries for relationship traversal (Task 4.3).

This module implements graph-based retrieval by traversing the Neo4j knowledge graph.
It supports:
- Entity resolution from query mentions to graph nodes
- Multi-hop relationship traversal (DEPENDS_ON, PART_OF, etc.)
- Hierarchical queries (CONTAINS paths)
- Sequential queries (PRECEDES chains)
- Procedural queries (REFERENCES)
- Path finding with depth limits
- Result scoring by graph distance and relevance
- Chunk extraction from graph results
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.retrieval.query_parser import EntityMention, ParsedQuery
from src.storage.neo4j_manager import Neo4jManager
from src.storage.schemas import EntityType, RelationshipType
from src.utils.config import Config, GraphSearchConfig


class TraversalStrategy(str, Enum):
    """Graph traversal strategy types."""

    MULTI_HOP = "multi_hop"  # Follow multiple relationship types
    HIERARCHICAL = "hierarchical"  # Follow CONTAINS/PART_OF hierarchy
    SEQUENTIAL = "sequential"  # Follow PRECEDES chains
    PROCEDURAL = "procedural"  # Follow REFERENCES for procedures
    SHORTEST_PATH = "shortest_path"  # Find shortest path between entities
    NEIGHBORS = "neighbors"  # Get immediate neighbors only


class GraphPath(BaseModel):
    """A path through the knowledge graph."""

    model_config = ConfigDict(extra="allow")

    start_entity_id: str = Field(..., description="Starting entity ID")
    end_entity_id: str = Field(..., description="Ending entity ID")
    nodes: List[Dict[str, Any]] = Field(..., description="Nodes in path")
    relationships: List[Dict[str, Any]] = Field(..., description="Relationships in path")
    length: int = Field(..., ge=0, description="Path length (number of relationships)")
    score: float = Field(..., ge=0.0, le=1.0, description="Path relevance score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Path confidence score")
    chunk_ids: Set[str] = Field(default_factory=set, description="Chunk IDs in path")


class ResolvedEntity(BaseModel):
    """Entity resolved from query mention."""

    model_config = ConfigDict(frozen=True)

    entity_id: str = Field(..., description="Resolved entity ID")
    canonical_name: str = Field(..., description="Entity canonical name")
    entity_type: EntityType = Field(..., description="Entity type")
    mention_text: str = Field(..., description="Original mention text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Resolution confidence")
    match_method: str = Field(..., description="How entity was matched (exact, fuzzy, search)")


class GraphRetrievalResult(BaseModel):
    """Result of graph-based retrieval."""

    model_config = ConfigDict(extra="allow")

    query_id: str = Field(..., description="Query identifier")
    query_text: str = Field(..., description="Original query text")
    resolved_entities: List[ResolvedEntity] = Field(
        default_factory=list, description="Entities resolved from query"
    )
    paths: List[GraphPath] = Field(default_factory=list, description="Graph paths found")
    chunk_ids: Set[str] = Field(default_factory=set, description="All chunk IDs from paths")
    entity_ids: Set[str] = Field(default_factory=set, description="All entity IDs from paths")
    strategy_used: TraversalStrategy = Field(..., description="Traversal strategy used")
    max_depth: int = Field(..., ge=0, description="Maximum traversal depth used")
    retrieval_time_ms: float = Field(..., ge=0.0, description="Retrieval time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Retrieval timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        data["strategy_used"] = self.strategy_used.value
        data["chunk_ids"] = list(self.chunk_ids)
        data["entity_ids"] = list(self.entity_ids)
        return data


class GraphRetriever:
    """Graph-based retriever using Neo4j Cypher queries."""

    def __init__(
        self,
        config: Optional[Config] = None,
        neo4j_manager: Optional[Neo4jManager] = None,
    ) -> None:
        """Initialize graph retriever.

        Args:
            config: Configuration object
            neo4j_manager: Neo4j database manager (created if None)
        """
        self.config = config or Config.from_yaml()
        self.graph_config: GraphSearchConfig = self.config.retrieval.graph_search

        # Initialize Neo4j manager
        if neo4j_manager is None:
            self.neo4j = Neo4jManager(config=self.config.database)
            self.neo4j.connect()
        else:
            self.neo4j = neo4j_manager

        # Cache existing relationship types to avoid Neo4j warnings
        try:
            self.existing_relationship_types = set(self.neo4j.get_existing_relationship_types())
        except Exception as e:
            logger.warning(f"Failed to fetch existing relationship types: {e}")
            self.existing_relationship_types = set()

        logger.info(
            "Initialized GraphRetriever",
            max_depth=self.graph_config.max_depth,
            relationship_types=len(self.graph_config.relationship_types),
            existing_types=len(self.existing_relationship_types),
        )

    def _filter_relationship_types(
        self, relationship_types: List[RelationshipType]
    ) -> List[RelationshipType]:
        """Filter relationship types to only those that exist in the database.

        Args:
            relationship_types: List of desired relationship types

        Returns:
            List of relationship types that actually exist
        """
        if not self.existing_relationship_types:
            # If we couldn't fetch types (e.g. empty DB), allow all to avoid blocking
            # But normally this set is populated.
            # Actually, if the set is empty, it means NO relationships exist, so we should return empty.
            # However, for safety if the fetch failed, we might want to return original.
            # But the fetch is wrapped in try/except.
            # If the DB is truly empty, returning empty list is correct.
            # If the DB has types but we missed them, returning original risks warnings.
            # Let's assume if it's empty, it's empty.
            return []

        filtered = [
            rt for rt in relationship_types if rt.value in self.existing_relationship_types
        ]
        
        if len(filtered) < len(relationship_types):
            logger.debug(
                "Filtered relationship types",
                original=len(relationship_types),
                filtered=len(filtered),
                excluded=[rt.value for rt in relationship_types if rt.value not in self.existing_relationship_types]
            )
            
        return filtered

    def retrieve(
        self,
        query: str | ParsedQuery,
        max_depth: Optional[int] = None,
        relationship_types: Optional[List[RelationshipType]] = None,
        strategy: Optional[TraversalStrategy] = None,
    ) -> GraphRetrievalResult:
        """Retrieve relevant graph paths for a query.

        Args:
            query: Query string or ParsedQuery object
            max_depth: Maximum traversal depth (default from config)
            relationship_types: Relationship types to traverse (default from config)
            strategy: Traversal strategy (auto-selected if None)

        Returns:
            GraphRetrievalResult with paths and entities

        Raises:
            ValueError: If query is empty or invalid
        """
        start_time = datetime.now()

        # Parse query if string
        if isinstance(query, str):
            raise ValueError(
                "GraphRetriever requires a ParsedQuery object. "
                "Use QueryParser to parse the query first."
            )

        parsed_query = query

        # Use config defaults if not specified
        max_depth = max_depth or self.graph_config.max_depth
        relationship_types = relationship_types or [
            RelationshipType(rt) for rt in self.graph_config.relationship_types
        ]

        # Select traversal strategy
        if strategy is None:
            strategy = self._select_strategy(parsed_query, relationship_types)

        # Resolve entities from query
        resolved_entities = self._resolve_entities(parsed_query.entity_mentions)

        # If no entities resolved, return empty result
        if not resolved_entities:
            logger.warning(
                "No entities resolved from query",
                query_id=parsed_query.query_id,
                num_mentions=len(parsed_query.entity_mentions),
            )
            retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
            return GraphRetrievalResult(
                query_id=parsed_query.query_id,
                query_text=parsed_query.original_text,
                resolved_entities=[],
                paths=[],
                chunk_ids=set(),
                entity_ids=set(),
                strategy_used=strategy,
                max_depth=max_depth,
                retrieval_time_ms=retrieval_time,
            )

        # Execute graph traversal based on strategy
        paths = self._execute_traversal(
            resolved_entities=resolved_entities,
            strategy=strategy,
            max_depth=max_depth,
            relationship_types=relationship_types,
        )

        # Extract chunk and entity IDs from paths
        all_chunk_ids: Set[str] = set()
        all_entity_ids: Set[str] = set()
        for path in paths:
            all_chunk_ids.update(path.chunk_ids)
            all_entity_ids.add(path.start_entity_id)
            all_entity_ids.add(path.end_entity_id)
            # Extract entity IDs from nodes
            for node in path.nodes:
                if "id" in node:
                    all_entity_ids.add(node["id"])

        # Calculate retrieval time
        retrieval_time = (datetime.now() - start_time).total_seconds() * 1000

        result = GraphRetrievalResult(
            query_id=parsed_query.query_id,
            query_text=parsed_query.original_text,
            resolved_entities=resolved_entities,
            paths=paths,
            chunk_ids=all_chunk_ids,
            entity_ids=all_entity_ids,
            strategy_used=strategy,
            max_depth=max_depth,
            retrieval_time_ms=retrieval_time,
        )

        logger.info(
            "Graph retrieval completed",
            query_id=parsed_query.query_id,
            num_resolved_entities=len(resolved_entities),
            num_paths=len(paths),
            num_chunks=len(all_chunk_ids),
            strategy=strategy.value,
            retrieval_time_ms=round(retrieval_time, 2),
        )

        return result

    def _resolve_entities(self, entity_mentions: List[EntityMention]) -> List[ResolvedEntity]:
        """Resolve entity mentions to graph nodes.

        Args:
            entity_mentions: List of entity mentions from query

        Returns:
            List of resolved entities with confidence scores
        """
        resolved: List[ResolvedEntity] = []
        seen_ids: Set[str] = set()

        for mention in entity_mentions:
            # Try exact match on canonical name first
            entity_dict = self.neo4j.get_entity_by_canonical_name(
                canonical_name=mention.normalized,
                entity_type=mention.entity_type,
            )

            if entity_dict:
                if entity_dict["id"] not in seen_ids:
                    resolved.append(
                        ResolvedEntity(
                            entity_id=entity_dict["id"],
                            canonical_name=entity_dict["canonical_name"],
                            entity_type=EntityType(
                                entity_dict.get("entity_type", EntityType.CONCEPT.value)
                            ),
                            mention_text=mention.text,
                            confidence=0.95,
                            match_method="exact",
                        )
                    )
                    seen_ids.add(entity_dict["id"])
                continue

            # Try full-text search if exact match fails
            search_results = self.neo4j.search_entities(
                query=mention.text,
                entity_types=[mention.entity_type] if mention.entity_type else None,
                limit=3,
            )

            # Take best match with sufficient score
            for result in search_results:
                if result.get("search_score", 0.0) >= 0.5:
                    if result["id"] not in seen_ids:
                        resolved.append(
                            ResolvedEntity(
                                entity_id=result["id"],
                                canonical_name=result["canonical_name"],
                                entity_type=EntityType(
                                    result.get("entity_type", EntityType.CONCEPT.value)
                                ),
                                mention_text=mention.text,
                                confidence=min(0.9, result["search_score"]),
                                match_method="search",
                            )
                        )
                        seen_ids.add(result["id"])
                        break

        logger.debug(
            "Resolved entities",
            num_mentions=len(entity_mentions),
            num_resolved=len(resolved),
        )

        return resolved

    def _select_strategy(
        self, parsed_query: ParsedQuery, relationship_types: List[RelationshipType]
    ) -> TraversalStrategy:
        """Select appropriate traversal strategy based on query.

        Args:
            parsed_query: Parsed query object
            relationship_types: Relationship types to consider

        Returns:
            Selected traversal strategy
        """
        # Check query intent and relationship types
        if RelationshipType.PRECEDES in relationship_types:
            return TraversalStrategy.SEQUENTIAL

        if RelationshipType.REFERENCES in relationship_types:
            return TraversalStrategy.PROCEDURAL

        if any(
            rt in relationship_types for rt in [RelationshipType.CONTAINS, RelationshipType.PART_OF]
        ):
            return TraversalStrategy.HIERARCHICAL

        # If multiple entities, try shortest path
        if len(parsed_query.entity_mentions) >= 2 and self.graph_config.enable_shortest_path:
            return TraversalStrategy.SHORTEST_PATH

        # Default to multi-hop traversal
        return TraversalStrategy.MULTI_HOP

    def _execute_traversal(
        self,
        resolved_entities: List[ResolvedEntity],
        strategy: TraversalStrategy,
        max_depth: int,
        relationship_types: List[RelationshipType],
    ) -> List[GraphPath]:
        """Execute graph traversal based on strategy.

        Args:
            resolved_entities: Resolved entities to start from
            strategy: Traversal strategy to use
            max_depth: Maximum traversal depth
            relationship_types: Relationship types to traverse

        Returns:
            List of graph paths
        """
        if strategy == TraversalStrategy.SHORTEST_PATH:
            return self._find_shortest_paths(resolved_entities, max_depth, relationship_types)
        elif strategy == TraversalStrategy.HIERARCHICAL:
            return self._traverse_hierarchical(resolved_entities, max_depth)
        elif strategy == TraversalStrategy.SEQUENTIAL:
            return self._traverse_sequential(resolved_entities, max_depth)
        elif strategy == TraversalStrategy.PROCEDURAL:
            return self._traverse_procedural(resolved_entities, max_depth)
        elif strategy == TraversalStrategy.NEIGHBORS:
            return self._get_neighbors(resolved_entities, relationship_types)
        else:  # MULTI_HOP
            return self._traverse_multi_hop(resolved_entities, max_depth, relationship_types)

    def _find_shortest_paths(
        self,
        resolved_entities: List[ResolvedEntity],
        max_depth: int,
        relationship_types: List[RelationshipType],
    ) -> List[GraphPath]:
        """Find shortest paths between resolved entities.

        Args:
            resolved_entities: Resolved entities
            max_depth: Maximum path length
            relationship_types: Relationship types to traverse

        Returns:
            List of shortest paths
        """
        paths: List[GraphPath] = []
        filtered_types = self._filter_relationship_types(relationship_types)

        if not filtered_types:
            return []

        # Find paths between all pairs of entities
        for i, source in enumerate(resolved_entities):
            for target in resolved_entities[i + 1 :]:
                raw_paths = self.neo4j.find_path(
                    source_id=source.entity_id,
                    target_id=target.entity_id,
                    max_depth=max_depth,
                    relationship_types=filtered_types,
                )

                for raw_path in raw_paths:
                    path = self._create_graph_path(
                        start_entity_id=source.entity_id,
                        end_entity_id=target.entity_id,
                        raw_path=raw_path,
                        source_confidence=source.confidence,
                        target_confidence=target.confidence,
                    )
                    paths.append(path)

        # Sort by score (descending)
        paths.sort(key=lambda p: p.score, reverse=True)

        return paths

    def _traverse_hierarchical(
        self, resolved_entities: List[ResolvedEntity], max_depth: int
    ) -> List[GraphPath]:
        """Traverse hierarchical relationships (CONTAINS, PART_OF).

        Args:
            resolved_entities: Resolved entities
            max_depth: Maximum traversal depth

        Returns:
            List of hierarchical paths
        """
        paths: List[GraphPath] = []
        
        part_of_rels = self._filter_relationship_types([RelationshipType.PART_OF])
        contains_rels = self._filter_relationship_types([RelationshipType.CONTAINS])

        for entity in resolved_entities:
            # Traverse upward (PART_OF)
            if part_of_rels:
                upward_entities = self.neo4j.traverse_relationships(
                    entity_id=entity.entity_id,
                    relationship_types=part_of_rels,
                    max_depth=max_depth,
                    direction="outgoing",
                )

                for related in upward_entities:
                    path = self._create_simple_path(
                        start_entity_id=entity.entity_id,
                        end_entity_id=related["id"],
                        depth=related["depth"],
                        base_confidence=entity.confidence,
                    )
                    paths.append(path)

            # Traverse downward (CONTAINS)
            if contains_rels:
                downward_entities = self.neo4j.traverse_relationships(
                    entity_id=entity.entity_id,
                    relationship_types=contains_rels,
                    max_depth=max_depth,
                    direction="outgoing",
                )

                for related in downward_entities:
                    path = self._create_simple_path(
                        start_entity_id=entity.entity_id,
                        end_entity_id=related["id"],
                        depth=related["depth"],
                        base_confidence=entity.confidence,
                    )
                    paths.append(path)

        # Sort by score
        paths.sort(key=lambda p: p.score, reverse=True)

        return paths

    def _traverse_sequential(
        self, resolved_entities: List[ResolvedEntity], max_depth: int
    ) -> List[GraphPath]:
        """Traverse sequential relationships (PRECEDES).

        Args:
            resolved_entities: Resolved entities
            max_depth: Maximum traversal depth

        Returns:
            List of sequential paths
        """
        paths: List[GraphPath] = []
        precedes_rels = self._filter_relationship_types([RelationshipType.PRECEDES])

        if not precedes_rels:
            return []

        for entity in resolved_entities:
            # Follow PRECEDES chains (forward)
            forward_entities = self.neo4j.traverse_relationships(
                entity_id=entity.entity_id,
                relationship_types=precedes_rels,
                max_depth=max_depth,
                direction="outgoing",
            )

            for related in forward_entities:
                path = self._create_simple_path(
                    start_entity_id=entity.entity_id,
                    end_entity_id=related["id"],
                    depth=related["depth"],
                    base_confidence=entity.confidence,
                )
                paths.append(path)

            # Follow PRECEDES chains (backward)
            backward_entities = self.neo4j.traverse_relationships(
                entity_id=entity.entity_id,
                relationship_types=precedes_rels,
                max_depth=max_depth,
                direction="incoming",
            )

            for related in backward_entities:
                path = self._create_simple_path(
                    start_entity_id=related["id"],
                    end_entity_id=entity.entity_id,
                    depth=related["depth"],
                    base_confidence=entity.confidence,
                )
                paths.append(path)

        # Sort by score
        paths.sort(key=lambda p: p.score, reverse=True)

        return paths

    def _traverse_procedural(
        self, resolved_entities: List[ResolvedEntity], max_depth: int
    ) -> List[GraphPath]:
        """Traverse procedural relationships (REFERENCES, REQUIRES_CHECK).

        Args:
            resolved_entities: Resolved entities
            max_depth: Maximum traversal depth

        Returns:
            List of procedural paths
        """
        paths: List[GraphPath] = []

        procedural_rels = [
            RelationshipType.REFERENCES,
            RelationshipType.REQUIRES_CHECK,
            RelationshipType.PRECEDES,
        ]
        filtered_rels = self._filter_relationship_types(procedural_rels)

        if not filtered_rels:
            return []

        for entity in resolved_entities:
            related_entities = self.neo4j.traverse_relationships(
                entity_id=entity.entity_id,
                relationship_types=filtered_rels,
                max_depth=max_depth,
                direction="both",
            )

            for related in related_entities:
                path = self._create_simple_path(
                    start_entity_id=entity.entity_id,
                    end_entity_id=related["id"],
                    depth=related["depth"],
                    base_confidence=entity.confidence,
                )
                paths.append(path)

        # Sort by score
        paths.sort(key=lambda p: p.score, reverse=True)

        return paths

    def _traverse_multi_hop(
        self,
        resolved_entities: List[ResolvedEntity],
        max_depth: int,
        relationship_types: List[RelationshipType],
    ) -> List[GraphPath]:
        """Traverse multiple relationship types.

        Args:
            resolved_entities: Resolved entities
            max_depth: Maximum traversal depth
            relationship_types: Relationship types to traverse

        Returns:
            List of multi-hop paths
        """
        paths: List[GraphPath] = []
        filtered_types = self._filter_relationship_types(relationship_types)

        if not filtered_types:
            return []

        for entity in resolved_entities:
            related_entities = self.neo4j.traverse_relationships(
                entity_id=entity.entity_id,
                relationship_types=filtered_types,
                max_depth=max_depth,
                direction="both",
            )

            for related in related_entities:
                path = self._create_simple_path(
                    start_entity_id=entity.entity_id,
                    end_entity_id=related["id"],
                    depth=related["depth"],
                    base_confidence=entity.confidence,
                )
                paths.append(path)

        # Sort by score
        paths.sort(key=lambda p: p.score, reverse=True)

        return paths

    def _get_neighbors(
        self, resolved_entities: List[ResolvedEntity], relationship_types: List[RelationshipType]
    ) -> List[GraphPath]:
        """Get immediate neighbors of resolved entities.

        Args:
            resolved_entities: Resolved entities
            relationship_types: Relationship types to consider

        Returns:
            List of single-hop paths to neighbors
        """
        paths: List[GraphPath] = []

        for entity in resolved_entities:
            # Get all relationships for entity
            relationships = self.neo4j.get_relationships(
                entity_id=entity.entity_id,
                relationship_type=None,  # Get all types
                direction="both",
            )

            # Filter by relationship types
            for rel in relationships:
                rel_type_str = rel.get("type", "")
                try:
                    rel_type = RelationshipType(rel_type_str)
                    if rel_type in relationship_types:
                        # Determine neighbor
                        if rel["source_entity_id"] == entity.entity_id:
                            neighbor_id = rel["target_entity_id"]
                        else:
                            neighbor_id = rel["source_entity_id"]

                        path = self._create_simple_path(
                            start_entity_id=entity.entity_id,
                            end_entity_id=neighbor_id,
                            depth=1,
                            base_confidence=entity.confidence,
                        )
                        paths.append(path)
                except ValueError:
                    # Unknown relationship type, skip
                    continue

        # Sort by score
        paths.sort(key=lambda p: p.score, reverse=True)

        return paths

    def _create_graph_path(
        self,
        start_entity_id: str,
        end_entity_id: str,
        raw_path: Dict[str, Any],
        source_confidence: float,
        target_confidence: float,
    ) -> GraphPath:
        """Create GraphPath from Neo4j path result.

        Args:
            start_entity_id: Starting entity ID
            end_entity_id: Ending entity ID
            raw_path: Raw path data from Neo4j
            source_confidence: Source entity confidence
            target_confidence: Target entity confidence

        Returns:
            GraphPath object with scoring
        """
        nodes = raw_path.get("nodes", [])
        relationships = raw_path.get("relationships", [])
        length = raw_path.get("length", len(relationships))

        # Extract chunk IDs from nodes
        chunk_ids: Set[str] = set()
        # Note: chunk IDs would need to be linked separately via entity-to-chunk relationships
        # For now, we'll extract them from node properties if available
        for node in nodes:
            if "chunk_ids" in node:
                chunk_ids.update(node.get("chunk_ids", []))

        # Calculate path score
        score = self._calculate_path_score(
            length=length,
            source_confidence=source_confidence,
            target_confidence=target_confidence,
        )

        # Calculate path confidence (average of node confidences)
        node_confidences = [node.get("confidence_score", 0.5) for node in nodes]
        avg_confidence = sum(node_confidences) / len(node_confidences) if node_confidences else 0.5

        return GraphPath(
            start_entity_id=start_entity_id,
            end_entity_id=end_entity_id,
            nodes=nodes,
            relationships=relationships,
            length=length,
            score=score,
            confidence=avg_confidence,
            chunk_ids=chunk_ids,
        )

    def _create_simple_path(
        self,
        start_entity_id: str,
        end_entity_id: str,
        depth: int,
        base_confidence: float,
    ) -> GraphPath:
        """Create a simple path from traversal result.

        Args:
            start_entity_id: Starting entity ID
            end_entity_id: Ending entity ID
            depth: Path depth (graph distance)
            base_confidence: Base confidence from query resolution

        Returns:
            GraphPath object
        """
        # Get entity details
        start_entity = self.neo4j.get_entity(start_entity_id)
        end_entity = self.neo4j.get_entity(end_entity_id)

        nodes = []
        if start_entity:
            nodes.append(start_entity)
        if end_entity:
            nodes.append(end_entity)

        # Calculate score based on depth
        score = self._calculate_path_score(
            length=depth,
            source_confidence=base_confidence,
            target_confidence=end_entity.get("confidence_score", 0.5) if end_entity else 0.5,
        )

        # Extract chunk IDs (simplified - would need proper implementation)
        chunk_ids: Set[str] = set()

        return GraphPath(
            start_entity_id=start_entity_id,
            end_entity_id=end_entity_id,
            nodes=nodes,
            relationships=[],  # Would need to fetch actual relationships
            length=depth,
            score=score,
            confidence=base_confidence * 0.9,  # Slight penalty for distance
            chunk_ids=chunk_ids,
        )

    def _calculate_path_score(
        self, length: int, source_confidence: float, target_confidence: float
    ) -> float:
        """Calculate path relevance score.

        Score is based on:
        - Path length (shorter is better)
        - Source entity confidence
        - Target entity confidence

        Args:
            length: Path length (number of hops)
            source_confidence: Source entity resolution confidence
            target_confidence: Target entity resolution confidence

        Returns:
            Path score between 0 and 1
        """
        # Distance penalty (exponential decay)
        distance_score = 1.0 / (1.0 + length)

        # Entity confidence contribution
        entity_confidence = (source_confidence + target_confidence) / 2.0

        # Combined score (weighted average)
        score = 0.6 * distance_score + 0.4 * entity_confidence

        return min(1.0, max(0.0, score))

    def get_chunks_for_entities(self, entity_ids: Set[str]) -> List[Dict[str, Any]]:
        """Get chunks that mention the given entities.

        Args:
            entity_ids: Set of entity IDs

        Returns:
            List of chunk dictionaries with metadata
        """
        if not entity_ids:
            return []

        # Query to find chunks that contain these entities
        # Note: This assumes entity_ids are stored in chunk metadata
        query = """
        MATCH (c:Chunk)
        WHERE any(eid IN $entity_ids WHERE eid IN c.entity_ids)
        RETURN DISTINCT c
        ORDER BY c.level, c.hierarchy_path
        LIMIT 100
        """

        try:
            results = self.neo4j.execute_cypher(query, {"entity_ids": list(entity_ids)})
            chunks = []
            for record in results:
                if "c" in record:
                    chunks.append(record["c"])
            return chunks
        except Exception as e:
            logger.warning(f"Failed to get chunks for entities: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph retrieval statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            stats = self.neo4j.get_statistics()
            # Convert relationship types to strings (may be RelationshipType enums or strings from config)
            rel_types = [
                rt.value if isinstance(rt, RelationshipType) else rt
                for rt in self.graph_config.relationship_types
            ]
            stats["graph_config"] = {
                "max_depth": self.graph_config.max_depth,
                "relationship_types": rel_types,
                "enable_shortest_path": self.graph_config.enable_shortest_path,
            }
            return stats
        except Exception as e:
            logger.warning(f"Failed to get statistics: {e}")
            return {}

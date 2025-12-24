"""Statistical co-occurrence relationship extractor."""

from itertools import combinations
from typing import Any, Dict, List, Optional

from loguru import logger

from src.extraction.models import ExtractedRelationship


class CooccurrenceRelationshipExtractor:
    """Extracts relationships based on entity co-occurrence in the same context."""

    def __init__(self, confidence: float = 0.4) -> None:
        self.confidence = confidence
        logger.info(f"Initialized CooccurrenceRelationshipExtractor (confidence={confidence})")

    def extract_relationships(
        self,
        chunk: Any,
        *,
        known_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ExtractedRelationship]:
        """Create RELATED_TO relationships between all pairs of entities in a chunk."""
        if not known_entities or len(known_entities) < 2:
            return []

        chunk_id = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None) or (chunk.get("chunk_id") if isinstance(chunk, dict) else None)
        document_id = getattr(chunk, "document_id", None) or (chunk.get("document_id") if isinstance(chunk, dict) else None)

        # Use canonical names or names to identify entities
        entity_names = set()
        for ent in known_entities:
            name = ""
            if isinstance(ent, dict):
                name = str(ent.get("name") or ent.get("text") or ent.get("canonical_name") or "")
            else:
                # Assume ExtractedEntity or similar object with name/text/canonical_name
                name = str(getattr(ent, "name", None) or getattr(ent, "text", None) or getattr(ent, "canonical_name", ""))
            
            if name.strip():
                entity_names.add(name.strip())

        # combinatorial explosion protection: if too many entities, they probably aren't all "related"
        # e.g. a list of parts. 8 entities = 28 pairs. 
        if len(entity_names) < 2 or len(entity_names) > 8:
            return []

        relationships: List[ExtractedRelationship] = []
        
        # Create a relationship for every unique pair (undirected, so we just do one way)
        # We sort to ensure deterministic pairs
        sorted_names = sorted(list(entity_names))
        
        for source, target in combinations(sorted_names, 2):
            relationships.append(
                ExtractedRelationship(
                    source=source,
                    target=target,
                    type="RELATED_TO",
                    description=f"Extracted via chunk co-occurrence",
                    confidence=self.confidence,
                    bidirectional=True,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source_extractor="cooccurrence",
                )
            )

        return relationships

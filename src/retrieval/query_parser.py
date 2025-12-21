"""Query parser and intent detection for retrieval system (Task 4.1).

This module provides query parsing functionality including:
- Entity mention extraction from natural language queries
- Query intent classification (semantic, structural, procedural, hybrid)
- Query parameter extraction (filters, constraints, metadata)
- Query expansion using synonyms and acronyms
- Query validation and normalization
- Query history storage for analysis
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import spacy
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from spacy.language import Language
from spacy.tokens import Doc

from src.normalization.acronym_resolver import AcronymResolver
from src.normalization.string_normalizer import StringNormalizer
from src.storage.schemas import EntityType, RelationshipType
from src.utils.config import Config, RetrievalConfig


class QueryIntent(str, Enum):
    """Query intent classification types."""

    SEMANTIC = "semantic"  # What is X? How does X work? Explain Y
    STRUCTURAL = "structural"  # What contains X? What is part of Y?
    PROCEDURAL = "procedural"  # How to do X? What are steps for Y?
    HYBRID = "hybrid"  # Combination of multiple intents
    UNKNOWN = "unknown"  # Cannot determine intent


class QueryConstraint(BaseModel):
    """Constraint or filter extracted from query."""

    model_config = ConfigDict(frozen=True)

    field: str = Field(..., description="Field to filter on")
    operator: str = Field(..., description="Comparison operator (eq, gt, lt, contains)")
    value: Any = Field(..., description="Filter value")
    negated: bool = Field(default=False, description="Whether constraint is negated")


class EntityMention(BaseModel):
    """Entity mention extracted from query text."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., description="Original mention text")
    normalized: str = Field(..., description="Normalized form")
    entity_type: Optional[EntityType] = Field(None, description="Detected entity type")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


class ParsedQuery(BaseModel):
    """Structured representation of parsed query."""

    model_config = ConfigDict(extra="allow")

    query_id: str = Field(..., description="Unique query identifier")
    original_text: str = Field(..., description="Original query text")
    normalized_text: str = Field(..., description="Normalized query text")
    intent: QueryIntent = Field(..., description="Detected query intent")
    intent_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Intent classification confidence"
    )
    entity_mentions: List[EntityMention] = Field(
        default_factory=list, description="Entity mentions in query"
    )
    relationship_types: List[RelationshipType] = Field(
        default_factory=list, description="Relationship types mentioned"
    )
    constraints: List[QueryConstraint] = Field(
        default_factory=list, description="Query constraints/filters"
    )
    expanded_terms: Dict[str, List[str]] = Field(
        default_factory=dict, description="Term expansions (synonyms, acronyms)"
    )
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    requires_graph_traversal: bool = Field(
        default=False, description="Whether query requires graph operations"
    )
    max_depth: Optional[int] = Field(None, description="Maximum traversal depth if applicable")
    timestamp: datetime = Field(default_factory=datetime.now, description="Query timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        data["intent"] = self.intent.value
        data["timestamp"] = self.timestamp.isoformat()
        data["entity_mentions"] = [
            {
                "text": m.text,
                "normalized": m.normalized,
                "entity_type": m.entity_type.value if m.entity_type else None,
                "start_char": m.start_char,
                "end_char": m.end_char,
                "confidence": m.confidence,
            }
            for m in self.entity_mentions
        ]
        data["relationship_types"] = [rt.value for rt in self.relationship_types]
        data["constraints"] = [c.model_dump() for c in self.constraints]
        return data


class QueryParser:
    """Query parser with intent detection and entity extraction."""

    # Intent classification patterns (keywords and phrases)
    _INTENT_PATTERNS: Dict[QueryIntent, List[str]] = {
        QueryIntent.SEMANTIC: [
            r"\bwhat is\b",
            r"\bwhat are\b",
            r"\bdefine\b",
            r"\bexplain\b",
            r"\bdescribe\b",
            r"\bhow does .* work\b",
            r"\bwhy\b",
            r"\bfunction of\b",
            r"\bpurpose of\b",
            r"\btell me about\b",
        ],
        QueryIntent.STRUCTURAL: [
            r"\bcontain\b",
            r"\bpart of\b",
            r"\bcomponent[s]?\b",
            r"\bsubsystem[s]?\b",
            r"\bmade of\b",
            r"\bconsist[s]? of\b",
            r"\binclude[s]?\b",
            r"\bbelongs? to\b",
            r"\bhierarchy\b",
            r"\bstructure\b",
            r"\bcomposed of\b",
        ],
        QueryIntent.PROCEDURAL: [
            r"\bhow to\b",
            r"\bsteps?\b",
            r"\bprocedure[s]?\b",
            r"\bprocess\b",
            r"\binstructions?\b",
            r"\bperform\b",
            r"\bexecute\b",
            r"\bcarry out\b",
            r"\bsequence\b",
            r"\border of\b",
        ],
    }

    # Constraint extraction patterns
    _CONSTRAINT_PATTERNS: List[Tuple[re.Pattern[str], str, str]] = [
        (re.compile(r"\bgreater than (\d+(?:\.\d+)?)\b", re.I), "gt", "numeric"),
        (re.compile(r"\bless than (\d+(?:\.\d+)?)\b", re.I), "lt", "numeric"),
        (re.compile(r"\bequals? (\w+)\b", re.I), "eq", "string"),
        (re.compile(r"\bin (?:the )?(\w+)\b", re.I), "contains", "string"),
        (re.compile(r"\bfrom (?:the )?(\w+)\b", re.I), "source", "string"),
        (re.compile(r"\b(?:in|on) page (\d+)\b", re.I), "page", "numeric"),
        (re.compile(r"\b(?:in|from) document (\w+)\b", re.I), "document", "string"),
    ]

    # Relationship keywords mapping (includes variations for verb conjugations)
    _RELATIONSHIP_KEYWORDS: Dict[str, RelationshipType] = {
        "part of": RelationshipType.PART_OF,
        "contains": RelationshipType.CONTAINS,
        "contain": RelationshipType.CONTAINS,
        "contained": RelationshipType.CONTAINS,
        "depends on": RelationshipType.DEPENDS_ON,
        "depend on": RelationshipType.DEPENDS_ON,
        "dependent": RelationshipType.DEPENDS_ON,
        "dependency": RelationshipType.DEPENDS_ON,
        "dependencies": RelationshipType.DEPENDS_ON,
        "controls": RelationshipType.CONTROLS,
        "control": RelationshipType.CONTROLS,
        "monitors": RelationshipType.MONITORS,
        "monitor": RelationshipType.MONITORS,
        "power": RelationshipType.PROVIDES_POWER_TO,
        "sends data": RelationshipType.SENDS_DATA_TO,
        "data flow": RelationshipType.SENDS_DATA_TO,
        "references": RelationshipType.REFERENCES,
        "reference": RelationshipType.REFERENCES,
        "precedes": RelationshipType.PRECEDES,
        "precede": RelationshipType.PRECEDES,
        "before": RelationshipType.PRECEDES,
        "after": RelationshipType.PRECEDES,
        "affects": RelationshipType.AFFECTS,
        "affect": RelationshipType.AFFECTS,
        "implements": RelationshipType.IMPLEMENTS,
        "implement": RelationshipType.IMPLEMENTS,
        "similar": RelationshipType.SIMILAR_TO,
        "caused by": RelationshipType.CAUSED_BY,
        "mitigated by": RelationshipType.MITIGATED_BY,
    }

    # Common synonyms for query expansion
    _SYNONYMS: Dict[str, List[str]] = {
        "system": ["subsystem", "module", "unit"],
        "component": ["part", "element", "unit"],
        "procedure": ["process", "operation", "workflow"],
        "satellite": ["spacecraft", "vehicle"],
        "power": ["electrical", "energy"],
        "data": ["information", "telemetry"],
        "command": ["instruction", "control"],
        "failure": ["anomaly", "issue", "problem"],
    }

    def __init__(
        self,
        config: Optional[Config] = None,
        nlp: Optional[Language] = None,
        acronym_resolver: Optional[AcronymResolver] = None,
        normalizer: Optional[StringNormalizer] = None,
        query_history_path: Optional[str | Path] = None,
    ) -> None:
        """Initialize query parser.

        Args:
            config: Configuration object
            nlp: spaCy language model (if None, loads en_core_web_lg)
            acronym_resolver: Acronym resolver for query expansion
            normalizer: String normalizer for entity names
            query_history_path: Path to store query history (default: data/queries/history.jsonl)
        """
        self.config = config or Config.from_yaml()
        self.retrieval_config: RetrievalConfig = self.config.retrieval

        # Load spaCy model
        if nlp is None:
            try:
                self.nlp: Language = spacy.load("en_core_web_lg")
            except OSError:
                logger.warning("spaCy model en_core_web_lg not found, using en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp

        # Initialize normalizer and acronym resolver
        self.normalizer = normalizer or StringNormalizer(config=self.config.normalization)
        self.acronym_resolver = acronym_resolver
        if self.acronym_resolver:
            # Load stored acronym mappings if available
            try:
                self.acronym_resolver.load_mappings()
                logger.info("Loaded acronym mappings for query expansion")
            except FileNotFoundError:
                logger.info("No stored acronym mappings found, query expansion will be limited")

        # Initialize query history
        self.query_history_path = Path(query_history_path or "data/queries/history.jsonl")
        self.query_history_path.parent.mkdir(parents=True, exist_ok=True)

        # Compile intent patterns
        self._compiled_intent_patterns: Dict[QueryIntent, List[re.Pattern[str]]] = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self._INTENT_PATTERNS.items()
        }

        logger.info(
            "Initialized QueryParser",
            spacy_model=self.nlp.meta["name"],
            has_acronym_resolver=self.acronym_resolver is not None,
            query_history=str(self.query_history_path),
        )

    def parse(self, query_text: str, **metadata: Any) -> ParsedQuery:
        """Parse a natural language query.

        Args:
            query_text: Query text to parse
            **metadata: Additional metadata to attach to parsed query

        Returns:
            ParsedQuery object with extracted information

        Raises:
            ValueError: If query is empty or invalid
        """
        # Validate input
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        start_time = datetime.now()

        # Normalize query text
        normalized = self._normalize_query(query_text)

        # Generate unique query ID
        query_id = self._generate_query_id(query_text, start_time)

        # Process with spaCy
        doc = self.nlp(normalized)

        # Extract components
        intent, intent_confidence = self._classify_intent(normalized, doc)
        entity_mentions = self._extract_entity_mentions(doc, query_text)
        relationship_types = self._extract_relationship_types(normalized)
        constraints = self._extract_constraints(normalized)
        expanded_terms = self._expand_query_terms(normalized)
        keywords = self._extract_keywords(doc)

        # Determine if graph traversal is needed
        requires_graph = self._requires_graph_traversal(intent, relationship_types, normalized)
        max_depth = self._determine_max_depth(normalized) if requires_graph else None

        # Create parsed query
        parsed = ParsedQuery(
            query_id=query_id,
            original_text=query_text,
            normalized_text=normalized,
            intent=intent,
            intent_confidence=intent_confidence,
            entity_mentions=entity_mentions,
            relationship_types=relationship_types,
            constraints=constraints,
            expanded_terms=expanded_terms,
            keywords=keywords,
            requires_graph_traversal=requires_graph,
            max_depth=max_depth,
            timestamp=start_time,
            metadata=metadata,
        )

        # Store in history
        if self.config.logging.enable_query_logging:
            self._store_query(parsed)

        # Log parsing time
        parse_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            "Parsed query",
            query_id=query_id,
            intent=intent.value,
            num_entities=len(entity_mentions),
            num_relationships=len(relationship_types),
            parse_time_ms=round(parse_time, 2),
        )

        return parsed

    def _normalize_query(self, query_text: str) -> str:
        """Normalize query text.

        Args:
            query_text: Raw query text

        Returns:
            Normalized query text
        """
        # Strip whitespace
        normalized = query_text.strip()

        # Remove multiple spaces
        normalized = re.sub(r"\s+", " ", normalized)

        # Remove trailing punctuation if question
        if normalized.endswith("?"):
            normalized = normalized[:-1].strip()

        return normalized

    def _classify_intent(self, query_text: str, doc: Doc) -> Tuple[QueryIntent, float]:
        """Classify query intent.

        Args:
            query_text: Normalized query text
            doc: spaCy Doc object

        Returns:
            Tuple of (intent, confidence)
        """
        # Score each intent
        scores: Dict[QueryIntent, int] = dict.fromkeys(QueryIntent, 0)

        # Pattern-based scoring
        for intent, patterns in self._compiled_intent_patterns.items():
            for pattern in patterns:
                if pattern.search(query_text):
                    scores[intent] += 1

        # Check for multiple intents (hybrid)
        non_zero_intents = [intent for intent, score in scores.items() if score > 0]

        if len(non_zero_intents) >= 2:
            # Hybrid query
            return QueryIntent.HYBRID, 0.8

        if len(non_zero_intents) == 1:
            # Single clear intent
            return non_zero_intents[0], 0.9

        # No clear pattern match, analyze syntactically
        # Check for question words
        question_words = {"what", "who", "where", "when", "why", "how"}
        first_token = doc[0].text.lower() if len(doc) > 0 else ""

        if first_token in question_words:
            if first_token in {"what", "who"}:
                return QueryIntent.SEMANTIC, 0.7
            elif first_token == "how":
                # Check if "how to" (procedural) or "how does" (semantic)
                if len(doc) > 1 and doc[1].text.lower() == "to":
                    return QueryIntent.PROCEDURAL, 0.7
                else:
                    return QueryIntent.SEMANTIC, 0.7

        # Default to semantic for information-seeking queries
        return QueryIntent.SEMANTIC, 0.5

    def _extract_entity_mentions(self, doc: Doc, original_text: str) -> List[EntityMention]:
        """Extract entity mentions from query.

        Args:
            doc: spaCy Doc object
            original_text: Original query text for character offsets

        Returns:
            List of EntityMention objects
        """
        mentions: List[EntityMention] = []
        seen_texts: Set[str] = set()

        # Extract from spaCy NER
        for ent in doc.ents:
            if ent.text.lower() in seen_texts:
                continue

            # Map spaCy labels to our entity types
            entity_type = self._map_spacy_label(ent.label_)

            # Normalize entity text
            normalized_result = self.normalizer.normalize(ent.text)
            normalized = normalized_result.normalized or ent.text.lower()

            mention = EntityMention(
                text=ent.text,
                normalized=normalized,
                entity_type=entity_type,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=0.8,  # spaCy NER confidence
            )
            mentions.append(mention)
            seen_texts.add(ent.text.lower())

        # Extract capitalized phrases (potential entities)
        for token in doc:
            if (
                token.pos_ in {"PROPN", "NOUN"}
                and token.text[0].isupper()
                and token.text.lower() not in seen_texts
                and len(token.text) > 2
            ):
                normalized_result = self.normalizer.normalize(token.text)
                normalized = normalized_result.normalized or token.text.lower()

                mention = EntityMention(
                    text=token.text,
                    normalized=normalized,
                    entity_type=None,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                    confidence=0.6,  # Lower confidence for capitalization-based
                )
                mentions.append(mention)
                seen_texts.add(token.text.lower())

        return mentions

    def _map_spacy_label(self, label: str) -> Optional[EntityType]:
        """Map spaCy NER label to EntityType.

        Args:
            label: spaCy entity label

        Returns:
            EntityType if mapping exists, None otherwise
        """
        mapping = {
            "ORG": EntityType.SYSTEM,
            "PRODUCT": EntityType.COMPONENT,
            "GPE": None,  # Geopolitical entity, not relevant
            "PERSON": None,
            "DATE": None,
            "TIME": None,
            "MONEY": None,
            "PERCENT": None,
        }
        return mapping.get(label)

    def _extract_relationship_types(self, query_text: str) -> List[RelationshipType]:
        """Extract relationship types mentioned in query.

        Args:
            query_text: Normalized query text

        Returns:
            List of detected relationship types
        """
        relationships: List[RelationshipType] = []
        query_lower = query_text.lower()

        for keyword, rel_type in self._RELATIONSHIP_KEYWORDS.items():
            if keyword in query_lower:
                relationships.append(rel_type)

        return list(set(relationships))  # Remove duplicates

    def _extract_constraints(self, query_text: str) -> List[QueryConstraint]:
        """Extract query constraints and filters.

        Args:
            query_text: Normalized query text

        Returns:
            List of QueryConstraint objects
        """
        constraints: List[QueryConstraint] = []

        for pattern, operator, value_type in self._CONSTRAINT_PATTERNS:
            for match in pattern.finditer(query_text):
                value_str = match.group(1)
                # Convert value based on type
                if value_type == "numeric":
                    value: Any = float(value_str)
                else:
                    value = value_str

                # Determine field name from pattern context
                # This is simplified - could be enhanced with more context analysis
                field = self._infer_constraint_field(operator)

                constraint = QueryConstraint(field=field, operator=operator, value=value)
                constraints.append(constraint)

        return constraints

    def _infer_constraint_field(self, operator: str) -> str:
        """Infer constraint field from operator type.

        Args:
            operator: Constraint operator

        Returns:
            Field name
        """
        field_map = {
            "gt": "confidence_score",
            "lt": "confidence_score",
            "eq": "entity_type",
            "contains": "content",
            "source": "document_id",
            "page": "page_number",
            "document": "document_id",
        }
        return field_map.get(operator, "metadata")

    def _expand_query_terms(self, query_text: str) -> Dict[str, List[str]]:
        """Expand query terms with synonyms and acronyms.

        Args:
            query_text: Normalized query text

        Returns:
            Dictionary mapping original terms to expanded terms
        """
        expansions: Dict[str, List[str]] = {}
        query_lower = query_text.lower()

        # Expand with synonyms
        for term, synonyms in self._SYNONYMS.items():
            if term in query_lower:
                expansions[term] = synonyms.copy()

        # Expand acronyms if resolver available
        if self.acronym_resolver:
            # Find potential acronyms (all caps words)
            acronym_pattern = re.compile(r"\b[A-Z]{2,}\b")
            for match in acronym_pattern.finditer(query_text):
                acronym = match.group(0)
                resolution = self.acronym_resolver.resolve(acronym, context=query_text)
                if resolution:
                    expansions[acronym] = resolution.aliases

        return expansions

    def _extract_keywords(self, doc: Doc) -> List[str]:
        """Extract important keywords from query.

        Args:
            doc: spaCy Doc object

        Returns:
            List of keywords
        """
        keywords: Set[str] = set()

        # Extract nouns and proper nouns
        for token in doc:
            if (
                token.pos_ in {"NOUN", "PROPN"}
                and not token.is_stop
                and len(token.text) > 2
                and token.text.isalpha()
            ):
                keywords.add(token.lemma_.lower())

        # Extract named entities
        for ent in doc.ents:
            keywords.add(ent.text.lower())

        return sorted(keywords)

    def _requires_graph_traversal(
        self, intent: QueryIntent, relationships: List[RelationshipType], query_text: str
    ) -> bool:
        """Determine if query requires graph traversal.

        Args:
            intent: Query intent
            relationships: Extracted relationship types
            query_text: Normalized query text

        Returns:
            True if graph traversal is needed
        """
        # Structural and hybrid queries usually need graph
        if intent in {QueryIntent.STRUCTURAL, QueryIntent.HYBRID}:
            return True

        # Queries with relationships need graph
        if relationships:
            return True

        # Check for graph-related keywords
        graph_keywords = [
            "connected",
            "related",
            "relationship",
            "path",
            "hierarchy",
            "structure",
            "dependency",
            "dependencies",
        ]
        query_lower = query_text.lower()
        return any(keyword in query_lower for keyword in graph_keywords)

    def _determine_max_depth(self, query_text: str) -> int:
        """Determine maximum graph traversal depth from query.

        Args:
            query_text: Normalized query text

        Returns:
            Maximum depth (default from config if not specified)
        """
        # Check for explicit depth mentions
        depth_pattern = re.compile(r"\b(\d+)\s*(?:levels?|hops?|degrees?)\b", re.IGNORECASE)
        match = depth_pattern.search(query_text)
        if match:
            return int(match.group(1))

        # Check for depth keywords
        if any(word in query_text.lower() for word in ["direct", "immediate", "first"]):
            return 1
        elif any(word in query_text.lower() for word in ["indirect", "all", "complete"]):
            return self.retrieval_config.graph_search.max_depth

        # Default from config
        return self.retrieval_config.graph_search.max_depth

    def _generate_query_id(self, query_text: str, timestamp: datetime) -> str:
        """Generate unique query ID.

        Args:
            query_text: Query text
            timestamp: Query timestamp

        Returns:
            Unique query ID
        """
        import hashlib

        # Create hash from query text and timestamp
        content = f"{query_text}:{timestamp.isoformat()}"
        query_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"query_{timestamp.strftime('%Y%m%d_%H%M%S')}_{query_hash}"

    def _store_query(self, parsed: ParsedQuery) -> None:
        """Store parsed query in history.

        Args:
            parsed: ParsedQuery to store
        """
        try:
            with open(self.query_history_path, "a") as f:
                json.dump(parsed.to_dict(), f)
                f.write("\n")
        except Exception as e:
            logger.warning("Failed to store query history", error=str(e))

    def load_query_history(self, limit: Optional[int] = None) -> List[ParsedQuery]:
        """Load query history from storage.

        Args:
            limit: Maximum number of queries to load (most recent first)

        Returns:
            List of ParsedQuery objects
        """
        if not self.query_history_path.exists():
            return []

        queries: List[ParsedQuery] = []
        try:
            with open(self.query_history_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    # Reconstruct ParsedQuery from dict
                    # Note: This is simplified - full reconstruction would need more logic
                    queries.append(self._reconstruct_query(data))

            # Return most recent first
            queries.reverse()
            if limit:
                queries = queries[:limit]

        except Exception as e:
            logger.warning("Failed to load query history", error=str(e))

        return queries

    def _reconstruct_query(self, data: Dict[str, Any]) -> ParsedQuery:
        """Reconstruct ParsedQuery from stored dict.

        Args:
            data: Query dictionary from storage

        Returns:
            ParsedQuery object
        """
        # Reconstruct entity mentions
        entity_mentions = [
            EntityMention(
                text=m["text"],
                normalized=m["normalized"],
                entity_type=EntityType(m["entity_type"]) if m.get("entity_type") else None,
                start_char=m["start_char"],
                end_char=m["end_char"],
                confidence=m.get("confidence", 1.0),
            )
            for m in data.get("entity_mentions", [])
        ]

        # Reconstruct relationship types
        relationship_types = [RelationshipType(rt) for rt in data.get("relationship_types", [])]

        # Reconstruct constraints
        constraints = [QueryConstraint(**c) for c in data.get("constraints", [])]

        return ParsedQuery(
            query_id=data["query_id"],
            original_text=data["original_text"],
            normalized_text=data["normalized_text"],
            intent=QueryIntent(data["intent"]),
            intent_confidence=data.get("intent_confidence", 1.0),
            entity_mentions=entity_mentions,
            relationship_types=relationship_types,
            constraints=constraints,
            expanded_terms=data.get("expanded_terms", {}),
            keywords=data.get("keywords", []),
            requires_graph_traversal=data.get("requires_graph_traversal", False),
            max_depth=data.get("max_depth"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )

    def validate_query(self, parsed: ParsedQuery) -> Tuple[bool, Optional[str]]:
        """Validate parsed query.

        Args:
            parsed: ParsedQuery to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if query is too short
        if len(parsed.normalized_text.split()) < 2:
            return False, "Query is too short (minimum 2 words)"

        # Check if intent confidence is too low
        if parsed.intent == QueryIntent.UNKNOWN or parsed.intent_confidence < 0.3:
            return False, "Could not determine query intent"

        # Warn if no entities found (but still valid)
        if not parsed.entity_mentions and not parsed.keywords:
            logger.warning(
                "Query has no entity mentions or keywords",
                query_id=parsed.query_id,
                query=parsed.original_text,
            )

        return True, None

    def get_query_statistics(self) -> Dict[str, Any]:
        """Get statistics about query history.

        Returns:
            Dictionary with query statistics
        """
        queries = self.load_query_history()

        if not queries:
            return {"total_queries": 0}

        intent_counts = {}
        for query in queries:
            intent_counts[query.intent.value] = intent_counts.get(query.intent.value, 0) + 1

        avg_entities = sum(len(q.entity_mentions) for q in queries) / len(queries)
        avg_keywords = sum(len(q.keywords) for q in queries) / len(queries)
        graph_queries = sum(1 for q in queries if q.requires_graph_traversal)

        return {
            "total_queries": len(queries),
            "intent_distribution": intent_counts,
            "avg_entities_per_query": round(avg_entities, 2),
            "avg_keywords_per_query": round(avg_keywords, 2),
            "queries_requiring_graph": graph_queries,
            "graph_query_percentage": round(graph_queries / len(queries) * 100, 1),
        }

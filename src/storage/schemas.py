"""Pydantic models for Neo4j graph entities and relationships."""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    """Entity types in the knowledge graph."""

    SYSTEM = "SYSTEM"
    SUBSYSTEM = "SUBSYSTEM"
    COMPONENT = "COMPONENT"
    PARAMETER = "PARAMETER"
    PROCEDURE = "PROCEDURE"
    PROCEDURE_STEP = "PROCEDURE_STEP"
    CONCEPT = "CONCEPT"
    DOCUMENT = "DOCUMENT"
    STANDARD = "STANDARD"
    ANOMALY = "ANOMALY"
    TABLE = "TABLE"
    FIGURE = "FIGURE"


class EntityStatus(str, Enum):
    """Entity curation status."""

    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class ExtractionMethod(str, Enum):
    """Method used to extract entity."""

    SPACY = "spacy"
    LLM = "llm"
    MANUAL = "manual"
    MERGED = "merged"


class CandidateStatus(str, Enum):
    """Curation status for extraction candidates (pre-entity)."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class EntityCandidate(BaseModel):
    """Pre-curation entity candidate stored separately from production entities."""

    id: Optional[str] = Field(default=None, description="Candidate identifier (optional)")
    candidate_key: str = Field(..., description="Deterministic key for aggregation/upserts")
    canonical_name: str = Field(..., description="Canonical surface form")
    candidate_type: EntityType = Field(..., description="Proposed entity type")
    aliases: List[str] = Field(default_factory=list, description="Aliases observed in extraction")
    description: str = Field(default="", description="Candidate description (best-effort)")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Candidate confidence")
    status: CandidateStatus = Field(default=CandidateStatus.PENDING, description="Curation status")
    mention_count: int = Field(default=0, ge=0, description="Mentions (aggregation-friendly delta)")
    source_documents: List[str] = Field(default_factory=list, description="Document IDs where seen")
    chunk_ids: List[str] = Field(default_factory=list, description="Chunk IDs where seen")
    conflicting_types: List[str] = Field(
        default_factory=list, description="Alternative types suggested by extractors"
    )
    provenance_events: List[str] = Field(
        default_factory=list,
        description="JSON-serialized provenance events (stored as list-of-strings in Neo4j)",
    )
    first_seen: datetime = Field(default_factory=datetime.now, description="First time seen")
    last_seen: datetime = Field(default_factory=datetime.now, description="Last time seen")

    def to_neo4j_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["candidate_type"] = self.candidate_type.value
        data["status"] = self.status.value
        data["first_seen"] = self.first_seen.isoformat()
        data["last_seen"] = self.last_seen.isoformat()
        return data

    @staticmethod
    def provenance_event(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, default=str)


class RelationshipCandidate(BaseModel):
    """Pre-curation relationship candidate stored separately from production relationships."""

    id: Optional[str] = Field(default=None, description="Candidate identifier (optional)")
    candidate_key: str = Field(..., description="Deterministic key for aggregation/upserts")
    source: str = Field(..., description="Source entity surface form")
    target: str = Field(..., description="Target entity surface form")
    type: str = Field(..., description="Proposed relationship type")
    description: str = Field(default="", description="Relationship description")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Candidate confidence")
    status: CandidateStatus = Field(default=CandidateStatus.PENDING, description="Curation status")
    mention_count: int = Field(default=0, ge=0, description="Mentions (aggregation-friendly delta)")
    source_documents: List[str] = Field(default_factory=list, description="Document IDs where seen")
    chunk_ids: List[str] = Field(default_factory=list, description="Chunk IDs where seen")
    provenance_events: List[str] = Field(
        default_factory=list,
        description="JSON-serialized provenance events (stored as list-of-strings in Neo4j)",
    )
    first_seen: datetime = Field(default_factory=datetime.now, description="First time seen")
    last_seen: datetime = Field(default_factory=datetime.now, description="Last time seen")

    def to_neo4j_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["status"] = self.status.value
        data["first_seen"] = self.first_seen.isoformat()
        data["last_seen"] = self.last_seen.isoformat()
        return data

    @staticmethod
    def provenance_event(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, default=str)


class Entity(BaseModel):
    """Base entity model for all entity types."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique entity identifier")
    canonical_name: str = Field(..., description="Canonical name of the entity")
    entity_type: EntityType = Field(..., description="Type of entity")
    aliases: List[str] = Field(default_factory=list, description="Alternative names and aliases")
    description: str = Field(default="", description="Entity description")
    abbreviations: List[str] = Field(default_factory=list, description="Common abbreviations")
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score of entity extraction"
    )
    extraction_method: ExtractionMethod = Field(
        default=ExtractionMethod.LLM, description="Method used to extract entity"
    )
    status: EntityStatus = Field(default=EntityStatus.DRAFT, description="Curation status")
    first_seen: datetime = Field(
        default_factory=datetime.now, description="First time entity was extracted"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )
    mention_count: int = Field(default=1, ge=0, description="Number of times entity is mentioned")
    source_documents: List[str] = Field(
        default_factory=list, description="List of document IDs where entity appears"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional entity properties"
    )

    @field_validator("canonical_name")
    @classmethod
    def validate_canonical_name(cls, v: str) -> str:
        """Validate and normalize canonical name."""
        if not v or not v.strip():
            raise ValueError("Canonical name cannot be empty")
        return v.strip().lower().replace(" ", "_")

    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert entity to Neo4j-compatible dictionary.

        Important:
            This includes subclass fields (e.g., `Document.filename`, `Document.checksum`)
            so specialized entity types persist their schema properly.
        """
        data: Dict[str, Any] = self.model_dump()

        # Convert enums to their stored values
        data["entity_type"] = self.entity_type.value
        data["extraction_method"] = self.extraction_method.value
        data["status"] = self.status.value

        # Convert datetimes to ISO strings
        data["first_seen"] = self.first_seen.isoformat()
        data["last_updated"] = self.last_updated.isoformat()

        # Merge arbitrary properties into the top-level (Neo4j-friendly)
        extra = data.pop("properties", None) or {}
        if isinstance(extra, dict):
            data.update(extra)

        return data


class System(Entity):
    """Top-level technical system entity."""

    entity_type: Literal[EntityType.SYSTEM] = EntityType.SYSTEM
    subsystems: List[str] = Field(default_factory=list, description="List of subsystem IDs")


class Subsystem(Entity):
    """Subsystem entity within a system."""

    entity_type: Literal[EntityType.SUBSYSTEM] = EntityType.SUBSYSTEM
    parent_system: Optional[str] = Field(None, description="Parent system ID")
    components: List[str] = Field(default_factory=list, description="List of component IDs")


class Component(Entity):
    """Individual component entity."""

    entity_type: Literal[EntityType.COMPONENT] = EntityType.COMPONENT
    parent_subsystem: Optional[str] = Field(None, description="Parent subsystem ID")
    part_number: Optional[str] = Field(None, description="Component part number")
    manufacturer: Optional[str] = Field(None, description="Component manufacturer")


class Parameter(Entity):
    """Measurable parameter entity."""

    entity_type: Literal[EntityType.PARAMETER] = EntityType.PARAMETER
    unit: Optional[str] = Field(None, description="Unit of measurement")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    nominal_value: Optional[float] = Field(None, description="Nominal value")
    parameter_type: Optional[str] = Field(
        None, description="Type of parameter (e.g., temperature, voltage)"
    )


class Procedure(Entity):
    """Operational procedure entity."""

    entity_type: Literal[EntityType.PROCEDURE] = EntityType.PROCEDURE
    procedure_type: Optional[str] = Field(
        None, description="Type of procedure (e.g., startup, shutdown)"
    )
    steps: List[str] = Field(default_factory=list, description="List of procedure step IDs")
    duration_minutes: Optional[int] = Field(None, description="Estimated duration in minutes")
    prerequisites: List[str] = Field(
        default_factory=list, description="List of prerequisite procedure IDs"
    )


class ProcedureStep(Entity):
    """Individual step within a procedure."""

    entity_type: Literal[EntityType.PROCEDURE_STEP] = EntityType.PROCEDURE_STEP
    parent_procedure: Optional[str] = Field(None, description="Parent procedure ID")
    step_number: Optional[int] = Field(None, description="Step number in sequence")
    step_text: str = Field(..., description="Step instruction text")
    warnings: List[str] = Field(default_factory=list, description="Safety warnings for this step")
    checks: List[str] = Field(default_factory=list, description="Verification checks for this step")


class Concept(Entity):
    """Technical concept or principle entity."""

    entity_type: Literal[EntityType.CONCEPT] = EntityType.CONCEPT
    concept_category: Optional[str] = Field(None, description="Category of concept")
    related_concepts: List[str] = Field(
        default_factory=list, description="List of related concept IDs"
    )


class Document(Entity):
    """Source document entity."""

    entity_type: Literal[EntityType.DOCUMENT] = EntityType.DOCUMENT
    filename: str = Field(..., description="Original filename")
    title: Optional[str] = Field(None, description="Document title")
    version: Optional[str] = Field(None, description="Document version")
    date: Optional[str] = Field(None, description="Document date")
    author: Optional[str] = Field(None, description="Document author")
    page_count: Optional[int] = Field(None, description="Number of pages")
    checksum: Optional[str] = Field(None, description="Document checksum for change detection")


class Standard(Entity):
    """Industry standard or regulation entity."""

    entity_type: Literal[EntityType.STANDARD] = EntityType.STANDARD
    standard_body: Optional[str] = Field(
        None, description="Standards organization (e.g., ISO, NASA)"
    )
    standard_number: Optional[str] = Field(None, description="Standard number or identifier")
    version: Optional[str] = Field(None, description="Standard version")
    publication_date: Optional[str] = Field(None, description="Publication date")


class Anomaly(Entity):
    """Known issue or failure mode entity."""

    entity_type: Literal[EntityType.ANOMALY] = EntityType.ANOMALY
    severity: Optional[str] = Field(
        None, description="Severity level (e.g., critical, major, minor)"
    )
    affected_components: List[str] = Field(
        default_factory=list, description="List of affected component IDs"
    )
    root_cause: Optional[str] = Field(None, description="Root cause description")
    mitigation_procedures: List[str] = Field(
        default_factory=list, description="List of mitigation procedure IDs"
    )


class Table(Entity):
    """Table entity containing structured data."""

    entity_type: Literal[EntityType.TABLE] = EntityType.TABLE
    table_number: Optional[str] = Field(None, description="Table number in document")
    caption: Optional[str] = Field(None, description="Table caption")
    page_number: Optional[int] = Field(None, description="Page number where table appears")
    parent_document: Optional[str] = Field(None, description="Parent document ID")
    column_headers: List[str] = Field(default_factory=list, description="Table column headers")
    row_count: Optional[int] = Field(None, description="Number of rows")
    structured_data: Optional[Dict[str, Any]] = Field(None, description="Structured table data")


class Figure(Entity):
    """Figure or diagram entity."""

    entity_type: Literal[EntityType.FIGURE] = EntityType.FIGURE
    figure_number: Optional[str] = Field(None, description="Figure number in document")
    caption: Optional[str] = Field(None, description="Figure caption")
    page_number: Optional[int] = Field(None, description="Page number where figure appears")
    parent_document: Optional[str] = Field(None, description="Parent document ID")
    figure_type: Optional[str] = Field(
        None, description="Type of figure (e.g., diagram, chart, schematic)"
    )
    image_path: Optional[str] = Field(None, description="Path to extracted image file")


class RelationshipType(str, Enum):
    """Relationship types in the knowledge graph."""

    # Structural relationships
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    DEPENDS_ON = "DEPENDS_ON"

    # Functional relationships
    CONTROLS = "CONTROLS"
    MONITORS = "MONITORS"
    PROVIDES_POWER_TO = "PROVIDES_POWER_TO"
    SENDS_DATA_TO = "SENDS_DATA_TO"

    # Procedural relationships
    REFERENCES = "REFERENCES"
    PRECEDES = "PRECEDES"
    REQUIRES_CHECK = "REQUIRES_CHECK"
    AFFECTS = "AFFECTS"

    # Semantic relationships
    IMPLEMENTS = "IMPLEMENTS"
    SIMILAR_TO = "SIMILAR_TO"
    CAUSED_BY = "CAUSED_BY"
    MITIGATED_BY = "MITIGATED_BY"

    # Table/Figure relationships
    REFERENCES_TABLE = "REFERENCES_TABLE"
    REFERENCES_FIGURE = "REFERENCES_FIGURE"
    DEFINED_IN_TABLE = "DEFINED_IN_TABLE"
    SHOWN_IN_FIGURE = "SHOWN_IN_FIGURE"
    CONTAINS_TABLE = "CONTAINS_TABLE"
    CONTAINS_FIGURE = "CONTAINS_FIGURE"

    # Document relationships
    CROSS_REFERENCES = "CROSS_REFERENCES"
    MENTIONED_IN = "MENTIONED_IN"
    PARENT_CHUNK = "PARENT_CHUNK"
    CHILD_CHUNK = "CHILD_CHUNK"


class RelationshipProvenance(BaseModel):
    """Provenance information for relationships tracking where they were found."""

    document_id: str = Field(..., description="Document ID where relationship was found")
    section: Optional[str] = Field(None, description="Section within document")
    page_number: Optional[int] = Field(None, description="Page number")
    chunk_id: Optional[str] = Field(None, description="Chunk ID where relationship was extracted")
    extracted_text: Optional[str] = Field(
        None, description="Extracted text supporting the relationship"
    )
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence of extraction"
    )


class Relationship(BaseModel):
    """Relationship between entities in the knowledge graph."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique relationship identifier"
    )
    type: RelationshipType = Field(..., description="Type of relationship")
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    description: str = Field(default="", description="Relationship description")
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score of relationship extraction"
    )
    extraction_method: ExtractionMethod = Field(
        default=ExtractionMethod.LLM, description="Method used to extract relationship"
    )
    bidirectional: bool = Field(default=False, description="Whether relationship is bidirectional")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )
    status: EntityStatus = Field(default=EntityStatus.DRAFT, description="Curation status")
    provenance: List[RelationshipProvenance] = Field(
        default_factory=list,
        description="Provenance information tracking where relationship was found",
    )
    confirmation_count: int = Field(
        default=1, ge=0, description="Number of sources confirming relationship"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional relationship properties"
    )

    @field_validator("provenance")
    @classmethod
    def update_confirmation_count(
        cls, v: List[RelationshipProvenance]
    ) -> List[RelationshipProvenance]:
        """Update confirmation count based on provenance."""
        return v

    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert relationship to Neo4j-compatible dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "confidence_score": self.confidence_score,
            "extraction_method": self.extraction_method.value,
            "bidirectional": self.bidirectional,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "status": self.status.value,
            "confirmation_count": self.confirmation_count,
            "source_documents": [p.document_id for p in self.provenance],
            "source_chunks": [p.chunk_id for p in self.provenance if p.chunk_id],
            **self.properties,
        }


class Chunk(BaseModel):
    """Document chunk model."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    level: Literal[1, 2, 3, 4] = Field(
        ..., description="Hierarchy level (1=doc, 2=section, 3=subsection, 4=para)"
    )
    parent_chunk_id: Optional[str] = Field(None, description="Parent chunk ID")
    child_chunk_ids: List[str] = Field(default_factory=list, description="Child chunk IDs")
    content: str = Field(..., description="Chunk text content")
    section_title: Optional[str] = Field(None, description="Section title")
    page_numbers: List[int] = Field(
        default_factory=list, description="Page numbers covered by chunk"
    )
    hierarchy_path: Optional[str] = Field(None, description="Section numbering (e.g., 1.2.3)")
    token_count: int = Field(..., ge=0, description="Number of tokens in chunk")
    entity_ids: List[str] = Field(default_factory=list, description="Entities mentioned in chunk")
    has_tables: bool = Field(default=False, description="Whether chunk contains tables")
    has_figures: bool = Field(default=False, description="Whether chunk contains figures")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert chunk to Neo4j-compatible dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "level": self.level,
            "parent_chunk_id": self.parent_chunk_id,
            "child_chunk_ids": self.child_chunk_ids,
            "content": self.content,
            "section_title": self.section_title,
            "page_numbers": self.page_numbers,
            "hierarchy_path": self.hierarchy_path,
            "token_count": self.token_count,
            "entity_ids": self.entity_ids,
            "has_tables": self.has_tables,
            "has_figures": self.has_figures,
            "created_at": self.created_at.isoformat(),
        }

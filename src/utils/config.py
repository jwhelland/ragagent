"""Configuration management using Pydantic for validation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PDFParserConfig(BaseSettings):
    """PDF Parser configuration."""

    ocr_enabled: bool = True
    preserve_layout: bool = True
    extract_tables: bool = True
    extract_figures: bool = True
    supported_formats: List[str] = [".pdf"]


class TextCleaningConfig(BaseSettings):
    """Text cleaning configuration."""

    enabled: bool = True
    patterns_file: str = "config/cleaning_patterns.yaml"
    remove_headers: bool = True
    remove_footers: bool = True
    remove_page_numbers: bool = True
    min_line_length: int = 3
    normalize_whitespace: bool = True
    preserve_code_blocks: bool = True
    preserve_equations: bool = True
    preserve_technical_terms: bool = True


class LLMConfig(BaseSettings):
    """LLM configuration."""

    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 60
    retry_attempts: int = 3
    base_url: str | None = None

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v


class TextRewritingConfig(BaseSettings):
    """Text rewriting configuration."""

    enabled: bool = False
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunk_level: Literal["section", "subsection"] = "section"
    preserve_original: bool = True
    max_chunk_tokens: int = 2000
    prompt_template: str = "config/rewriting_prompt.yaml"


class ChunkingConfig(BaseSettings):
    """Chunking configuration."""

    strategy: Literal["hierarchical", "fixed", "semantic"] = "hierarchical"
    levels: List[str] = ["document", "section", "subsection", "paragraph"]
    max_tokens: int = 512
    overlap_tokens: int = 50
    min_chunk_size: int = 100


class IngestionConfig(BaseSettings):
    """Ingestion pipeline configuration."""

    pdf_parser: PDFParserConfig = Field(default_factory=PDFParserConfig)
    text_cleaning: TextCleaningConfig = Field(default_factory=TextCleaningConfig)
    text_rewriting: TextRewritingConfig = Field(default_factory=TextRewritingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class SpacyConfig(BaseSettings):
    """spaCy NER configuration."""

    model: str = "en_core_web_trf"
    custom_patterns: str = "config/entity_patterns.jsonl"
    batch_size: int = 100
    confidence_threshold: float = 0.5


class ExtractionConfig(BaseSettings):
    """Entity extraction configuration."""

    spacy: SpacyConfig = Field(default_factory=SpacyConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    entity_types: List[str] = Field(
        default=[
            "SYSTEM",
            "SUBSYSTEM",
            "COMPONENT",
            "PARAMETER",
            "PROCEDURE",
            "PROCEDURE_STEP",
            "CONCEPT",
            "DOCUMENT",
            "STANDARD",
            "ANOMALY",
            "TABLE",
            "FIGURE",
        ]
    )
    relationship_types: List[str] = Field(
        default=[
            "PART_OF",
            "CONTAINS",
            "DEPENDS_ON",
            "CONTROLS",
            "MONITORS",
            "PROVIDES_POWER_TO",
            "SENDS_DATA_TO",
            "REFERENCES",
            "PRECEDES",
            "REQUIRES_CHECK",
            "AFFECTS",
            "IMPLEMENTS",
            "SIMILAR_TO",
            "CAUSED_BY",
            "MITIGATED_BY",
            "REFERENCES_TABLE",
            "REFERENCES_FIGURE",
            "DEFINED_IN_TABLE",
            "SHOWN_IN_FIGURE",
            "CONTAINS_TABLE",
            "CONTAINS_FIGURE",
            "CROSS_REFERENCES",
        ]
    )


class NormalizationConfig(BaseSettings):
    """Normalization configuration."""

    fuzzy_threshold: float = 0.90
    embedding_similarity_threshold: float = 0.85
    auto_merge_threshold: float = 0.95
    min_mention_count: int = 2
    enable_acronym_resolution: bool = True
    enable_fuzzy_matching: bool = True
    enable_semantic_matching: bool = True


class VectorSearchConfig(BaseSettings):
    """Vector search configuration."""

    top_k: int = 20
    min_score: float = 0.5
    enable_mmr: bool = False
    mmr_lambda: float = 0.5


class GraphSearchConfig(BaseSettings):
    """Graph search configuration."""

    max_depth: int = 3
    relationship_types: List[str] = ["PART_OF", "DEPENDS_ON", "REFERENCES", "CONTAINS"]
    enable_shortest_path: bool = True


class HybridSearchConfig(BaseSettings):
    """Hybrid search configuration."""

    enabled: bool = True
    parallel_execution: bool = True
    strategy_selection: Literal["auto", "vector_first", "graph_first", "hybrid"] = "auto"


class RerankingConfig(BaseSettings):
    """Reranking configuration."""

    enabled: bool = True
    weights: Dict[str, float] = {
        "vector_similarity": 0.4,
        "graph_relevance": 0.3,
        "entity_coverage": 0.15,
        "confidence": 0.10,
        "diversity": 0.05,
    }
    max_results: int = 10


class RetrievalConfig(BaseSettings):
    """Retrieval configuration."""

    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    graph_search: GraphSearchConfig = Field(default_factory=GraphSearchConfig)
    hybrid: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)


class CurationConfig(BaseSettings):
    """Curation configuration."""

    auto_approve_threshold: float = 0.95
    review_queue_size: int = 50
    batch_size: int = 10
    enable_audit_trail: bool = True


class PipelineConfig(BaseSettings):
    """Pipeline configuration."""

    batch_size: int = 10
    max_workers: int = 4
    enable_progress_bar: bool = True
    checkpoint_interval: int = 100
    enable_checkpointing: bool = True


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = "INFO"
    format: Literal["json", "text"] = "json"
    file: str = "logs/ragagent2.log"
    max_size_mb: int = 100
    backup_count: int = 5
    enable_query_logging: bool = True


class DatabaseConfig(BaseSettings):
    """Database configuration from environment variables."""

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="ragagent2024")
    neo4j_database: str = Field(default="neo4j")

    # Qdrant
    # If set (e.g. ":memory:"), QdrantClient will use local/in-memory mode and no server is required.
    qdrant_location: str = Field(default="")
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_grpc_port: int = Field(default=6334)
    qdrant_prefer_grpc: bool = Field(default=False)
    qdrant_api_key: str = Field(default="")
    qdrant_https: bool = Field(default=False)

    # Embedding
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dimension: int = Field(default=768)
    embedding_batch_size: int = Field(default=32)


class Config(BaseSettings):
    """Main configuration class."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # Configuration sections
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # Environment variables
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Data paths
    raw_data_path: Path = Field(default=Path("data/raw"))
    processed_data_path: Path = Field(default=Path("data/processed"))
    entities_data_path: Path = Field(default=Path("data/entities"))
    normalization_data_path: Path = Field(default=Path("data/normalization"))

    @staticmethod
    def _deep_merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge two dicts (overrides win).

        This is used to apply environment-derived overrides on top of YAML defaults.
        """
        merged: Dict[str, Any] = dict(base)
        for key, value in overrides.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = Config._deep_merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def from_yaml(cls, yaml_path: str | Path = "config/config.yaml") -> "Config":
        """Load configuration from YAML file and environment variables.

        Precedence (highest to lowest):
        1) Environment variables / .env
        2) YAML file
        3) Model defaults

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance with loaded settings

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML file is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path) as f:
            yaml_config = yaml.safe_load(f) or {}

        if not isinstance(yaml_config, dict):
            raise ValueError(f"YAML config root must be a mapping/dict: {yaml_path}")

        # Load env/.env into a settings instance, then apply only *non-default* values
        # on top of YAML.
        #
        # Note: nested BaseSettings (like DatabaseConfig) do NOT automatically pick up
        # plain env vars (e.g. NEO4J_PASSWORD) via the parent model, so we explicitly
        # compute env overrides for DatabaseConfig and merge them under "database".
        env_overrides = cls().model_dump(exclude_defaults=True)

        db_env_overrides = DatabaseConfig().model_dump(exclude_defaults=True)
        if db_env_overrides:
            env_overrides["database"] = cls._deep_merge_dict(
                yaml_config.get("database", {}) if isinstance(yaml_config.get("database", {}), dict) else {},
                db_env_overrides,
            )

        merged = cls._deep_merge_dict(yaml_config, env_overrides)

        return cls(**merged)

    def validate_config(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate paths exist or can be created
        for path_name in ["raw_data_path", "processed_data_path", "entities_data_path", "normalization_data_path"]:
            path = getattr(self, path_name)
            path.mkdir(parents=True, exist_ok=True)

        # Validate LLM configuration
        if self.llm_provider == "openai" and not self.openai_api_key:
            if "api.openai.com" in (self.openai_base_url or ""):
                raise ValueError("OpenAI API key required when using openai provider")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("Anthropic API key required when using anthropic provider")

        # Validate embedding dimension matches model
        valid_dimensions = {
            "BAAI/bge-small-en-v1.5": 768,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
        }
        if self.database.embedding_model in valid_dimensions:
            expected_dim = valid_dimensions[self.database.embedding_model]
            if self.database.embedding_dimension != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: {self.database.embedding_model} "
                    f"requires {expected_dim} dimensions, got {self.database.embedding_dimension}"
                )


# Global configuration instance
_config: Config | None = None


def get_config() -> Config:
    """Get or create global configuration instance.

    Returns:
        Global Config instance

    Raises:
        RuntimeError: If configuration hasn't been initialized
    """
    global _config
    if _config is None:
        raise RuntimeError(
            "Configuration not initialized. Call load_config() first."
        )
    return _config


def load_config(yaml_path: str | Path = "config/config.yaml") -> Config:
    """Load and validate configuration.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Loaded and validated Config instance
    """
    global _config
    _config = Config.from_yaml(yaml_path)
    _config.validate_config()
    return _config


def reset_config() -> None:
    """Reset global configuration (mainly for testing)."""
    global _config
    _config = None

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from src.utils.config import Config

logger = logging.getLogger(__name__)


class PipelineContext(BaseModel):
    """Context object passed between pipeline stages."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Input
    file_path: Path
    config: Config
    force_reingest: bool = False
    topics: Optional[List[str]] = None

    # Processing State
    parsed_document: Optional[Any] = None  # ParsedDocument
    chunks: List[Any] = []  # List[HierarchicalChunker.Chunk]
    embeddings: List[Any] = []  # List[numpy.ndarray]

    # Extraction State
    spacy_entities_by_chunk: Dict[str | None, List[Any]] = {}
    llm_entities_by_chunk: Dict[str | None, List[Any]] = {}

    # Output/Result State
    ingestion_result: Optional[Any] = None  # IngestionResult

    # Stats
    stats: Dict[str, Any] = {}

    # Control flow
    skipped: bool = False
    skip_reason: Optional[str] = None

    def update_stats(self, key: str, value: Any, operation: str = "add") -> None:
        """Update a statistic value."""
        if operation == "add":
            self.stats[key] = self.stats.get(key, 0) + value
        elif operation == "set":
            self.stats[key] = value


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the stage logic."""
        pass


class Pipeline:
    """Orchestrator for executing a sequence of stages."""

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run all stages in sequence."""
        start_time = time.time()
        logger.info(f"Starting pipeline for {context.file_path.name}")

        try:
            for stage in self.stages:
                if context.skipped:
                    logger.info(f"Skipping remaining stages: {context.skip_reason}")
                    break

                stage_start = time.time()
                logger.debug(f"Starting stage: {stage.name}")

                context = stage.run(context)

                duration = time.time() - stage_start
                logger.debug(f"Completed stage: {stage.name} in {duration:.2f}s")

        except Exception as e:
            logger.error(
                f"Pipeline failed at stage {stage.name if 'stage' in locals() else 'unknown'}: {e}"
            )
            raise e

        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.2f}s")
        return context

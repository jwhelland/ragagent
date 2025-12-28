from typing import Optional

from loguru import logger

from src.ingestion.chunker import HierarchicalChunker
from src.normalization.acronym_resolver import AcronymResolver
from src.normalization.string_normalizer import StringNormalizer
from src.pipeline.base import PipelineContext, PipelineStage


class ChunkingStage(PipelineStage):
    """Stage for hierarchical chunking and acronym discovery."""

    def __init__(self, config):
        super().__init__("Chunking")
        self.config = config
        self.chunker = HierarchicalChunker(config.ingestion.chunking)

        self.acronym_resolver: Optional[AcronymResolver] = None
        if config.normalization.enable_acronym_resolution:
            self.string_normalizer = StringNormalizer(config.normalization)
            self.acronym_resolver = AcronymResolver(
                config=config.normalization,
                normalizer=self.string_normalizer,
            )

    def run(self, context: PipelineContext) -> PipelineContext:
        if not context.parsed_document:
            raise ValueError("No parsed document found in context")

        # Create chunks
        chunks = self.chunker.chunk_document(context.parsed_document)
        if not chunks:
            raise Exception("No chunks created from document")

        context.chunks = chunks
        context.update_stats("chunks_created", len(chunks))

        # Build/update acronym dictionary
        added = self._update_acronym_dictionary(chunks)
        context.update_stats("acronym_definitions_added", added)

        return context

    def _update_acronym_dictionary(self, chunks) -> int:
        if not self.acronym_resolver:
            return 0
        try:
            return int(self.acronym_resolver.update_dictionary_from_chunks(chunks))
        except Exception as exc:
            logger.warning(f"Acronym dictionary update failed: {exc}")
            return 0

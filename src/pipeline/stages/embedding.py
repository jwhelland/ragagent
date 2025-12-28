from loguru import logger

from src.pipeline.base import PipelineContext, PipelineStage
from src.utils.embeddings import EmbeddingGenerator


class EmbeddingStage(PipelineStage):
    """Stage for generating embeddings for chunks."""

    def __init__(self, config):
        super().__init__("Embedding")
        self.config = config
        self.generator = EmbeddingGenerator(config.database)

    def run(self, context: PipelineContext) -> PipelineContext:
        chunks = context.chunks
        if not chunks:
            return context

        logger.debug(f"Generating embeddings for {len(chunks)} chunks")
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.generator.generate(chunk_texts)

        if len(embeddings) != len(chunks):
            raise Exception(
                f"Embedding count mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks"
            )

        context.embeddings = embeddings
        return context

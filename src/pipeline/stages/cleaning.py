from typing import Dict

from loguru import logger

from src.ingestion.text_cleaner import TextCleaner
from src.ingestion.text_rewriter import TextRewriter
from src.pipeline.base import PipelineContext, PipelineStage


class CleaningStage(PipelineStage):
    """Stage for cleaning text."""

    def __init__(self, config):
        super().__init__("Cleaning")
        self.config = config.ingestion.text_cleaning
        self.cleaner = TextCleaner(self.config)

    def run(self, context: PipelineContext) -> PipelineContext:
        if self.config.enabled and context.parsed_document:
            context.parsed_document.raw_text = self.cleaner.clean(context.parsed_document.raw_text)
        return context


class RewritingStage(PipelineStage):
    """Stage for rewriting text using LLM."""

    def __init__(self, config):
        super().__init__("Rewriting")
        self.config = config.ingestion.text_rewriting
        self.llm_config = config.llm.resolve("rewriting")
        self.rewriter = None
        if self.config.enabled:
            self.rewriter = TextRewriter(self.config, llm_config=self.llm_config)

    def run(self, context: PipelineContext) -> PipelineContext:
        if not self.config.enabled or not self.rewriter or not context.parsed_document:
            return context

        logger.debug("Rewriting text (optional)")
        self._rewrite_parsed_document(context.parsed_document)
        return context

    def _rewrite_parsed_document(self, parsed_doc) -> None:
        """Rewrite parsed document content based on config chunk_level."""
        chunk_level = self.config.chunk_level

        # Preserve original for audit if requested
        if self.config.preserve_original:
            parsed_doc.metadata.setdefault("rewriting", {})
            parsed_doc.metadata["rewriting"]["original_raw_text"] = parsed_doc.raw_text

        rewritten_count = 0

        if chunk_level == "section":
            original_sections: Dict[str, str] = {}
            for section in parsed_doc.structure.get("sections", []):
                key = getattr(section, "hierarchy_path", "") or getattr(section, "title", "")
                if self.config.preserve_original:
                    original_sections[key] = section.content

                result = self.rewriter.rewrite(
                    section.content,
                    metadata={
                        "section_title": getattr(section, "title", ""),
                        "hierarchy_path": getattr(section, "hierarchy_path", ""),
                    },
                )
                if result.used_rewrite:
                    section.content = result.rewritten
                    rewritten_count += 1

            if self.config.preserve_original:
                parsed_doc.metadata.setdefault("rewriting", {})
                parsed_doc.metadata["rewriting"]["original_sections"] = original_sections

        elif chunk_level == "subsection":
            original_subsections: Dict[str, str] = {}
            for section in parsed_doc.structure.get("sections", []):
                for subsection in getattr(section, "subsections", []) or []:
                    key = getattr(subsection, "hierarchy_path", "") or getattr(
                        subsection, "title", ""
                    )
                    if self.config.preserve_original:
                        original_subsections[key] = subsection.content

                    result = self.rewriter.rewrite(
                        subsection.content,
                        metadata={
                            "subsection_title": getattr(subsection, "title", ""),
                            "hierarchy_path": getattr(subsection, "hierarchy_path", ""),
                        },
                    )
                    if result.used_rewrite:
                        subsection.content = result.rewritten
                        rewritten_count += 1
        else:
            logger.warning(f"Unknown rewriting chunk_level: {chunk_level}. Skipping rewriting.")
            return

        # Always rewrite doc-level raw_text too
        doc_result = self.rewriter.rewrite(
            parsed_doc.raw_text,
            metadata={
                "document_title": parsed_doc.metadata.get("title", ""),
                "filename": parsed_doc.metadata.get("filename", ""),
            },
        )
        if doc_result.used_rewrite:
            parsed_doc.raw_text = doc_result.rewritten
            rewritten_count += 1

        parsed_doc.metadata.setdefault("rewriting", {})
        parsed_doc.metadata["rewriting"].update(
            {
                "enabled": True,
                "chunk_level": chunk_level,
                "preserve_original": self.config.preserve_original,
                "rewritten_units": rewritten_count,
            }
        )

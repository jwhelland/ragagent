from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_file_parser import TextFileParser
from src.pipeline.base import PipelineContext, PipelineStage


class ParsingStage(PipelineStage):
    """Stage for parsing documents (PDF, TXT, MD)."""

    def __init__(self, config):
        super().__init__("Parsing")
        self.config = config
        self.pdf_parser = PDFParser(config.ingestion.pdf_parser)
        self.text_parser = TextFileParser()

    def run(self, context: PipelineContext) -> PipelineContext:
        suffix = context.file_path.suffix.lower()

        if suffix == ".pdf":
            context.parsed_document = self.pdf_parser.parse_pdf(context.file_path)
        elif suffix in {".txt", ".md", ".markdown"}:
            context.parsed_document = self.text_parser.parse_file(context.file_path)
        else:
            raise ValueError(
                f"Unsupported document type: {suffix} (supported: .pdf, .txt, .md, .markdown)"
            )

        if context.parsed_document.error:
            raise Exception(f"Document parsing failed: {context.parsed_document.error}")

        return context

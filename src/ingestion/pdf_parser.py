"""PDF parsing module using Docling with OCR support.

This module provides PDF parsing capabilities using Docling, extracting text,
structure, tables, figures, and metadata from PDF documents.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.metadata_extractor import MetadataExtractor
from src.utils.config import PDFParserConfig
from src.utils.lzma_compat import ensure_lzma


class TableData(BaseModel):
    """Represents an extracted table."""

    model_config = ConfigDict(extra="allow")

    table_id: str
    caption: str = ""
    content: str = ""  # Markdown or text representation
    page_number: int = 1
    position: int = 0  # Position in document


class FigureData(BaseModel):
    """Represents an extracted figure."""

    model_config = ConfigDict(extra="allow")

    figure_id: str
    caption: str = ""
    page_number: int = 1
    position: int = 0
    description: str = ""  # Optional description


class Section(BaseModel):
    """Represents a document section."""

    model_config = ConfigDict(extra="allow")

    level: int = 1  # 1=section, 2=subsection, 3=subsubsection
    title: str = ""
    content: str = ""
    start_page: int = 1
    end_page: int = 1
    hierarchy_path: str = ""  # e.g., "1.2.3"
    subsections: List[Section] = Field(default_factory=list)
    tables: List[TableData] = Field(default_factory=list)
    figures: List[FigureData] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """Represents a fully parsed PDF document."""

    model_config = ConfigDict(extra="allow")

    document_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    structure: Dict[str, List[Section]] = Field(default_factory=lambda: {"sections": []})
    raw_text: str = ""
    tables: List[TableData] = Field(default_factory=list)
    figures: List[FigureData] = Field(default_factory=list)
    page_count: int = 0
    error: Optional[str] = None


Section.model_rebuild()


class PDFParser:
    """Parse PDF documents using Docling with OCR support.

    This class handles PDF parsing, extracting text, structure (sections,
    subsections), tables, figures, and metadata. It uses Docling for robust
    PDF parsing with OCR capabilities for scanned documents.

    Example:
        >>> parser = PDFParser(config)
        >>> parsed_doc = parser.parse_pdf("document.pdf")
        >>> print(f"Found {len(parsed_doc.tables)} tables")
    """

    def __init__(self, config: Optional[PDFParserConfig] = None) -> None:
        """Initialize the PDF parser.

        Docling is imported lazily (only when parsing an existing PDF) to avoid
        importing heavy optional dependencies during test collection.

        Args:
            config: PDF parser configuration. If None, uses default settings.
        """
        self.config = config or PDFParserConfig()
        self.metadata_extractor = MetadataExtractor()
        self.converter = None  # initialized lazily

        logger.info(
            f"Initialized PDFParser (lazy Docling). OCR={'enabled' if self.config.ocr_enabled else 'disabled'}"
        )

    def parse_pdf(self, pdf_path: Path | str) -> ParsedDocument:
        """Parse a PDF document.

        Docling is imported/initialized only after we confirm the file exists.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ParsedDocument containing extracted content and structure

        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path.name}")

        try:
            self._ensure_converter()

            # Convert PDF using Docling
            result = self.converter.convert(str(pdf_path))

            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(
                pdf_path, first_page_text=self._get_first_page_text(result)
            )

            # Extract text content
            raw_text = result.document.export_to_text()

            # Extract structure (sections, subsections)
            structure = self._extract_structure(result)

            # Extract tables
            tables = self._extract_tables(result)

            # Extract figures
            figures = self._extract_figures(result)

            # Create deterministic document ID for resume/change-detection.
            # Prefer checksum; fall back to filename if checksum is unavailable.
            checksum = metadata.get("checksum") or pdf_path.name
            document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(checksum)))

            # Create parsed document
            parsed_doc = ParsedDocument(
                document_id=document_id,
                metadata=metadata,
                structure=structure,
                raw_text=raw_text,
                tables=tables,
                figures=figures,
                page_count=metadata.get("page_count", 0),
            )

            logger.success(
                f"Parsed {pdf_path.name}: {parsed_doc.page_count} pages, "
                f"{len(structure.get('sections', []))} sections, "
                f"{len(tables)} tables, {len(figures)} figures"
            )

            return parsed_doc

        except Exception as e:
            logger.error(f"Error parsing {pdf_path.name}: {e}")
            # Return error document
            return ParsedDocument(
                document_id=str(uuid.uuid4()),
                metadata={"filename": pdf_path.name, "error": str(e)},
                structure={"sections": []},
                raw_text="",
                page_count=0,
                error=str(e),
            )

    def _ensure_converter(self) -> None:
        """Initialize Docling converter lazily."""
        if self.converter is not None:
            return

        # Some Python builds omit stdlib lzma; ensure a compatible module before importing docling/transformers.
        ensure_lzma()

        # Import Docling only when needed.
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.ocr_enabled
        pipeline_options.do_table_structure = self.config.extract_tables

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def _get_first_page_text(self, result: Any) -> str:
        """Extract text from the first page.

        Args:
            result: Docling conversion result

        Returns:
            Text from the first page
        """
        try:
            # Get first page content
            pages = list(result.document.pages)
            if pages:
                return pages[0].export_to_text()
            return ""
        except Exception:
            return ""

    def _extract_structure(self, result: Any) -> Dict[str, List[Section]]:
        """Extract document structure (sections, subsections).

        Args:
            result: Docling conversion result

        Returns:
            Dictionary with sections list
        """
        sections: List[Section] = []

        try:
            current_section: Optional[Section] = None
            current_subsection: Optional[Section] = None
            section_counter = 0
            subsection_counter = 0

            for item_iter in result.document.iterate_items():
                # Unpack tuple if necessary (docling 2.0+ behavior)
                item = (
                    item_iter[0]
                    if isinstance(item_iter, tuple) and len(item_iter) > 0
                    else item_iter
                )

                # Get label safely
                item_type = str(item.label) if hasattr(item, "label") else ""

                # Handle different heading levels
                if item_type == "title" or item_type == "section_header":
                    section_counter += 1
                    subsection_counter = 0

                    # Save previous section
                    if current_section:
                        sections.append(current_section)

                    # Create new section
                    current_section = Section(
                        level=1,
                        title=item.text if hasattr(item, "text") else "",
                        content="",
                        start_page=(
                            item.prov[0].page_no if hasattr(item, "prov") and item.prov else 1
                        ),
                        end_page=item.prov[0].page_no if hasattr(item, "prov") and item.prov else 1,
                        hierarchy_path=str(section_counter),
                    )
                    current_subsection = None

                elif item_type == "subtitle" or item_type == "subsection_header":
                    if current_section:
                        subsection_counter += 1

                        current_subsection = Section(
                            level=2,
                            title=item.text if hasattr(item, "text") else "",
                            content="",
                            start_page=(
                                item.prov[0].page_no if hasattr(item, "prov") and item.prov else 1
                            ),
                            end_page=(
                                item.prov[0].page_no if hasattr(item, "prov") and item.prov else 1
                            ),
                            hierarchy_path=f"{section_counter}.{subsection_counter}",
                        )

                elif item_type == "paragraph" or item_type == "text":
                    text = item.text if hasattr(item, "text") else ""

                    # Add to current subsection or section
                    if current_subsection:
                        current_subsection.content += text + "\n\n"
                        if hasattr(item, "prov") and item.prov:
                            current_subsection.end_page = item.prov[0].page_no
                    elif current_section:
                        current_section.content += text + "\n\n"
                        if hasattr(item, "prov") and item.prov:
                            current_section.end_page = item.prov[0].page_no

                # Handle subsection completion
                if current_subsection and item_type in ["subtitle", "title", "section_header"]:
                    if current_section:
                        current_section.subsections.append(current_subsection)
                        current_subsection = None

            # Add final subsection and section
            if current_subsection and current_section:
                current_section.subsections.append(current_subsection)
            if current_section:
                sections.append(current_section)

        except Exception as e:
            logger.warning(f"Error extracting structure: {e}")

        return {"sections": sections}

    def _extract_tables(self, result: Any) -> List[TableData]:
        """Extract tables from the document.

        Args:
            result: Docling conversion result

        Returns:
            List of extracted tables
        """
        tables: List[TableData] = []

        try:
            position = 0
            for item_iter in result.document.iterate_items():
                # Unpack tuple if necessary
                item = (
                    item_iter[0]
                    if isinstance(item_iter, tuple) and len(item_iter) > 0
                    else item_iter
                )

                label = str(item.label) if hasattr(item, "label") else ""

                if label == "table":
                    position += 1

                    # Extract table data
                    caption = ""
                    content = ""

                    if hasattr(item, "caption"):
                        caption = item.caption

                    if hasattr(item, "export_to_markdown"):
                        try:
                            content = item.export_to_markdown(doc=result.document)
                        except TypeError:
                            content = item.export_to_markdown()
                    elif hasattr(item, "text"):
                        content = item.text

                    page_num = 1
                    if hasattr(item, "prov") and item.prov:
                        page_num = item.prov[0].page_no

                    table = TableData(
                        table_id=f"table_{position}",
                        caption=caption,
                        content=content,
                        page_number=page_num,
                        position=position,
                    )
                    tables.append(table)

        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")

        return tables

    def _extract_figures(self, result: Any) -> List[FigureData]:
        """Extract figures from the document.

        Args:
            result: Docling conversion result

        Returns:
            List of extracted figures
        """
        figures: List[FigureData] = []

        try:
            position = 0
            for item_iter in result.document.iterate_items():
                # Unpack tuple if necessary
                item = (
                    item_iter[0]
                    if isinstance(item_iter, tuple) and len(item_iter) > 0
                    else item_iter
                )

                label = str(item.label) if hasattr(item, "label") else ""

                if label in ["figure", "picture", "image"]:
                    position += 1

                    caption = ""
                    if hasattr(item, "caption"):
                        caption = item.caption
                    elif hasattr(item, "text"):
                        caption = item.text

                    page_num = 1
                    if hasattr(item, "prov") and item.prov:
                        page_num = item.prov[0].page_no

                    figure = FigureData(
                        figure_id=f"figure_{position}",
                        caption=caption,
                        page_number=page_num,
                        position=position,
                    )
                    figures.append(figure)

        except Exception as e:
            logger.warning(f"Error extracting figures: {e}")

        return figures

    def parse_batch(self, pdf_paths: List[Path | str], max_errors: int = 5) -> List[ParsedDocument]:
        """Parse multiple PDF documents.

        Args:
            pdf_paths: List of paths to PDF files
            max_errors: Maximum number of consecutive errors before stopping

        Returns:
            List of parsed documents
        """
        results = []
        consecutive_errors = 0

        for pdf_path in pdf_paths:
            try:
                parsed_doc = self.parse_pdf(pdf_path)
                results.append(parsed_doc)

                # Reset error counter on success
                if not parsed_doc.error:
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1

            except Exception as e:
                logger.error(f"Failed to parse {pdf_path}: {e}")
                consecutive_errors += 1

                # Add error document
                results.append(
                    ParsedDocument(
                        document_id=str(uuid.uuid4()),
                        metadata={"filename": Path(pdf_path).name, "error": str(e)},
                        structure={"sections": []},
                        raw_text="",
                        error=str(e),
                    )
                )

            # Stop if too many consecutive errors
            if consecutive_errors >= max_errors:
                logger.error(f"Stopping batch parsing after {max_errors} consecutive errors")
                break

        logger.info(f"Batch parsing complete: {len(results)} documents processed")
        return results

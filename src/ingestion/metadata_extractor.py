"""PDF metadata extraction module.

This module extracts metadata from PDF documents including title, date, version,
author, and page count.
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from pypdf import PdfReader


class MetadataExtractor:
    """Extract metadata from PDF documents.

    This class extracts standard PDF metadata fields as well as attempts to
    extract domain-specific metadata like document version and date from
    the document content.
    """

    def __init__(self) -> None:
        """Initialize the metadata extractor."""
        # Common patterns for extracting metadata from text
        self.date_patterns = [
            r"date[:\s]+(\d{4}-\d{2}-\d{2})",
            r"date[:\s]+(\d{2}/\d{2}/\d{4})",
            r"dated[:\s]+(\d{4}-\d{2}-\d{2})",
            r"(\d{4}-\d{2}-\d{2})",
        ]

        self.version_patterns = [
            r"version[:\s]+([\d.]+)",
            r"rev[:\s]+([\d.]+)",
            r"revision[:\s]+([\d.]+)",
            r"v([\d.]+)",
        ]

    def extract_metadata(
        self,
        pdf_path: Path | str,
        first_page_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            first_page_text: Optional text from the first page for enhanced extraction

        Returns:
            Dictionary containing extracted metadata:
                - filename: Name of the PDF file
                - title: Document title (from PDF metadata or filename)
                - date: Document date (if found)
                - version: Document version (if found)
                - author: Document author (if found)
                - page_count: Number of pages
                - file_size: File size in bytes
                - creation_date: PDF creation date
                - modification_date: PDF modification date

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF cannot be read
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting metadata from: {pdf_path.name}")

        metadata: Dict[str, Any] = {
            "filename": pdf_path.name,
            "file_path": str(pdf_path.absolute()),
            "file_size": pdf_path.stat().st_size,
            "checksum": self._sha256_file(pdf_path),
        }

        try:
            # Extract PDF metadata using pypdf
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)

                # Get page count
                metadata["page_count"] = len(reader.pages)

                # Get PDF metadata
                pdf_metadata = reader.metadata
                if pdf_metadata:
                    metadata["title"] = pdf_metadata.get("/Title", "")
                    metadata["author"] = pdf_metadata.get("/Author", "")
                    metadata["subject"] = pdf_metadata.get("/Subject", "")
                    metadata["creator"] = pdf_metadata.get("/Creator", "")

                    # Extract dates
                    creation_date = pdf_metadata.get("/CreationDate")
                    if creation_date:
                        metadata["creation_date"] = self._parse_pdf_date(creation_date)

                    mod_date = pdf_metadata.get("/ModDate")
                    if mod_date:
                        metadata["modification_date"] = self._parse_pdf_date(mod_date)

                # If no title in metadata, use filename without extension
                if not metadata.get("title"):
                    metadata["title"] = pdf_path.stem.replace("_", " ").replace("-", " ")

                # Try to extract version and date from first page text if provided
                if first_page_text:
                    extracted = self._extract_from_text(first_page_text)
                    metadata.update(extracted)

                logger.success(
                    f"Extracted metadata: {metadata['page_count']} pages, "
                    f"title='{metadata.get('title', 'N/A')}'"
                )

                return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path.name}: {e}")
            # Return basic metadata even if extraction fails
            metadata["page_count"] = 0
            metadata["title"] = pdf_path.stem
            metadata["error"] = str(e)
            return metadata

    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract metadata from document text.

        Args:
            text: Text content to extract from

        Returns:
            Dictionary with extracted version and date if found
        """
        extracted: Dict[str, Any] = {}

        # Extract version
        for pattern in self.version_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted["version"] = match.group(1)
                logger.debug(f"Extracted version: {extracted['version']}")
                break

        # Extract date
        for pattern in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                try:
                    # Try to parse the date
                    if "-" in date_str:
                        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                    else:
                        parsed_date = datetime.strptime(date_str, "%m/%d/%Y")
                    extracted["date"] = parsed_date.strftime("%Y-%m-%d")
                    logger.debug(f"Extracted date: {extracted['date']}")
                    break
                except ValueError:
                    continue

        return extracted

    def _sha256_file(self, path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Compute SHA-256 checksum of a file.

        Args:
            path: File path
            chunk_size: Read chunk size in bytes

        Returns:
            Hex-encoded SHA-256 digest
        """
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _parse_pdf_date(self, date_str: str) -> str:
        """Parse PDF date format to ISO format.

        PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'

        Args:
            date_str: PDF date string

        Returns:
            ISO formatted date string (YYYY-MM-DD)
        """
        try:
            # Remove 'D:' prefix if present
            if date_str.startswith("D:"):
                date_str = date_str[2:]

            # Extract date components (YYYYMMDD)
            if len(date_str) >= 8:
                year = date_str[0:4]
                month = date_str[4:6]
                day = date_str[6:8]
                return f"{year}-{month}-{day}"

            return date_str
        except Exception:
            return date_str

    def extract_batch(
        self,
        pdf_paths: list[Path | str]
    ) -> list[Dict[str, Any]]:
        """Extract metadata from multiple PDF files.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            List of metadata dictionaries
        """
        results = []

        for pdf_path in pdf_paths:
            try:
                metadata = self.extract_metadata(pdf_path)
                results.append(metadata)
            except Exception as e:
                logger.error(f"Failed to extract metadata from {pdf_path}: {e}")
                # Add minimal metadata for failed files
                results.append({
                    "filename": Path(pdf_path).name,
                    "error": str(e),
                    "page_count": 0,
                })

        return results

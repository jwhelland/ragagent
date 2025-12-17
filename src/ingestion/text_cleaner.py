"""Text cleaning module for preprocessing extracted PDF text.

This module provides regex-based text cleaning to remove headers, footers,
page numbers, watermarks, and other noise while preserving important technical
content like code blocks, equations, and technical terms.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from loguru import logger

from src.utils.config import TextCleaningConfig


class TextCleaner:
    """Clean and preprocess text extracted from PDFs.

    This class applies regex-based patterns to remove noise from extracted text
    while preserving technical content. Patterns are loaded from a YAML
    configuration file and can be enabled/disabled individually.

    Example:
        >>> cleaner = TextCleaner(config)
        >>> clean_text = cleaner.clean(raw_text)
        >>> print(f"Removed {len(raw_text) - len(clean_text)} characters")
    """

    def __init__(
        self,
        config: Optional[TextCleaningConfig] = None,
        patterns_file: Optional[str | Path] = None
    ) -> None:
        """Initialize the text cleaner.

        Args:
            config: Text cleaning configuration. If None, uses default settings.
            patterns_file: Path to patterns YAML file. Overrides config setting.
        """
        self.config = config or TextCleaningConfig()

        # Load patterns from file
        patterns_path = Path(patterns_file or self.config.patterns_file)
        self.patterns = self._load_patterns(patterns_path)

        # Compile regex patterns for performance
        self._compile_patterns()

        # Track what's preserved
        self._preserved_chunks: List[str] = []

        logger.info(f"Initialized TextCleaner with patterns from {patterns_path}")

    def _load_patterns(self, patterns_file: Path) -> Dict[str, Any]:
        """Load cleaning patterns from YAML file.

        Args:
            patterns_file: Path to patterns YAML file

        Returns:
            Dictionary of pattern categories and patterns

        Raises:
            FileNotFoundError: If patterns file doesn't exist
            yaml.YAMLError: If patterns file is invalid
        """
        if not patterns_file.exists():
            raise FileNotFoundError(f"Patterns file not found: {patterns_file}")

        try:
            with open(patterns_file) as f:
                patterns = yaml.safe_load(f)
                logger.debug(f"Loaded patterns from {patterns_file}")
                return patterns
        except yaml.YAMLError as e:
            logger.error(f"Error parsing patterns file: {e}")
            raise

    def _compile_patterns(self) -> None:
        """Compile regex patterns for better performance."""
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}

        # Compile removal patterns.
        # Use MULTILINE so patterns like `^...$` operate per-line rather than only
        # matching the start/end of the entire string.
        for category in ["headers", "footers", "page_numbers", "watermarks", "noise"]:
            if category in self.patterns and self.patterns[category].get("enabled", False):
                self.compiled_patterns[category] = [
                    re.compile(pattern, flags=re.MULTILINE)
                    for pattern in self.patterns[category].get("patterns", [])
                ]

        # Compile preservation patterns
        if "preserve" in self.patterns and self.patterns["preserve"].get("enabled", False):
            self.compiled_patterns["preserve"] = [
                re.compile(pattern, flags=re.MULTILINE)
                for pattern in self.patterns["preserve"].get("patterns", [])
            ]

        # Compile OCR correction patterns
        if "ocr_corrections" in self.patterns and self.patterns["ocr_corrections"].get("enabled", False):
            self.compiled_patterns["ocr_corrections"] = []
            for item in self.patterns["ocr_corrections"].get("patterns", []):
                if isinstance(item, dict):
                    self.compiled_patterns["ocr_corrections"].append({
                        "pattern": re.compile(item["pattern"]),
                        "replacement": item.get("replacement", ""),
                        "context": item.get("context", ""),
                    })

    def clean(self, text: str) -> str:
        """Clean text by removing noise and normalizing whitespace.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text with noise removed
        """
        if not text:
            return ""

        original_length = len(text)

        # Step 1: Preserve important content
        text, preserved = self._preserve_content(text)

        # Step 2: Remove noise (in order)
        if self.config.remove_headers:
            text = self._apply_patterns(text, "headers")

        if self.config.remove_footers:
            text = self._apply_patterns(text, "footers")

        if self.config.remove_page_numbers:
            text = self._apply_patterns(text, "page_numbers")

        text = self._apply_patterns(text, "watermarks")
        text = self._apply_patterns(text, "noise")

        # Step 3: Apply OCR corrections
        text = self._apply_ocr_corrections(text)

        # Step 4: Normalize whitespace
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Step 5: Restore preserved content
        text = self._restore_content(text, preserved)

        logger.debug(
            f"Cleaned text: {original_length} -> {len(text)} chars "
            f"({100*(1-len(text)/max(original_length, 1)):.1f}% removed)"
        )

        return text

    def _preserve_content(self, text: str) -> tuple[str, Dict[str, str]]:
        """Preserve important content before cleaning.

        Args:
            text: Input text

        Returns:
            Tuple of (modified text with placeholders, preserved content dict)
        """
        preserved: Dict[str, str] = {}

        if "preserve" not in self.compiled_patterns:
            return text, preserved

        # Preserve code blocks
        if self.config.preserve_code_blocks:
            for i, match in enumerate(self.compiled_patterns["preserve"][0].finditer(text)):
                placeholder = f"__CODE_BLOCK_{i}__"
                preserved[placeholder] = match.group(0)

            # Replace with placeholders
            for placeholder, content in preserved.items():
                text = text.replace(content, placeholder)

        # Preserve equations
        if self.config.preserve_equations:
            equation_patterns = self.compiled_patterns["preserve"][1:3]
            for pattern in equation_patterns:
                for i, match in enumerate(pattern.finditer(text)):
                    placeholder = f"__EQUATION_{i}__"
                    if placeholder not in preserved:
                        preserved[placeholder] = match.group(0)
                        text = text.replace(match.group(0), placeholder)

        # Preserve technical terms (if config enabled).
        #
        # Important: do NOT preserve common watermark/header tokens (e.g. CONFIDENTIAL/DRAFT),
        # otherwise the preservation step will "shield" them from being removed later.
        if self.config.preserve_technical_terms and len(self.compiled_patterns["preserve"]) > 3:
            watermark_stoplist = {
                "CONFIDENTIAL",
                "DRAFT",
                "PRELIMINARY",
                "PROPRIETARY",
                "RESTRICTED",
                "INTERNAL",
                "FINAL",
            }

            for pattern in self.compiled_patterns["preserve"][3:]:
                for i, match in enumerate(pattern.finditer(text)):
                    value = match.group(0)
                    if value.upper() in watermark_stoplist:
                        continue

                    placeholder = f"__TECH_{i}__"
                    if placeholder not in preserved:
                        preserved[placeholder] = value
                        text = text.replace(value, placeholder)

        return text, preserved

    def _restore_content(self, text: str, preserved: Dict[str, str]) -> str:
        """Restore preserved content after cleaning.

        Args:
            text: Cleaned text with placeholders
            preserved: Dictionary of placeholders and original content

        Returns:
            Text with preserved content restored
        """
        for placeholder, content in preserved.items():
            text = text.replace(placeholder, content)

        return text

    def _apply_patterns(self, text: str, category: str) -> str:
        """Apply regex patterns from a category.

        Args:
            text: Input text
            category: Pattern category (headers, footers, etc.)

        Returns:
            Text with patterns applied
        """
        if category not in self.compiled_patterns:
            return text

        for pattern in self.compiled_patterns[category]:
            text = pattern.sub("", text)

        return text

    def _apply_ocr_corrections(self, text: str) -> str:
        """Apply OCR error corrections.

        Args:
            text: Input text

        Returns:
            Text with OCR errors corrected
        """
        if "ocr_corrections" not in self.compiled_patterns:
            return text

        for correction in self.compiled_patterns["ocr_corrections"]:
            pattern = correction["pattern"]
            replacement = correction["replacement"]
            context = correction.get("context", "")

            # Apply context-aware corrections
            if context == "numeric":
                # Only replace in numeric contexts
                text = self._replace_in_numeric_context(text, pattern, replacement)
            elif context == "word":
                # Only replace within words
                text = self._replace_in_word_context(text, pattern, replacement)
            else:
                # General replacement
                text = pattern.sub(replacement, text)

        return text

    def _replace_in_numeric_context(
        self,
        text: str,
        pattern: re.Pattern,
        replacement: str
    ) -> str:
        """Replace pattern only in numeric contexts.

        Args:
            text: Input text
            pattern: Regex pattern
            replacement: Replacement string

        Returns:
            Text with replacements in numeric context
        """
        # Find numeric contexts (surrounded by digits)
        numeric_context = re.compile(r'\d+[^\d]*\d+')

        result = []
        last_end = 0

        for match in numeric_context.finditer(text):
            # Keep text before match
            result.append(text[last_end:match.start()])

            # Apply replacement in numeric context
            context_text = match.group(0)
            context_text = pattern.sub(replacement, context_text)
            result.append(context_text)

            last_end = match.end()

        # Add remaining text
        result.append(text[last_end:])

        return "".join(result)

    def _replace_in_word_context(
        self,
        text: str,
        pattern: re.Pattern,
        replacement: str
    ) -> str:
        """Replace pattern only within words.

        Args:
            text: Input text
            pattern: Regex pattern
            replacement: Replacement string

        Returns:
            Text with replacements in word context
        """
        # Match words with the pattern inside
        word_pattern = re.compile(r'\b\w*' + pattern.pattern + r'\w*\b')

        result = []
        last_end = 0

        for match in word_pattern.finditer(text):
            # Keep text before match
            result.append(text[last_end:match.start()])

            # Apply replacement within word
            word = match.group(0)
            word = pattern.sub(replacement, word)
            result.append(word)

            last_end = match.end()

        # Add remaining text
        result.append(text[last_end:])

        return "".join(result)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Collapse multiple spaces to single space
        text = re.sub(r' {2,}', ' ', text)

        # Normalize newlines (max 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing spaces on lines
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)

        # Remove lines with only whitespace
        text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)

        # Normalize special whitespace characters
        text = re.sub(r'[\t\r\f\v]', ' ', text)

        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean multiple texts.

        Args:
            texts: List of texts to clean

        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]

    def get_stats(self, original: str, cleaned: str) -> Dict[str, Any]:
        """Get statistics about text cleaning.

        Args:
            original: Original text
            cleaned: Cleaned text

        Returns:
            Dictionary with cleaning statistics
        """
        return {
            "original_length": len(original),
            "cleaned_length": len(cleaned),
            "removed_chars": len(original) - len(cleaned),
            "removal_percentage": 100 * (1 - len(cleaned) / max(len(original), 1)),
            "original_lines": original.count("\n") + 1,
            "cleaned_lines": cleaned.count("\n") + 1,
        }

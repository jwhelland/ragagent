"""LLM-powered entity and relationship extraction.

This module provides a provider-agnostic interface over OpenAI and Anthropic
chat APIs. Prompts are rendered from YAML templates and responses are parsed
into structured Python objects for downstream processing.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from loguru import logger

from src.extraction.models import ExtractedEntity, ExtractedRelationship
from src.utils.config import LLMConfig
from src.utils.llm_client import create_openai_client


class LLMExtractor:
    """LLM extractor with provider switch, retries, and structured parsing."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        prompts_path: str | Path = "config/extraction_prompts.yaml",
        *,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or LLMConfig()
        self.prompts_path = Path(prompts_path)
        self.prompts = self._load_prompts(self.prompts_path)
        self._sleep = sleep_fn or time.sleep

        logger.info(
            "Initialized LLMExtractor",
            provider=self.config.provider,
            model=self.config.model,
            prompts=str(self.prompts_path),
        )

    # -----------------------
    # Public API
    # -----------------------
    def extract_entities(
        self,
        chunk: Any,
        *,
        document_context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedEntity]:
        """Extract entities from a chunk using the configured LLM provider."""
        chunk_text, chunk_id, document_id, metadata = self._coerce_chunk(chunk)

        context = self._build_prompt_context(
            chunk_text=chunk_text,
            chunk_metadata=metadata,
            document_context=document_context,
            entities_list=None,
        )
        system, user = self._render_prompt("entity_extraction", context)
        raw_response = self._call_llm(system=system, user=user)
        return self._parse_entities_response(raw_response, chunk_id, document_id)

    def extract_relationships(
        self,
        chunk: Any,
        *,
        known_entities: Optional[Iterable[Any]] = None,
        document_context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedRelationship]:
        """Extract relationships from a chunk using the configured LLM provider."""
        chunk_text, chunk_id, document_id, metadata = self._coerce_chunk(chunk)

        entities_list = self._format_entities_for_prompt(known_entities or [])
        context = self._build_prompt_context(
            chunk_text=chunk_text,
            chunk_metadata=metadata,
            document_context=document_context,
            entities_list=entities_list,
        )
        system, user = self._render_prompt("relationship_extraction", context)
        raw_response = self._call_llm(system=system, user=user)
        return self._parse_relationships_response(raw_response, chunk_id, document_id)

    # -----------------------
    # Prompt handling
    # -----------------------
    def _load_prompts(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Extraction prompt template not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Prompt template root must be a mapping/dict: {path}")
        return data

    def _render_prompt(self, key: str, context: Dict[str, Any]) -> Tuple[str, str]:
        if key not in self.prompts:
            raise KeyError(f"Prompt key not found in template: {key}")

        prompt = self.prompts.get(key) or {}
        system = str(prompt.get("system", "")).strip()
        user_template = str(prompt.get("user_template", "{chunk_text}"))

        try:
            user = user_template.format(**context)
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(f"Missing placeholder '{missing}' in prompt context for '{key}'")
        return system, user

    def _build_prompt_context(
        self,
        *,
        chunk_text: str,
        chunk_metadata: Dict[str, Any],
        document_context: Optional[Dict[str, Any]],
        entities_list: Optional[str],
    ) -> Dict[str, Any]:
        metadata = chunk_metadata or {}
        base_context = {
            "chunk_text": chunk_text,
            "document_title": metadata.get("document_title", ""),
            "section_title": metadata.get("section_title", metadata.get("hierarchy_path", "")),
            "page_numbers": metadata.get("page_numbers", []),
            "entities_list": entities_list or "[]",
        }
        if document_context:
            base_context.update({k: v for k, v in document_context.items() if v is not None})
        return base_context

    def _format_entities_for_prompt(self, entities: Iterable[Any]) -> str:
        lines: List[str] = []
        for ent in entities:
            name = ""
            ent_type = ""
            if isinstance(ent, ExtractedEntity):
                name, ent_type = ent.name, ent.type
            elif isinstance(ent, dict):
                name = str(
                    ent.get("name")
                    or ent.get("canonical_name")
                    or ent.get("text")
                    or ent.get("entity")
                    or ""
                ).strip()
                ent_type = str(ent.get("type") or ent.get("label") or ent.get("entity_type") or "")
            else:
                name = str(ent).strip()

            if not name:
                continue

            if ent_type:
                lines.append(f"- {name} ({ent_type})")
            else:
                lines.append(f"- {name}")

        return "\n".join(lines) if lines else "[]"

    # -----------------------
    # LLM invocation
    # -----------------------
    def _call_llm(self, *, system: str, user: str) -> str:
        attempts = max(1, self.config.retry_attempts)
        last_error: Exception | None = None

        logger.info(f"Calling LLM for extraction using {self.config.provider}: {self.config.model}")

        for attempt in range(1, attempts + 1):
            try:
                if self.config.provider == "openai":
                    return self._call_openai(system=system, user=user)
                if self.config.provider == "anthropic":
                    return self._call_anthropic(system=system, user=user)
                raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "LLM request failed",
                    attempt=attempt,
                    max_attempts=attempts,
                    error=str(exc),
                )
                if attempt >= attempts:
                    break
                backoff = min(2 ** (attempt - 1), 8)
                self._sleep(backoff)

        if last_error:
            raise last_error
        raise RuntimeError("LLM request failed for unknown reasons")

    def _call_openai(self, *, system: str, user: str) -> str:
        from openai import BadRequestError

        client = create_openai_client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            api_key=self.config.openai_api_key if hasattr(self.config, "openai_api_key") else None,
        )

        # Base arguments for chat completion
        completion_kwargs = {
            "model": self.config.model,
            "timeout": self.config.timeout,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        # Proactive heuristics for known restricted model families (O1, some 'mini' models)
        model_lower = self.config.model.lower()
        if "o1-" in model_lower or "mini" in model_lower:
            pass  # No temperature heuristic needed anymore

        try:
            # First attempt (possibly with heuristics applied)
            response = client.chat.completions.create(**completion_kwargs)
        except BadRequestError as e:
            changed = False

            # No temperature restrictions to handle anymore

            if changed:
                # Retry with adjusted parameters
                response = client.chat.completions.create(**completion_kwargs)
            else:
                # If it's some other bad request, re-raise
                raise e

        content = response.choices[0].message.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        return str(content or "")

    def _call_anthropic(self, *, system: str, user: str) -> str:
        import anthropic

        client_kwargs: Dict[str, Any] = {}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        client = anthropic.Anthropic(**client_kwargs)
        message = client.messages.create(
            model=self.config.model,
            timeout=self.config.timeout,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=4096,  # Anthropic usually requires max_tokens, setting a safe default
        )
        parts = []
        for block in message.content:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "\n".join(parts).strip()

    # -----------------------
    # Parsing helpers
    # -----------------------
    def _parse_entities_response(
        self, response_text: str, chunk_id: Optional[str], document_id: Optional[str]
    ) -> List[ExtractedEntity]:
        data = self._extract_json(response_text)
        if data is None:
            logger.warning("Failed to parse LLM entity response as JSON", chunk_id=chunk_id)
            return []

        if isinstance(data, dict) and "entities" in data:
            raw_entities = data.get("entities", [])
        elif isinstance(data, list):
            raw_entities = data
        else:
            logger.warning("Unexpected entity response structure", chunk_id=chunk_id)
            return []

        entities: List[ExtractedEntity] = []
        for item in raw_entities:
            if not isinstance(item, dict):
                continue

            name = str(
                item.get("name")
                or item.get("canonical_name")
                or item.get("entity")
                or item.get("text")
                or ""
            ).strip()
            if not name:
                continue

            ent_type = str(
                item.get("type") or item.get("label") or item.get("entity_type") or "UNKNOWN"
            ).strip()
            aliases = self._normalize_aliases(item.get("aliases") or item.get("abbreviations"))
            description = str(item.get("description", "") or "").strip()
            confidence = self._clamp_confidence(item.get("confidence") or item.get("score"))

            entities.append(
                ExtractedEntity(
                    name=name,
                    type=ent_type.upper(),
                    description=description,
                    aliases=aliases,
                    confidence=confidence,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source="llm",
                    raw=item,
                )
            )

        return entities

    def _parse_relationships_response(
        self, response_text: str, chunk_id: Optional[str], document_id: Optional[str]
    ) -> List[ExtractedRelationship]:
        data = self._extract_json(response_text)
        if data is None:
            logger.warning("Failed to parse LLM relationship response as JSON", chunk_id=chunk_id)
            return []

        if isinstance(data, dict) and "relationships" in data:
            raw_relationships = data.get("relationships", [])
        elif isinstance(data, list):
            raw_relationships = data
        else:
            logger.warning("Unexpected relationship response structure", chunk_id=chunk_id)
            return []

        relationships: List[ExtractedRelationship] = []
        for item in raw_relationships:
            if not isinstance(item, dict):
                continue

            source = str(item.get("source") or item.get("from") or "").strip()
            source_type = str(item.get("source_type") or "").strip().upper() or None
            target = str(item.get("target") or item.get("to") or "").strip()
            target_type = str(item.get("target_type") or "").strip().upper() or None
            rel_type = str(item.get("type") or item.get("relationship") or "").strip()

            if not source or not target or not rel_type:
                continue

            # Skip self-loops (relationships to self)
            if source.lower() == target.lower():
                continue

            description = str(item.get("description", "") or "").strip()
            confidence = self._clamp_confidence(item.get("confidence") or item.get("score"))
            bidirectional = bool(item.get("bidirectional", False))

            relationships.append(
                ExtractedRelationship(
                    source=source,
                    source_type=source_type,
                    target=target,
                    target_type=target_type,
                    type=rel_type.upper(),
                    description=description,
                    confidence=confidence,
                    bidirectional=bidirectional,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source_extractor="llm",
                    raw=item,
                )
            )

        return relationships

    def _extract_json(self, text: str) -> Any:
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    def _normalize_aliases(self, aliases: Any) -> List[str]:
        if aliases is None:
            return []
        if isinstance(aliases, str):
            return [aliases.strip()] if aliases.strip() else []
        if isinstance(aliases, Sequence):
            return [str(a).strip() for a in aliases if str(a).strip()]
        return []

    def _clamp_confidence(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(score, 1.0))

    def _coerce_chunk(self, chunk: Any) -> Tuple[str, Optional[str], Optional[str], Dict[str, Any]]:
        text = getattr(chunk, "content", None)
        if text is None and isinstance(chunk, dict):
            text = chunk.get("content")
        if text is None:
            raise ValueError("Chunk must provide a 'content' attribute or key.")

        chunk_id = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None)
        document_id = getattr(chunk, "document_id", None)
        metadata = getattr(chunk, "metadata", {}) or {}

        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or chunk_id
            document_id = chunk.get("document_id") or document_id
            metadata = chunk.get("metadata") or metadata

        if not isinstance(metadata, dict):
            metadata = {}

        return text, chunk_id, document_id, metadata

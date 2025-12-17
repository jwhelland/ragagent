"""Optional LLM-based text rewriting for improved readability.

Phase 1 Task 1.7: Optional LLM Text Rewriting

Goals:
- Improve readability while preserving information (technical terms, acronyms, numbers)
- Disabled by default via config
- Uses separate LLM configuration from extraction pipeline
- Preserves original text for audit/comparison

This module is intentionally conservative: if validation suggests information loss,
it returns the original text.

Config reference:
- [`TextRewritingConfig`](src/utils/config.py:56)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.utils.config import LLMConfig, TextRewritingConfig


class RewriteResult(BaseModel):
    """Result of a rewrite attempt."""

    model_config = ConfigDict(frozen=True, extra="allow")

    original: str
    rewritten: str
    used_rewrite: bool
    reason: str


class TextRewriter:
    """Rewrite text using an LLM, with preservation/validation checks."""

    def __init__(
        self,
        config: Optional[TextRewritingConfig] = None,
        prompt_path: Optional[str | Path] = None,
        *,
        enable: Optional[bool] = None,
    ) -> None:
        self.config = config or TextRewritingConfig()
        self.enabled = self.config.enabled if enable is None else enable
        self.llm: LLMConfig = self.config.llm

        self.prompt_path = Path(prompt_path or self.config.prompt_template)
        self.prompt = self._load_prompt(self.prompt_path)

    def rewrite(
        self,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RewriteResult:
        """Rewrite text if enabled; otherwise return original."""

        if not text.strip():
            return RewriteResult(
                original=text, rewritten=text, used_rewrite=False, reason="empty_input"
            )

        if not self.enabled:
            return RewriteResult(
                original=text, rewritten=text, used_rewrite=False, reason="disabled"
            )

        # Size guard
        if self._approx_tokens(text) > self.config.max_chunk_tokens:
            return RewriteResult(
                original=text,
                rewritten=text,
                used_rewrite=False,
                reason="over_max_chunk_tokens",
            )

        metadata = metadata or {}

        rewritten = self._call_llm(text, metadata=metadata)
        if not rewritten or not rewritten.strip():
            return RewriteResult(
                original=text, rewritten=text, used_rewrite=False, reason="empty_llm_output"
            )

        rewritten = self._postprocess(rewritten)

        ok, reason = self._validate_preservation(original=text, rewritten=rewritten)
        if not ok:
            logger.warning(f"Rewrite rejected: {reason}")
            return RewriteResult(
                original=text, rewritten=text, used_rewrite=False, reason=f"rejected:{reason}"
            )

        return RewriteResult(original=text, rewritten=rewritten, used_rewrite=True, reason="ok")

    # -----------------------
    # Prompt + LLM execution
    # -----------------------

    def _load_prompt(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Rewriting prompt template not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Prompt template root must be a mapping/dict: {path}")

        # Minimal expected fields
        data.setdefault("system", "You are a careful technical editor.")
        data.setdefault(
            "user_template",
            "{text}",
        )
        return data

    def _call_llm(self, text: str, *, metadata: Dict[str, Any]) -> str:
        """Call the configured LLM provider.

        This method is easy to monkeypatch in tests.
        """
        provider = self.llm.provider

        system = str(self.prompt.get("system", "")).strip()
        user_tmpl = str(self.prompt.get("user_template", "{text}"))
        user = user_tmpl.format(text=text, **metadata)

        if provider == "openai":
            return self._call_openai(system=system, user=user)

        if provider == "anthropic":
            return self._call_anthropic(system=system, user=user)

        raise ValueError(f"Unsupported rewriting LLM provider: {provider}")

    def _call_openai(self, *, system: str, user: str) -> str:
        from openai import OpenAI

        client_kwargs: Dict[str, Any] = {}
        if self.llm.base_url:
            client_kwargs["base_url"] = self.llm.base_url

        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=self.llm.model,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return str(resp.choices[0].message.content or "")

    def _call_anthropic(self, *, system: str, user: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.llm.model,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Anthropic returns content blocks
        parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "\n".join(parts).strip()

    # -----------------------
    # Validation / safeguards
    # -----------------------

    def _postprocess(self, text: str) -> str:
        # Normalize whitespace, keep paragraphs.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _approx_tokens(self, text: str) -> int:
        # Same heuristic used elsewhere in Phase 1 code.
        words = len(text.split())
        return int(words * 1.3)

    def _validate_preservation(self, *, original: str, rewritten: str) -> Tuple[bool, str]:
        """Conservative checks to reject obvious info loss.

        - All numbers from the original must appear in the rewritten output.
        - All ALL-CAPS acronyms (len>=2) must appear.
        - Rewritten shouldn't be drastically shorter (heuristic).
        """
        original_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", original))
        rewritten_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", rewritten))
        missing_numbers = sorted(original_numbers - rewritten_numbers)
        if missing_numbers:
            return False, f"missing_numbers:{','.join(missing_numbers[:10])}"

        acronyms = set(re.findall(r"\b[A-Z]{2,}\b", original))
        missing_acronyms = sorted(a for a in acronyms if a not in rewritten)
        if missing_acronyms:
            return False, f"missing_acronyms:{','.join(missing_acronyms[:10])}"

        # Length heuristic: don't allow >60% shrink unless original is tiny.
        if len(original) > 400 and len(rewritten) < int(len(original) * 0.4):
            return False, "too_short"

        return True, "ok"

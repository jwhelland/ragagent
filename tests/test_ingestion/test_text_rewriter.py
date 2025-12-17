"""Tests for optional LLM-based text rewriting (Phase 1 Task 1.7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.text_rewriter import TextRewriter
from src.utils.config import LLMConfig, TextRewritingConfig


def _cfg(tmp_path: Path, *, enabled: bool) -> TextRewritingConfig:
    prompt_path = tmp_path / "rewriting_prompt.yaml"
    prompt_path.write_text(
        "system: |\n  test\nuser_template: |\n  {text}\n",
        encoding="utf-8",
    )
    return TextRewritingConfig(
        enabled=enabled,
        llm=LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
        chunk_level="section",
        preserve_original=True,
        max_chunk_tokens=2000,
        prompt_template=str(prompt_path),
    )


def test_rewriter_disabled_returns_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, enabled=False)
    rewriter = TextRewriter(cfg)

    # If disabled, LLM should never be called; ensure it would fail if called.
    monkeypatch.setattr(rewriter, "_call_llm", lambda *_args, **_kwargs: "SHOULD_NOT_CALL")  # type: ignore[attr-defined]

    res = rewriter.rewrite("Hello 123 EPS")

    assert res.used_rewrite is False
    assert res.rewritten == "Hello 123 EPS"
    assert res.reason == "disabled"


def test_rewriter_accepts_safe_rewrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, enabled=True)
    rewriter = TextRewriter(cfg)

    def _fake_call_llm(text: str, *, metadata: dict) -> str:
        assert "123" in text
        return "Hello EPS. Value is 123."

    monkeypatch.setattr(rewriter, "_call_llm", _fake_call_llm)  # type: ignore[attr-defined]

    res = rewriter.rewrite("Hello EPS value 123.", metadata={"section_title": "X"})

    assert res.used_rewrite is True
    assert "123" in res.rewritten
    assert "EPS" in res.rewritten


def test_rewriter_rejects_missing_number(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, enabled=True)
    rewriter = TextRewriter(cfg)

    monkeypatch.setattr(rewriter, "_call_llm", lambda *_args, **_kwargs: "Hello EPS.")  # type: ignore[attr-defined]

    res = rewriter.rewrite("Hello EPS 123.")

    assert res.used_rewrite is False
    assert res.rewritten == "Hello EPS 123."
    assert res.reason.startswith("rejected:missing_numbers")


def test_rewriter_rejects_missing_acronym(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, enabled=True)
    rewriter = TextRewriter(cfg)

    monkeypatch.setattr(rewriter, "_call_llm", lambda *_args, **_kwargs: "Hello 123.")  # type: ignore[attr-defined]

    res = rewriter.rewrite("Hello EPS 123.")

    assert res.used_rewrite is False
    assert res.rewritten == "Hello EPS 123."
    assert res.reason.startswith("rejected:missing_acronyms")

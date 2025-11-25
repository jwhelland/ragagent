from __future__ import annotations

from typing import Optional

from ..retrieval.models import ContextBundle


CHAT_SYSTEM_PROMPT = """You are an evidence-grounded research assistant. Use only the provided context sections to answer.
- Cite every statement with the corresponding source tag in square brackets (e.g., [S1]).
- If the context lacks the answer, state that clearly instead of guessing.
- Prefer precise numbers, page references, and table details when available.
- Return a concise answer followed by an explicit "Sources: [S#,…]" line."""


def build_user_prompt(question: str, context: ContextBundle, guidance: Optional[str] = None) -> str:
    parts = [
        "Context sections:",
        context.as_prompt_section(),
    ]
    if guidance:
        parts.append(f"Additional guidance: {guidance.strip()}")
    parts.extend(
        [
            f"User question: {question.strip()}",
            "Respond in markdown, cite the supporting [S#] tags inline, and end with a 'Sources:' line listing the tags you used in order.",
        ]
    )
    return "\n\n".join(parts)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvalExample:
    question: str
    answers: List[str]
    guidance: Optional[str] = None
    metadata: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalExample":
        answers = data.get("answers") or data.get("expected_answers") or []
        if isinstance(answers, str):
            answers = [answers]
        return cls(
            question=data["question"],
            answers=[a.strip() for a in answers if a],
            guidance=data.get("guidance"),
            metadata=data.get("metadata") or {},
        )


def load_dataset(path: str | Path) -> List[EvalExample]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    examples: List[EvalExample] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        examples.append(EvalExample.from_dict(data))
    return examples

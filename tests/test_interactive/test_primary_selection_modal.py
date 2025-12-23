from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.curation.interactive.widgets.primary_selection_modal import PrimarySelectionModal
from src.storage.schemas import EntityCandidate, EntityType


@dataclass
class FakeRadioButton:
    id: str
    value: bool = False


class FakeRadioSet:
    def __init__(
        self,
        radio_buttons: Iterable[FakeRadioButton],
        *,
        pressed_index: int | None = None,
    ) -> None:
        self._radio_buttons = list(radio_buttons)
        self.pressed_index = pressed_index

    def query(self, _cls: object) -> list[FakeRadioButton]:
        return list(self._radio_buttons)


def _candidate(candidate_id: str, name: str) -> EntityCandidate:
    return EntityCandidate(
        id=candidate_id,
        candidate_key=f"key-{candidate_id}",
        canonical_name=name,
        candidate_type=EntityType.ORGANIZATION,
        confidence_score=0.5,
        mention_count=1,
    )


def test_primary_selection_uses_pressed_radio_over_child_index() -> None:
    candidates = [_candidate("1", "NASA"), _candidate("2", "National Aeronautics and Space Admin.")]
    modal = PrimarySelectionModal(candidates)

    # Simulate a RadioSet with extra non-radio children (e.g., description rows) where
    # pressed_index may not line up with the candidate list indices.
    radioset = FakeRadioSet(
        [FakeRadioButton("radio-1", value=False), FakeRadioButton("radio-2", value=True)],
        pressed_index=2,
    )

    selected = modal._resolve_selected_candidate(radioset)  # noqa: SLF001 (regression helper)
    assert selected is not None
    assert selected.id == "2"


def test_primary_selection_falls_back_to_pressed_index_when_needed() -> None:
    candidates = [_candidate("1", "NASA"), _candidate("2", "National Aeronautics and Space Admin.")]
    modal = PrimarySelectionModal(candidates)

    radioset = FakeRadioSet(
        [FakeRadioButton("radio-1", value=False), FakeRadioButton("radio-2", value=False)],
        pressed_index=1,
    )

    selected = modal._resolve_selected_candidate(radioset)  # noqa: SLF001 (regression helper)
    assert selected is not None
    assert selected.id == "2"

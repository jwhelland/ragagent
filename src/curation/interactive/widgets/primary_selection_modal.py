"""Primary selection modal for choosing which candidate should be the primary in a merge."""

from typing import List, Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, RadioButton, RadioSet, Static

from src.storage.schemas import EntityCandidate


class PrimarySelectionModal(ModalScreen[Optional[EntityCandidate]]):
    """Modal screen for selecting which candidate should be the primary in a merge."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "select", "Select", show=True),
    ]

    CSS = """
    PrimarySelectionModal {
        align: center middle;
    }

    #primary-dialog {
        width: 90;
        height: auto;
        max-height: 40;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #primary-dialog-title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #primary-instruction {
        width: 100%;
        margin-bottom: 1;
        color: $text-muted;
    }

    #candidates-scroll {
        width: 100%;
        height: 1fr;
        border: solid $secondary;
        margin-bottom: 1;
    }

    RadioButton {
        margin-bottom: 0;
    }

    .candidate-description {
        margin-left: 4;
        margin-bottom: 1;
        color: $text-muted;
        text-style: italic;
    }

    .candidate-description-empty {
        margin-left: 4;
        margin-bottom: 1;
        color: $text-disabled;
        text-style: italic;
    }

    #button-container {
        width: 100%;
        height: auto;
        align: center middle;
    }

    #button-container Horizontal {
        width: auto;
        height: auto;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, candidates: List[EntityCandidate]) -> None:
        """Initialize primary selection modal.

        Args:
            candidates: List of candidates to choose from
        """
        super().__init__()
        self.candidates = candidates
        self._radio_ids: list[str] = []
        self._candidate_by_radio_id: dict[str, EntityCandidate] = {}
        for idx, candidate in enumerate(candidates):
            radio_id = f"radio-{candidate.id}" if candidate.id is not None else f"radio-idx-{idx}"
            self._radio_ids.append(radio_id)
            self._candidate_by_radio_id[radio_id] = candidate

        # Auto-select the candidate with highest confidence as default
        self.default_index = 0
        if candidates:
            max_conf_index = max(
                range(len(candidates)), key=lambda i: candidates[i].confidence_score
            )
            self.default_index = max_conf_index

    def compose(self) -> ComposeResult:
        """Create modal widgets."""
        with Container(id="primary-dialog"):
            yield Label("Select Primary Candidate", id="primary-dialog-title")

            yield Static(
                "Choose which candidate should be the primary (its name and type will be used):",
                id="primary-instruction",
            )

            # Scrollable candidate list with radio buttons and descriptions
            with VerticalScroll(id="candidates-scroll"):
                with RadioSet(id="candidate-radioset"):
                    for idx, candidate in enumerate(self.candidates):
                        is_default = idx == self.default_index

                        # Radio button label with key stats
                        label = (
                            f"{candidate.canonical_name} "
                            f"({candidate.candidate_type.value}) "
                            f"[{candidate.confidence_score:.2f}] "
                            f"- {candidate.mention_count} mentions"
                        )
                        yield RadioButton(label, value=is_default, id=self._radio_ids[idx])

                        # Full description below radio button (if available)
                        if candidate.description:
                            yield Static(
                                f"    {candidate.description}",
                                classes="candidate-description",
                            )
                        else:
                            yield Static(
                                "    (no description)", classes="candidate-description-empty"
                            )

            # Buttons
            with Container(id="button-container"):
                with Horizontal():
                    yield Button("Select Primary", id="select-button", variant="success")
                    yield Button("Cancel", id="cancel-button", variant="error")

    def on_mount(self) -> None:
        """Handle mount event - set focus on radioset."""
        try:
            radioset = self.query_one("#candidate-radioset", RadioSet)
            self.set_focus(radioset)
        except Exception:
            pass

    def _resolve_selected_candidate(self, radioset: RadioSet) -> Optional[EntityCandidate]:
        pressed_button = getattr(radioset, "pressed_button", None)
        if pressed_button is not None:
            candidate = self._candidate_by_radio_id.get(getattr(pressed_button, "id", None))
            if candidate is not None:
                return candidate

        radio_buttons = list(radioset.query(RadioButton))
        for radio_button in radio_buttons:
            if getattr(radio_button, "value", False):
                candidate = self._candidate_by_radio_id.get(getattr(radio_button, "id", None))
                if candidate is not None:
                    return candidate

        pressed_index = getattr(radioset, "pressed_index", None)
        if pressed_index is not None and 0 <= pressed_index < len(radio_buttons):
            candidate = self._candidate_by_radio_id.get(
                getattr(radio_buttons[pressed_index], "id", None)
            )
            if candidate is not None:
                return candidate

        return None

    @on(Button.Pressed, "#select-button")
    def action_select(self) -> None:
        """Handle select button press."""
        try:
            radioset = self.query_one("#candidate-radioset", RadioSet)
            selected = self._resolve_selected_candidate(radioset)
            if selected is not None:
                self.dismiss(selected)
                return

            # No selection or invalid, return default
            self.dismiss(self.candidates[self.default_index] if self.candidates else None)
        except Exception:
            # Fallback to default
            self.dismiss(self.candidates[self.default_index] if self.candidates else None)

    @on(Button.Pressed, "#cancel-button")
    def action_cancel(self) -> None:
        """Handle cancel button press."""
        self.dismiss(None)

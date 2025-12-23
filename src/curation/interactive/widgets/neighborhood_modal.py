"""Modal for resolving neighborhood issues (blocked relationships)."""

from typing import List

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Label, Static

from src.curation.entity_approval import NeighborhoodIssue


class NeighborhoodResolutionModal(ModalScreen[List[NeighborhoodIssue]]):
    """Modal screen for resolving blocked relationships."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "confirm", "Confirm", show=True),
    ]

    CSS = """
    NeighborhoodResolutionModal {
        align: center middle;
    }

    #resolution-dialog {
        width: 90;
        height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #dialog-title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #summary {
        width: 100%;
        margin-bottom: 1;
        padding: 1;
        background: $panel;
        color: $text;
    }

    #issue-list {
        width: 100%;
        height: 1fr;
        border: solid $secondary;
        margin-bottom: 1;
    }

    .issue-row {
        height: auto;
        padding: 1;
        border-bottom: solid $secondary 50%;
    }

    .issue-description {
        margin-left: 2;
        color: $text-muted;
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

    def __init__(self, issues: List[NeighborhoodIssue]) -> None:
        super().__init__()
        self.issues = issues
        self.checkboxes: List[Checkbox] = []

    def compose(self) -> ComposeResult:
        with Container(id="resolution-dialog"):
            yield Label("Neighborhood Resolution", id="dialog-title")

            yield Static(
                f"Found {len(self.issues)} pending relationships blocked by peer entities.\n"
                "Select actions to resolve them:",
                id="summary"
            )

            with VerticalScroll(id="issue-list"):
                for idx, issue in enumerate(self.issues):
                    action_text = ""
                    if issue.issue_type == "promotable":
                        action_text = f"Promote relationship to approved '{issue.peer_name}'"
                    elif issue.issue_type == "resolvable":
                        action_text = f"Approve pending candidate '{issue.peer_name}'"
                    elif issue.issue_type == "missing":
                        action_text = f"Create new entity '{issue.peer_name}' (as CONCEPT)"

                    # Default to checked
                    cb = Checkbox(action_text, value=True, id=f"cb-{idx}")
                    self.checkboxes.append(cb)

                    with Container(classes="issue-row"):
                        yield cb
                        rel_desc = f"{issue.relationship_candidate.source} --[{issue.relationship_candidate.type}]--> {issue.relationship_candidate.target}"
                        yield Static(rel_desc, classes="issue-description")

            with Container(id="button-container"):
                with Horizontal():
                    yield Button("Confirm Selected", id="confirm-button", variant="success")
                    yield Button("Skip All", id="cancel-button", variant="error")

    @on(Button.Pressed, "#confirm-button")
    def action_confirm(self) -> None:
        selected_issues = []
        for idx, cb in enumerate(self.checkboxes):
            if cb.value:
                selected_issues.append(self.issues[idx])
        self.dismiss(selected_issues)

    @on(Button.Pressed, "#cancel-button")
    def action_cancel(self) -> None:
        self.dismiss([])

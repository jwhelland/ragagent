"""Scrollable list widget for displaying entity and relationship candidates."""

from typing import List, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from src.curation.interactive.widgets.relationship_row import RelationshipRow
from src.storage.schemas import EntityCandidate, RelationshipCandidate


class CandidateRow(Static):
    """A single row displaying a candidate's information."""

    def __init__(
        self,
        candidate: EntityCandidate,
        index: int,
        is_selected: bool = False,
        is_checked: bool = False,
    ) -> None:
        """Initialize candidate row.

        Args:
            candidate: The entity candidate to display
            index: Row index in the list
            is_selected: Whether this row is currently selected (navigation cursor)
            is_checked: Whether this row is checked for batch operations
        """
        super().__init__()
        self.candidate = candidate
        self.index = index
        self.is_selected = is_selected
        self.is_checked = is_checked
        self.update_content()

    def update_content(self) -> None:
        """Update the row's display content."""
        # Format confidence score with color
        conf = self.candidate.confidence_score
        if conf >= 0.9:
            conf_color = "green"
        elif conf >= 0.7:
            conf_color = "yellow"
        else:
            conf_color = "red"

        # Build the row text
        text = Text()

        # Checkbox indicator for batch selection
        if self.is_checked:
            text.append("[✓] ", style="bold green")
        else:
            text.append("[ ] ", style="dim")

        # Navigation cursor indicator
        if self.is_selected:
            text.append("► ", style="bold cyan")
        else:
            text.append("  ")

        # Index
        text.append(f"[{self.index + 1:>3}] ", style="dim")

        # Name (truncate if too long)
        name = self.candidate.canonical_name
        if len(name) > 35:
            name = name[:32] + "..."
        text.append(f"{name:<35} ", style="bold" if self.is_selected else "")

        # Type
        type_str = f"{self.candidate.candidate_type.value:<12}"
        text.append(type_str, style="cyan")

        # Confidence
        text.append(f" {conf:.2f} ", style=f"bold {conf_color}")

        # Mentions
        text.append(f"({self.candidate.mention_count:>3} mentions)", style="dim")

        # Status indicator
        if self.candidate.status.value == "approved":
            text.append(" ✓", style="green")
        elif self.candidate.status.value == "rejected":
            text.append(" ✗", style="red")

        self.update(text)

    def set_selected(self, selected: bool) -> None:
        """Update navigation selection state and refresh display.

        Args:
            selected: Whether this row should be selected
        """
        self.is_selected = selected
        self.update_content()

    def set_checked(self, checked: bool) -> None:
        """Update batch selection (checkbox) state and refresh display.

        Args:
            checked: Whether this row should be checked
        """
        self.is_checked = checked
        self.update_content()


class CandidateList(Widget):
    """Scrollable list of entity or relationship candidates with keyboard navigation."""

    # Make widget focusable for keyboard input
    can_focus = True

    BINDINGS = [
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("pageup", "page_up", "Page Up", show=False),
        Binding("pagedown", "page_down", "Page Down", show=False),
        Binding("home", "goto_first", "First", show=False),
        Binding("end", "goto_last", "Last", show=False),
    ]

    # Reactive attributes
    candidates: reactive[List[EntityCandidate | RelationshipCandidate]] = reactive(
        [], recompose=True
    )
    current_index: reactive[int] = reactive(0)
    selected_ids: set[str] = set()  # IDs of candidates selected for batch operations

    def __init__(
        self, candidates: Optional[List[EntityCandidate | RelationshipCandidate]] = None
    ) -> None:
        """Initialize candidate list.

        Args:
            candidates: Optional initial list of entity or relationship candidates
        """
        super().__init__()
        if candidates:
            self.candidates = candidates

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with VerticalScroll():
            if not self.candidates:
                yield Static("No candidates to display", classes="empty-state")
            else:
                for idx, candidate in enumerate(self.candidates):
                    # Determine which row type to use based on candidate type
                    if isinstance(candidate, EntityCandidate):
                        row = CandidateRow(
                            candidate=candidate,
                            index=idx,
                            is_selected=(idx == self.current_index),
                            is_checked=(candidate.id in self.selected_ids),
                        )
                    else:  # RelationshipCandidate
                        row = RelationshipRow(
                            candidate=candidate,
                            index=idx,
                            is_selected=(idx == self.current_index),
                            is_checked=(
                                candidate.id in self.selected_ids if candidate.id else False
                            ),
                        )
                    yield row

    def watch_current_index(self, old_index: int, new_index: int) -> None:
        """React to current index changes by updating row selection.

        Args:
            old_index: Previous selected index
            new_index: New selected index
        """
        # Query for both types of row widgets dynamically (avoids stale list issues)
        candidate_rows = list(self.query(CandidateRow))
        relationship_rows = list(self.query(RelationshipRow))
        rows = candidate_rows + relationship_rows

        # Sort by index to maintain correct order
        rows.sort(key=lambda r: r.index)

        # Update row selection states
        if 0 <= old_index < len(rows):
            rows[old_index].set_selected(False)

        if 0 <= new_index < len(rows):
            rows[new_index].set_selected(True)
            # Scroll to keep selected row visible
            rows[new_index].scroll_visible()

        # Notify parent app of selection change (if app context exists)
        try:
            if hasattr(self.app, "current_index"):
                self.app.current_index = new_index
        except Exception:
            # No app context (e.g., during testing)
            pass

    def watch_candidates(
        self,
        old: List[EntityCandidate | RelationshipCandidate],
        new: List[EntityCandidate | RelationshipCandidate],
    ) -> None:
        """React to candidates list changes by recomposing.

        Args:
            old: Previous candidates list
            new: New candidates list
        """
        # Recomposition will happen automatically due to recompose=True
        pass

    @property
    def current_candidate(self) -> Optional[EntityCandidate | RelationshipCandidate]:
        """Get the currently selected candidate."""
        if 0 <= self.current_index < len(self.candidates):
            return self.candidates[self.current_index]
        return None

    def action_cursor_up(self) -> None:
        """Move selection up by one."""
        if self.current_index > 0:
            self.current_index -= 1

    def action_cursor_down(self) -> None:
        """Move selection down by one."""
        if self.current_index < len(self.candidates) - 1:
            self.current_index += 1

    def action_page_up(self) -> None:
        """Move selection up by one page (10 items)."""
        self.current_index = max(0, self.current_index - 10)

    def action_page_down(self) -> None:
        """Move selection down by one page (10 items)."""
        self.current_index = min(len(self.candidates) - 1, self.current_index + 10)

    def action_goto_first(self) -> None:
        """Jump to first candidate."""
        if self.candidates:
            self.current_index = 0

    def action_goto_last(self) -> None:
        """Jump to last candidate."""
        if self.candidates:
            self.current_index = len(self.candidates) - 1

    def refresh_candidates(self, candidates: List[EntityCandidate | RelationshipCandidate]) -> None:
        """Refresh the candidate list with new data.

        Args:
            candidates: New list of entity or relationship candidates to display
        """
        self.candidates = candidates
        # Note: Don't reset current_index here - let the app manage the index position

    def update_selection_checkboxes(self) -> None:
        """Update checkbox states for all rows based on selected_ids.

        This method updates the checkbox display without recomposing the entire list.
        """
        candidate_rows = list(self.query(CandidateRow))
        relationship_rows = list(self.query(RelationshipRow))
        all_rows = candidate_rows + relationship_rows

        for row in all_rows:
            candidate_id = (
                row.candidate.id if hasattr(row.candidate, "id") and row.candidate.id else None
            )
            is_checked = candidate_id in self.selected_ids if candidate_id else False
            if row.is_checked != is_checked:
                row.set_checked(is_checked)

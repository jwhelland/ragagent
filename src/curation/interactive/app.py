"Main Textual application for interactive entity candidate review."

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from src.curation.batch_operations import BatchCurationService
from src.curation.entity_approval import (
    EntityCurationService,
    NeighborhoodIssue,
    get_neighborhood_issues,
)
from src.curation.interactive.command_parser import CommandHistory, ParsedCommand
from src.curation.interactive.keybindings import ALL_BINDINGS
from src.curation.interactive.preferences import PreferencesManager
from src.curation.interactive.session_tracker import SessionTracker
from src.curation.interactive.tui_logging import setup_tui_logging
from src.curation.interactive.widgets import (
    STATUS_BAR_CSS,
    BatchOperationModal,
    CandidateList,
    CommandModalScreen,
    ComparisonModalScreen,
    DetailPanel,
    DuplicateSuggestionsPanel,
    EditModalScreen,
    EntityCandidateMergePreviewModal,
    EntitySearchModal,
    MergePreviewModal,
    NeighborhoodResolutionModal,
    PrimarySelectionModal,
    SearchFilters,
    SearchModalScreen,
    StatusBar,
)
from src.normalization.normalization_table import NormalizationTable
from src.storage.neo4j_manager import Neo4jManager
from src.storage.schemas import (
    CandidateStatus,
    EntityCandidate,
    EntityType,
    RelationshipCandidate,
)
from src.utils.config import Config, load_config


class LoadingMessage(Static):
    """Loading message widget."""

    def compose(self) -> ComposeResult:
        yield Static("Loading candidates from Neo4j...", classes="loading")


class ReviewApp(App):
    """Interactive TUI application for reviewing entity candidates."""

    CSS = (
        """
    Screen {
        background: $surface;
    }

    LoadingMessage {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }

    #main-container {
        width: 100%;
        height: 100%;
    }

    CandidateList {
        width: 50%;
        height: 100%;
        border: solid $primary;
    }

    DetailPanel {
        width: 30%;
        height: 100%;
        border: solid $secondary;
        padding: 1;
    }

    DuplicateSuggestionsPanel {
        width: 20%;
        height: 100%;
        border: solid $accent;
        padding: 1;
    }

    .empty-state {
        width: 100%;
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }

    .empty-detail {
        width: 100%;
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }

    .detail-header {
        margin-bottom: 1;
    }

    DetailSection {
        margin-bottom: 1;
        padding: 1;
    }
    """
        + STATUS_BAR_CSS
    )

    BINDINGS = ALL_BINDINGS + [
        Binding("t", "toggle_review_mode", "Toggle Mode", show=True),
    ]

    # Reactive attributes for application state
    is_loading: reactive[bool] = reactive(True, recompose=True)
    candidates: reactive[List[EntityCandidate | RelationshipCandidate]] = reactive(
        [], recompose=True
    )
    current_index: reactive[int] = reactive(0)
    filter_status: reactive[str] = reactive("pending")
    review_mode: reactive[str] = reactive("entities")  # "entities" or "relationships"
    approved_count: reactive[int] = reactive(0)
    rejected_count: reactive[int] = reactive(0)
    edited_count: reactive[int] = reactive(0)
    current_filters: Optional[SearchFilters] = None
    search_results: List[int] = []  # Indices of matching candidates
    current_search_result_index: int = -1  # Current position in search results

    # Selection mode attributes
    selection_mode: reactive[bool] = reactive(False)
    selected_candidate_ids: set[str] = set()  # Not reactive to avoid recompose on every selection

    def __init__(
        self,
        config_path: Path = Path("config/config.yaml"),
        table_path: Optional[Path] = None,
        prefs_path: Optional[Path] = None,
    ) -> None:
        """Initialize the review application.

        Args:
            config_path: Path to configuration file
            table_path: Optional path to normalization table
            prefs_path: Optional path to preferences file
        """
        super().__init__()
        setup_tui_logging()
        self.config_path = config_path
        self.table_path = table_path
        self.config: Optional[Config] = None
        self._startup_error: Optional[str] = None
        self.session_tracker = SessionTracker()
        self.command_history = CommandHistory()

        # Initialize preferences manager
        self.preferences_manager = PreferencesManager(prefs_path)
        self.preferences = self.preferences_manager.load()

        # Load configuration
        try:
            self.config = load_config(config_path)
        except Exception as e:
            self._startup_error = f"Failed to load config: {e}"

    @property
    def current_candidate(self) -> Optional[EntityCandidate | RelationshipCandidate]:
        """Get the currently selected candidate."""
        if 0 <= self.current_index < len(self.candidates):
            return self.candidates[self.current_index]
        return None

    def _set_focus_to_candidate_list(self) -> None:
        """Helper to set focus to candidate list (called after DOM refresh)."""
        try:
            candidate_list = self.query_one(CandidateList)
            self.set_focus(candidate_list)
        except Exception:
            # Widget doesn't exist yet
            pass

    def _update_status_bar(self) -> None:
        """Update status bar with current session statistics (called every second)."""
        try:
            status_bar = self.query_one(StatusBar)
            stats = self.session_tracker.stats

            # Update status bar reactive attributes
            status_bar.elapsed_time = stats.formatted_elapsed
            status_bar.approved_count = stats.approved_count
            status_bar.rejected_count = stats.rejected_count
            status_bar.velocity = stats.velocity
            status_bar.total_candidates = len(self.candidates)

            # Calculate remaining candidates (pending only)
            pending_count = sum(1 for c in self.candidates if c.status == CandidateStatus.PENDING)
            status_bar.time_remaining = self.session_tracker.estimate_time_remaining(pending_count)
        except Exception:
            # Widget might not exist yet during startup
            pass

    def _save_session_state(self) -> None:
        """Save session state periodically."""
        # Save session statistics
        self.session_tracker.save_session()

        # Save preferences with current state
        self.preferences_manager.update_session_state(
            filter_status=self.filter_status,
            current_index=self.current_index,
            search_text=self.current_filters.search_text if self.current_filters else None,
            entity_type_filter=self.current_filters.entity_type if self.current_filters else None,
            min_confidence=self.current_filters.min_confidence if self.current_filters else 0.0,
        )

    def _check_milestones(self) -> None:
        """Check if any milestones have been reached and show notifications."""
        total_processed = self.session_tracker.stats.total_processed

        # Define milestones
        milestones = [10, 25, 50, 100, 250, 500, 1000]

        # Check if we just hit a milestone
        for milestone in milestones:
            if total_processed == milestone:
                self.notify(
                    f"🎉 Milestone: {milestone} candidates reviewed!",
                    severity="information",
                    timeout=5,
                )
                break

    def _show_session_summary(self) -> None:
        """Show session summary before exit."""
        summary = self.session_tracker.get_summary()

        summary_text = (
            f"\n📊 Session Summary\n"
            f"{ '='*40}\n"
            f"Time elapsed: {summary['elapsed_time']}\n"
            f"Approved: {summary['approved']}\n"
            f"Rejected: {summary['rejected']}\n"
            f"Edited: {summary['edited']}\n"
            f"Total processed: {summary['total_processed']}\n"
            f"Velocity: {summary['velocity']}\n"
            f"{ '='*40}\n"
        )

        self.notify(summary_text, severity="information", timeout=10)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        if self.is_loading:
            yield LoadingMessage()
        else:
            with Horizontal(id="main-container"):
                yield CandidateList(candidates=self.candidates)
                yield DetailPanel(candidate=self.current_candidate)
                yield DuplicateSuggestionsPanel()
        yield StatusBar()
        yield Footer()

    def on_mount(self) -> None:
        """Handle app mount event."""
        self.title = "Entity Candidate Review"
        self._update_subtitle()

        if self._startup_error:
            self.notify(self._startup_error, severity="error")

        # Start timer to update status bar every second
        self.set_interval(1.0, self._update_status_bar)

        # Start timer to save session state every 30 seconds
        self.set_interval(30.0, self._save_session_state)

        # Check if we should resume from last session
        should_resume = self.preferences_manager.should_resume_session(max_age_minutes=60)
        if should_resume:
            # Load session state
            session_state = self.preferences.session_state
            self.filter_status = session_state.filter_status

            # Attempt to load session stats
            if self.session_tracker.load_session():
                self.notify(
                    f"📂 Resumed session from {int(self.preferences_manager.get_session_age_minutes())} minutes ago",
                    severity="information",
                    timeout=5,
                )

                # Sync counts from loaded session stats
                self.approved_count = self.session_tracker.stats.approved_count
                self.rejected_count = self.session_tracker.stats.rejected_count
                self.edited_count = self.session_tracker.stats.edited_count

        # Load initial candidates (will apply resume filters if any)
        self.load_candidates(show_loading=True, show_loaded_notification=False)

    @work(thread=True)
    def load_candidates(
        self,
        preserve_index: bool = False,
        notification: Optional[str] = None,
        *,
        show_loading: bool = False,
        show_loaded_notification: bool = True,
    ) -> None:
        """Load candidates from Neo4j (runs in worker thread).

        Uses @work(thread=True) to run synchronous Neo4j operations
        in a worker thread without blocking the async UI.

        Args:
            preserve_index: If True, try to maintain the current index position
            notification: Optional notification message to show after load completes
            show_loading: If True, show the loading screen while fetching
            show_loaded_notification: If True, show a "Loaded N candidates" toast
        """
        if not self.config:
            self.call_from_thread(self.notify, "Configuration not loaded", severity="error")
            return

        # Save current index if preserving
        saved_index = self.current_index if preserve_index else 0

        if show_loading:
            self.call_from_thread(self._set_loading, True)

        try:
            # Create Neo4j manager and fetch candidates
            manager = Neo4jManager(self.config.database)
            manager.connect()

            try:
                candidates = []
                if self.review_mode == "entities":
                    results = manager.get_entity_candidates(
                        status=self.filter_status if self.filter_status != "all" else None,
                        limit=50,  # Pagination - fetch 50 at a time
                        offset=0,
                    )
                    # Convert to EntityCandidate objects
                    for i, r in enumerate(results):
                        try:
                            candidates.append(EntityCandidate.model_validate(r))
                        except Exception as validation_error:
                            logger.error(
                                f"Failed to validate candidate {i}: {validation_error}\nData: {r}"
                            )
                            continue
                else:
                    # Relationship candidates
                    results = manager.get_relationship_candidates(
                        status=self.filter_status if self.filter_status != "all" else None,
                        limit=50,
                        offset=0,
                    )
                    # Convert to RelationshipCandidate objects
                    for i, r in enumerate(results):
                        try:
                            candidates.append(RelationshipCandidate.model_validate(r))
                        except Exception as validation_error:
                            logger.error(
                                f"Failed to validate relationship candidate {i}: {validation_error}\nData: {r}"
                            )
                            continue

                # Apply client-side filters if search is active
                search_results = []
                if self.current_filters:
                    filtered_candidates, search_results = self._apply_filters(
                        candidates, self.current_filters
                    )
                    candidates = filtered_candidates

                # Calculate target index before any updates
                if preserve_index and candidates:
                    target_index = min(saved_index, len(candidates) - 1)
                else:
                    target_index = 0

                self.call_from_thread(
                    self._apply_loaded_candidates,
                    candidates,
                    target_index,
                    notification,
                    show_loaded_notification,
                    search_results,
                )

            finally:
                manager.close()

        except Exception as e:
            self.call_from_thread(self._set_loading, False)
            self.call_from_thread(
                self.notify, f"Error loading candidates: {e}", severity="error", markup=False
            )

    def _set_loading(self, is_loading: bool) -> None:
        """Update loading state (must run on the UI thread)."""
        self.is_loading = is_loading

    def _apply_filters(
        self,
        candidates: List[EntityCandidate | RelationshipCandidate],
        filters: SearchFilters,
    ) -> tuple[List[EntityCandidate | RelationshipCandidate], List[int]]:
        """Apply search and filter criteria to candidates.

        Args:
            candidates: List of candidates to filter
            filters: Search and filter criteria

        Returns:
            Tuple of (filtered_candidates, search_result_indices)
        """
        filtered = []
        search_result_indices = []

        for idx, candidate in enumerate(candidates):
            # Apply confidence filter
            if candidate.confidence_score < filters.min_confidence:
                continue

            # Apply entity type filter (only for entities)
            if isinstance(candidate, EntityCandidate):
                if filters.entity_type and candidate.candidate_type.value != filters.entity_type:
                    continue

            # Apply search text filter
            if filters.search_text:
                search_text_lower = filters.search_text.lower()

                if isinstance(candidate, EntityCandidate):
                    name_lower = candidate.canonical_name.lower()
                    aliases_lower = [a.lower() for a in candidate.aliases]
                    match = search_text_lower in name_lower or any(
                        search_text_lower in alias for alias in aliases_lower
                    )
                else:
                    # Relationship search
                    match = (
                        search_text_lower in candidate.source.lower()
                        or search_text_lower in candidate.target.lower()
                        or search_text_lower in candidate.type.lower()
                    )

                if match:
                    # This is a search result
                    search_result_indices.append(len(filtered))
                    filtered.append(candidate)
            else:
                # No search text, include all candidates that pass other filters
                filtered.append(candidate)

        return filtered, search_result_indices

    def _apply_loaded_candidates(
        self,
        candidates: List[EntityCandidate | RelationshipCandidate],
        target_index: int,
        notification: Optional[str],
        show_loaded_notification: bool,
        search_results: Optional[List[int]] = None,
    ) -> None:
        """Apply loaded candidates to the UI (must run on the UI thread)."""
        self.candidates = candidates
        self.is_loading = False

        # Store search results for n/N navigation
        if search_results is not None:
            self.search_results = search_results
            self.current_search_result_index = 0 if search_results else -1

        def finalize() -> None:
            self.current_index = target_index if candidates else 0
            self._update_subtitle()
            self._set_focus_to_candidate_list()

            if candidates:
                # Ensure detail panel updates even if selection doesn't change.
                try:
                    detail_panel = self.query_one(DetailPanel)
                    detail_panel.update_candidate(self.current_candidate)
                except Exception:
                    pass

            if notification:
                self.notify(notification, severity="information")
            elif not candidates:
                self.notify("No candidates found", severity="information")
            elif show_loaded_notification:
                self.notify(f"Loaded {len(candidates)} candidates", severity="information")

        self.call_after_refresh(finalize)

    def get_curation_service(self) -> EntityCurationService:
        """Factory method to create curation service with fresh connections.

        Returns:
            EntityCurationService instance
        """
        if not self.config:
            raise RuntimeError("Configuration not loaded")

        # Determine normalization table path
        table_path = (
            self.table_path
            if self.table_path
            else Path(self.config.normalization.normalization_table_path)
        )

        # Create service with fresh connections
        norm_table = NormalizationTable(table_path=table_path, config=self.config.normalization)
        manager = Neo4jManager(self.config.database)
        manager.connect()

        return EntityCurationService(
            manager=manager,
            normalization_table=norm_table,
            config=self.config,
        )

    # Selection management methods

    def get_selected_candidates(self) -> List[EntityCandidate | RelationshipCandidate]:
        """Get list of currently selected candidates.

        Returns:
            List of candidates that are currently selected
        """
        return [c for c in self.candidates if c.id in self.selected_candidate_ids]

    def toggle_current_selection(self) -> None:
        """Toggle selection state of the current candidate."""
        if not self.current_candidate:
            return

        candidate_id = self.current_candidate.id
        if candidate_id in self.selected_candidate_ids:
            self.selected_candidate_ids.remove(candidate_id)
        else:
            self.selected_candidate_ids.add(candidate_id)

        # Update candidate list widget to show selection change
        try:
            candidate_list = self.query_one(CandidateList)
            candidate_list.selected_ids = self.selected_candidate_ids.copy()
            candidate_list.update_selection_checkboxes()
        except Exception:
            pass

        # Update subtitle to show selection count
        self._update_subtitle()

    def select_all_visible(self) -> None:
        """Select all currently visible candidates."""
        self.selected_candidate_ids = {c.id for c in self.candidates}

        try:
            candidate_list = self.query_one(CandidateList)
            candidate_list.selected_ids = self.selected_candidate_ids.copy()
            candidate_list.update_selection_checkboxes()
        except Exception:
            pass

        self.notify(
            f"Selected {len(self.selected_candidate_ids)} candidates", severity="information"
        )
        self._update_subtitle()

    def deselect_all(self) -> None:
        """Clear all selections."""
        self.selected_candidate_ids.clear()

        try:
            candidate_list = self.query_one(CandidateList)
            candidate_list.selected_ids = set()
            candidate_list.update_selection_checkboxes()
        except Exception:
            pass

        self.notify("Cleared selection", severity="information")
        self._update_subtitle()

    @work(thread=True)
    def approve_candidate(self, candidate: EntityCandidate | RelationshipCandidate) -> None:
        """Approve a candidate and update UI.

        Runs in worker thread to avoid blocking UI.

        Args:
            candidate: The candidate to approve
        """
        service = self.get_curation_service()
        try:
            if isinstance(candidate, EntityCandidate):
                entity_id = service.approve_candidate(candidate)
                msg = f"✓ Approved: {candidate.canonical_name} (ID: {entity_id})"

                try:
                    self.call_from_thread(
                        self.notify,
                        "Scanning neighborhood for connections...",
                        timeout=2,
                    )
                    issues = get_neighborhood_issues(
                        service, candidate.canonical_name, candidate.aliases
                    )
                    self.call_from_thread(
                        self._handle_entity_approved_ui,
                        msg,
                        issues,
                    )
                    return
                except Exception as e:
                    logger.error(f"Failed to check neighborhood issues: {e}")
            else:
                service.approve_relationship_candidate(candidate)
                msg = f"✓ Approved: {candidate.source} -> {candidate.target}"

            def on_success() -> None:
                self.session_tracker.record_approval()
                self.approved_count += 1
                self._check_milestones()
                self.load_candidates(
                    preserve_index=True,
                    notification=msg,
                    show_loaded_notification=False,
                )

            self.call_from_thread(on_success)

        except Exception as e:
            self.call_from_thread(
                self.notify, f"✗ Error approving candidate: {e}", severity="error", markup=False
            )
        finally:
            service.manager.close()

    def _handle_entity_approved_ui(self, msg: str, issues: List[NeighborhoodIssue]) -> None:
        """Update UI after approving an entity candidate.

        If neighborhood issues exist, show the modal before reloading candidates. Reloading triggers
        a recompose that can otherwise hide/dismiss modal screens.
        """
        self.session_tracker.record_approval()
        self.approved_count += 1
        self._check_milestones()

        if issues:
            self.push_screen(
                NeighborhoodResolutionModal(issues),
                lambda selected: self._after_neighborhood_modal(selected, msg),
            )
            return

        self.load_candidates(
            preserve_index=True,
            notification=msg,
            show_loaded_notification=False,
        )

    def _after_neighborhood_modal(self, selected_issues: List[NeighborhoodIssue], msg: str) -> None:
        """Handle the neighborhood modal dismissal and reload candidates."""
        if selected_issues:
            self.process_neighborhood_resolutions(selected_issues, post_notification=msg)
            return

        self.load_candidates(
            preserve_index=True,
            notification=msg,
            show_loaded_notification=False,
        )

    def _show_neighborhood_modal(self, issues: List[NeighborhoodIssue]) -> None:
        """Show the neighborhood resolution modal."""
        self.push_screen(
            NeighborhoodResolutionModal(issues),
            self._handle_neighborhood_resolution,
        )

    def _handle_neighborhood_resolution(self, selected_issues: List[NeighborhoodIssue]) -> None:
        """Handle the result from the neighborhood resolution modal."""
        if selected_issues:
            self.process_neighborhood_resolutions(selected_issues)

    @work(thread=True)
    def process_neighborhood_resolutions(
        self,
        issues: List[NeighborhoodIssue],
        *,
        post_notification: str | None = None,
    ) -> None:
        """Process selected neighborhood resolutions."""
        service = self.get_curation_service()

        def candidate_store_factory() -> Neo4jManager:
            return Neo4jManager(self.config.database)

        try:
            count = 0
            # Reuse logic similar to CLI recursive check, but simplified for batch processing
            # We need a candidate store to fetch candidates by key
            manager = candidate_store_factory()
            manager.connect()

            try:
                for issue in issues:
                    if issue.issue_type == "promotable":
                        service.approve_relationship_candidate(issue.relationship_candidate)
                        count += 1

                    elif issue.issue_type == "resolvable" and issue.peer_candidate_key:
                        # Fetch the candidate
                        cand_dict = manager.get_entity_candidate(issue.peer_candidate_key)
                        if cand_dict:
                            peer_cand = EntityCandidate.model_validate(cand_dict)
                            service.approve_candidate(peer_cand)
                            # Note: We are NOT recursing here in the TUI for simplicity,
                            # or we could collect new issues and pop up another modal?
                            # For now, let's keep it one-level deep.
                            count += 1

                    elif issue.issue_type == "missing":
                        # Create synthetic candidate
                        # For TUI, we default to CONCEPT as per modal logic (or we could improve this)
                        new_type = EntityType.CONCEPT
                        synthetic = EntityCandidate(
                            candidate_key=issue.peer_name.lower().replace(" ", "_"),
                            canonical_name=issue.peer_name,
                            candidate_type=new_type,
                            status=CandidateStatus.PENDING,
                            confidence_score=1.0,
                            description="Created during neighborhood approval",
                            source_documents=issue.relationship_candidate.source_documents,
                            chunk_ids=issue.relationship_candidate.chunk_ids,
                        )
                        service.approve_candidate(synthetic)
                        count += 1

                def on_finish():
                    if post_notification:
                        self.notify(post_notification, severity="information")
                    self.notify(f"Resolved {count} neighborhood issues", severity="information")
                    # Reload to reflect changes (e.g. relationships might now be approved)
                    self.load_candidates(preserve_index=True, show_loaded_notification=False)

                self.call_from_thread(on_finish)

            finally:
                manager.close()

        except Exception as e:
            self.call_from_thread(
                self.notify, f"Error resolving issues: {e}", severity="error", markup=False
            )
        finally:
            service.manager.close()

    @work(thread=True)
    def reject_candidate(
        self, candidate: EntityCandidate | RelationshipCandidate, reason: str = ""
    ) -> None:
        """Reject a candidate and update UI.

        Runs in worker thread to avoid blocking UI.

        Args:
            candidate: The candidate to reject
            reason: Optional reason for rejection
        """
        service = self.get_curation_service()
        try:
            if isinstance(candidate, EntityCandidate):
                service.reject_candidate(candidate, reason=reason)
                name = candidate.canonical_name
            else:
                service.reject_relationship_candidate(candidate, reason=reason)
                name = f"{candidate.source} -> {candidate.target}"

            def on_success() -> None:
                self.session_tracker.record_rejection()
                self.rejected_count += 1
                self._check_milestones()
                self.load_candidates(
                    preserve_index=True,
                    notification=f"✗ Rejected: {name}",
                    show_loaded_notification=False,
                )

            self.call_from_thread(on_success)

        except Exception as e:
            self.call_from_thread(
                self.notify, f"✗ Error rejecting candidate: {e}", severity="error", markup=False
            )
        finally:
            service.manager.close()

    @work(thread=True)
    def flag_candidate(self, candidate: EntityCandidate | RelationshipCandidate) -> None:
        """Flag a candidate for later review.

        This is implemented as a temporary status change until proper
        flag support is added to the schema.

        Args:
            candidate: The candidate to flag
        """
        name = (
            candidate.canonical_name
            if isinstance(candidate, EntityCandidate)
            else f"{candidate.source} -> {candidate.target}"
        )

        def on_flag() -> None:
            self.session_tracker.record_flag()
            self.notify(
                f"🚩 Flagged: {name} (feature coming soon)",
                severity="information",
            )

        self.call_from_thread(on_flag)

    @work(thread=True)
    def undo_last_operation(self) -> None:
        """Undo the last curation operation.

        Runs in worker thread to avoid blocking UI.
        """
        service = self.get_curation_service()
        try:
            success = service.undo_last_operation()

            if success:

                def on_success() -> None:
                    self.session_tracker.record_undo()

                    # Adjust counts based on what was undone.
                    # Note: We don't know exactly what was undone, so this is approximate.
                    if self.approved_count > 0:
                        self.approved_count -= 1
                    elif self.rejected_count > 0:
                        self.rejected_count -= 1

                    self.load_candidates(
                        preserve_index=True,
                        notification="↶ Undo successful",
                        show_loaded_notification=False,
                    )

                self.call_from_thread(on_success)
            else:
                self.call_from_thread(self.notify, "No operation to undo", severity="warning")

        except Exception as e:
            self.call_from_thread(
                self.notify, f"✗ Error undoing operation: {e}", severity="error", markup=False
            )
        finally:
            service.manager.close()

    @work(thread=True)
    def edit_candidate(self, original: EntityCandidate, updated: EntityCandidate) -> None:
        """Edit a candidate and update UI.

        Runs in worker thread to avoid blocking UI.

        Args:
            candidate: The updated candidate to save
        """
        updates: dict[str, object] = {}
        for field in (
            "canonical_name",
            "aliases",
            "description",
            "candidate_type",
            "confidence_score",
        ):
            if getattr(original, field) != getattr(updated, field):
                updates[field] = getattr(updated, field)

        if not updates:
            self.call_from_thread(self.notify, "No changes to save", severity="information")
            return

        service = self.get_curation_service()
        try:
            service.edit_candidate(original, updates)

            def on_success() -> None:
                self.session_tracker.record_edit()
                self.edited_count += 1
                self.load_candidates(
                    preserve_index=True,
                    notification=f"✎ Edited: {updated.canonical_name}",
                    show_loaded_notification=False,
                )

            self.call_from_thread(on_success)

        except Exception as e:
            self.call_from_thread(
                self.notify, f"✗ Error editing candidate: {e}", severity="error", markup=False
            )
        finally:
            service.manager.close()

    @work(thread=True)
    def batch_approve_candidates(
        self, candidates: List[EntityCandidate | RelationshipCandidate]
    ) -> None:
        """Batch approve multiple candidates.

        Runs in worker thread to avoid blocking UI.

        Args:
            candidates: List of candidates to approve
        """
        if not candidates:
            self.call_from_thread(self.notify, "No candidates to approve", severity="warning")
            return

        # Split candidates by type
        entity_candidates = [c for c in candidates if isinstance(c, EntityCandidate)]
        rel_candidates = [c for c in candidates if isinstance(c, RelationshipCandidate)]

        service = self.get_curation_service()
        try:
            approved_count = 0

            # 1. Handle entities using BatchCurationService
            if entity_candidates:
                # Use BatchCurationService for transaction support
                batch_service = BatchCurationService(
                    curation_service=service,
                    config=self.config.curation if self.config else None,
                )

                # Execute batch approve (with checkpoint/rollback)
                result = batch_service.batch_approve(
                    entity_candidates, threshold=0.0
                )  # No threshold, approve all selected
                approved_count += len(result.approved_entities)

            # 2. Handle relationships individually (for now)
            if rel_candidates:
                for rc in rel_candidates:
                    service.approve_relationship_candidate(rc)
                    approved_count += 1

            def on_success() -> None:
                for _ in range(approved_count):
                    self.session_tracker.record_approval()
                    self.approved_count += 1

                # Clear selection after batch operation
                self.selected_candidate_ids.clear()
                self.selection_mode = False

                self.load_candidates(
                    preserve_index=True,
                    notification=f"✓ Batch approved {approved_count} candidates",
                    show_loaded_notification=False,
                )

            self.call_from_thread(on_success)

        except Exception as e:
            self.call_from_thread(
                self.notify, f"✗ Error in batch approve: {e}", severity="error", markup=False
            )
        finally:
            service.manager.close()

    @work(thread=True)
    def batch_reject_candidates(
        self, candidates: List[EntityCandidate | RelationshipCandidate]
    ) -> None:
        """Batch reject multiple candidates.

        Runs in worker thread to avoid blocking UI.

        Args:
            candidates: List of candidates to reject
        """
        if not candidates:
            self.call_from_thread(self.notify, "No candidates to reject", severity="warning")
            return

        service = self.get_curation_service()
        try:
            # Create checkpoint for rollback support
            checkpoint = service.undo_checkpoint()

            rejected_count = 0
            try:
                for candidate in candidates:
                    if candidate.status == CandidateStatus.PENDING:
                        if isinstance(candidate, EntityCandidate):
                            service.reject_candidate(candidate, reason="Batch rejection")
                        else:
                            service.reject_relationship_candidate(
                                candidate, reason="Batch rejection"
                            )
                        rejected_count += 1
            except Exception as e:
                # Rollback on error
                service.rollback_to_checkpoint(checkpoint)
                raise e

            def on_success() -> None:
                for _ in range(rejected_count):
                    self.session_tracker.record_rejection()
                    self.rejected_count += 1

                # Clear selection after batch operation
                self.selected_candidate_ids.clear()
                self.selection_mode = False

                self.load_candidates(
                    preserve_index=True,
                    notification=f"✗ Batch rejected {rejected_count} candidates",
                    show_loaded_notification=False,
                )

            self.call_from_thread(on_success)

        except Exception as e:
            self.call_from_thread(
                self.notify, f"✗ Error in batch reject: {e}", severity="error", markup=False
            )
        finally:
            service.manager.close()

    @work(thread=True)
    def merge_candidates_workflow(
        self,
        primary: EntityCandidate,
        duplicates: List[EntityCandidate],
    ) -> None:
        """Merge multiple candidates into a single entity.

        Runs in worker thread to avoid blocking UI.

        Args:
            primary: The primary candidate (will become the entity)
            duplicates: The duplicate candidates (will be rejected)
        """
        if not duplicates:
            # Only one candidate, just approve it
            self.approve_candidate(primary)
            return

        service = self.get_curation_service()
        try:
            # Use BatchCurationService for transaction support
            batch_service = BatchCurationService(
                curation_service=service,
                config=self.config.curation if self.config else None,
            )

            # Execute merge via batch_merge_clusters
            all_candidates = [primary] + duplicates
            batch_service.batch_merge_clusters([[primary] + duplicates])

            def on_success() -> None:
                # Record merge in session statistics
                self.session_tracker.record_merge()
                self.session_tracker.record_approval()  # Primary becomes entity
                for _ in duplicates:
                    self.session_tracker.record_rejection()  # Duplicates are rejected

                self.approved_count += 1

                # Clear selection after merge
                self.selected_candidate_ids.clear()
                self.selection_mode = False

                self.load_candidates(
                    preserve_index=True,
                    notification=f"✓ Merged {len(all_candidates)} candidates into: {primary.canonical_name}",
                    show_loaded_notification=False,
                )

            self.call_from_thread(on_success)

        except Exception as e:
            self.call_from_thread(
                self.notify, f"✗ Error in merge: {e}", severity="error", markup=False
            )
        finally:
            service.manager.close()

    # Action handlers (called by key bindings)

    def action_toggle_review_mode(self) -> None:
        """Toggle between entity and relationship review mode."""
        if self.review_mode == "entities":
            self.review_mode = "relationships"
            self.notify("Switched to Relationship Review Mode", severity="information")
        else:
            self.review_mode = "entities"
            self.notify("Switched to Entity Review Mode", severity="information")

        # Clear selections and reload
        self.selected_candidate_ids.clear()
        self.selection_mode = False
        self.load_candidates(show_loading=True)

    def action_approve_current(self) -> None:
        """Handle approve action (bound to 'a' key)."""
        if self.current_candidate:
            if self.current_candidate.status == CandidateStatus.APPROVED:
                self.notify("Already approved", severity="warning")
            else:
                self.approve_candidate(self.current_candidate)
        else:
            self.notify("No candidate selected", severity="warning")

    def action_reject_current(self) -> None:
        """Handle reject action (bound to 'r' key)."""
        if self.current_candidate:
            if self.current_candidate.status == CandidateStatus.REJECTED:
                self.notify("Already rejected", severity="warning")
            else:
                self.reject_candidate(self.current_candidate)
        else:
            self.notify("No candidate selected", severity="warning")

    def action_flag_current(self) -> None:
        """Handle flag action (bound to 'f' key)."""
        if self.current_candidate:
            self.flag_candidate(self.current_candidate)
        else:
            self.notify("No candidate selected", severity="warning")

    def action_undo_last(self) -> None:
        """Handle undo action (bound to 'u' key)."""
        self.undo_last_operation()

    def action_edit_current(self) -> None:
        """Handle edit action (bound to 'e' key)."""
        if self.review_mode == "relationships":
            self.notify("Editing relationships is not supported yet", severity="warning")
            return

        if self.current_candidate and isinstance(self.current_candidate, EntityCandidate):
            original = self.current_candidate

            def on_save(updated: EntityCandidate) -> None:
                self.edit_candidate(original, updated)

            # Open edit modal and handle result
            self.push_screen(
                EditModalScreen(original, on_save=on_save),
                self._handle_edit_result,
            )
        else:
            self.notify("No candidate selected", severity="warning")

    def _handle_edit_result(self, result: Optional[EntityCandidate]) -> None:
        """Handle the result from the edit modal.

        Args:
            result: The edited candidate, or None if cancelled
        """
        if result:
            # The modal's on_save callback already called edit_candidate
            # This is just for logging/debugging
            pass
        else:
            # User cancelled the edit
            self.notify("Edit cancelled", severity="information")

    def action_merge_into_entity(self) -> None:
        """Handle merge into existing entity action (bound to 'Shift+M' key)."""
        if self.review_mode == "relationships":
            self.notify("Merging relationships is not supported", severity="warning")
            return

        if not self.current_candidate:
            self.notify("No candidate selected", severity="warning")
            return

        if self.current_candidate.status != CandidateStatus.PENDING:
            self.notify("Can only merge PENDING candidates", severity="warning")
            return

        # Open search modal to find target entity
        def on_entity_selected(entity_data: Optional[Dict[str, Any]]) -> None:
            if entity_data:
                self._handle_merge_entity_search_result(entity_data)

        if isinstance(self.current_candidate, EntityCandidate):
            self.push_screen(
                EntitySearchModal(self.config, initial_query=self.current_candidate.canonical_name),
                on_entity_selected,
            )

    def _handle_merge_entity_search_result(self, entity_data: Dict[str, Any]) -> None:
        """Handle selected target entity from search."""
        candidate = self.current_candidate

        def on_confirm(confirmed: bool) -> None:
            if confirmed and isinstance(candidate, EntityCandidate):
                self.merge_into_entity_workflow(entity_data, candidate)

        if isinstance(candidate, EntityCandidate):
            self.push_screen(EntityCandidateMergePreviewModal(entity_data, candidate), on_confirm)

    @work(thread=True)
    def merge_into_entity_workflow(
        self,
        entity_data: Dict[str, Any],
        candidate: EntityCandidate,
    ) -> None:
        """Execute merge of candidate into existing entity."""
        service = self.get_curation_service()
        try:
            entity_id = entity_data.get("id")
            success = service.merge_candidate_into_entity(entity_id, candidate)

            if success:

                def on_success() -> None:
                    self.session_tracker.record_merge()
                    self.notify(
                        f"✓ Merged into entity: {entity_data.get('canonical_name')}",
                        severity="information",
                    )
                    self.load_candidates(preserve_index=True, show_loaded_notification=False)

                self.call_from_thread(on_success)
            else:
                self.call_from_thread(
                    self.notify, "✗ Failed to merge into entity", severity="error"
                )

        except Exception as e:
            self.call_from_thread(
                self.notify, f"✗ Error in merge to entity: {e}", severity="error", markup=False
            )
        finally:
            service.manager.close()

    def action_search(self) -> None:
        """Handle search action (bound to '/' key)."""
        # Open search modal with current filters
        self.push_screen(
            SearchModalScreen(self.current_filters, on_search=self._execute_search),
            self._handle_search_result,
        )

    def _handle_search_result(self, result: Optional[SearchFilters]) -> None:
        """Handle the result from the search modal.

        Args:
            result: The search filters, or None if cancelled
        """
        if result:
            # The modal's on_search callback already called _execute_search
            pass
        else:
            # User cancelled the search
            pass

    def _execute_search(self, filters: SearchFilters) -> None:
        """Execute search with given filters.

        Args:
            filters: The search and filter criteria
        """
        self.current_filters = filters
        self.search_results = []
        self.current_search_result_index = -1

        # Update filter_status to match search filters
        self.filter_status = filters.status

        # Reload candidates with new filters
        self.load_candidates(
            preserve_index=False,
            show_loaded_notification=True,
        )

    def action_next_search(self) -> None:
        """Handle next search result (bound to 'n' key)."""
        if not self.search_results:
            self.notify("No search results. Press '/' to search.", severity="information")
            return

        # Move to next result
        self.current_search_result_index = (self.current_search_result_index + 1) % len(
            self.search_results
        )
        self.current_index = self.search_results[self.current_search_result_index]

        self.notify(
            f"Result {self.current_search_result_index + 1}/{len(self.search_results)}",
            severity="information",
        )

    def action_prev_search(self) -> None:
        """Handle previous search result (bound to 'N' key)."""
        if not self.search_results:
            self.notify("No search results. Press '/' to search.", severity="information")
            return

        # Move to previous result
        self.current_search_result_index = (self.current_search_result_index - 1) % len(
            self.search_results
        )
        self.current_index = self.search_results[self.current_search_result_index]

        self.notify(
            f"Result {self.current_search_result_index + 1}/{len(self.search_results)}",
            severity="information",
        )

    def action_help(self) -> None:
        """Show help information."""
        self.notify("Help screen coming soon", severity="information")

    def action_compare_with_duplicate(self) -> None:
        """Compare current candidate with first duplicate suggestion."""
        if self.review_mode == "relationships":
            self.notify("Duplicate comparison not supported for relationships", severity="warning")
            return

        if not self.current_candidate:
            self.notify("No candidate selected", severity="warning")
            return

        # Get duplicate suggestions from the panel
        try:
            dup_panel = self.query_one(DuplicateSuggestionsPanel)
            if not dup_panel.suggestions:
                self.notify("No duplicate suggestions available", severity="information")
                return

            # Get first suggestion
            duplicate_candidate, similarity, reason = dup_panel.suggestions[0]

            # Open comparison modal
            def on_merge(primary: EntityCandidate, duplicate: EntityCandidate) -> None:
                """Handle merge from comparison modal."""
                self.merge_candidates_workflow(primary, [duplicate])

            if isinstance(self.current_candidate, EntityCandidate):
                self.push_screen(
                    ComparisonModalScreen(
                        self.current_candidate,
                        duplicate_candidate,
                        on_merge=on_merge,
                    ),
                    self._handle_comparison_result,
                )

        except Exception as e:
            self.notify(f"Error opening comparison: {e}", severity="error", markup=False)

    def _handle_comparison_result(self, result: Optional[str]) -> None:
        """Handle the result from the comparison modal.

        Args:
            result: 'merge_a_b', 'merge_b_a', 'select_different', or None (cancelled)
        """
        if result in ("merge_a_b", "merge_b_a"):
            # Merge was already handled by the on_merge callback
            pass
        elif result == "select_different":
            self.notify("Select different candidates - feature coming soon", severity="information")
        else:
            # Cancelled
            pass

    def action_command_mode(self) -> None:
        """Open command mode (bound to ':' key)."""
        self.push_screen(
            CommandModalScreen(self.command_history, on_execute=self._execute_command),
            self._handle_command_result,
        )

    def _handle_command_result(self, result: Optional[ParsedCommand]) -> None:
        """Handle the result from the command modal.

        Args:
            result: The parsed command, or None if cancelled
        """
        if result:
            # The modal's on_execute callback already called _execute_command
            pass
        else:
            # User cancelled the command
            pass

    def _execute_command(self, parsed: ParsedCommand) -> None:
        """Execute a parsed command.

        Args:
            parsed: Parsed command to execute
        """
        command = parsed.command

        try:
            if command == "filter":
                self._command_filter(parsed)
            elif command == "sort":
                self._command_sort(parsed)
            elif command in ("batch-approve", "batch_approve"):
                self._command_batch_approve(parsed)
            elif command in ("batch-reject", "batch_reject"):
                self._command_batch_reject(parsed)
            elif command == "export":
                self._command_export(parsed)
            elif command == "help":
                self._command_help(parsed)
            elif command in ("quit", "q"):
                self._command_quit(parsed)
            else:
                self.notify(f"Unknown command: {command}", severity="error")
        except Exception as e:
            self.notify(f"Command error: {e}", severity="error", markup=False)

    def _command_filter(self, parsed: ParsedCommand) -> None:
        """Execute filter command.

        Args:
            parsed: Parsed command with filter arguments
        """
        from src.curation.interactive.command_parser import CommandParser

        parser = CommandParser()
        try:
            # Validate and normalize filter arguments
            validated = parser.validate_filter_args(parsed.kwargs)

            # Create search filters
            filters = SearchFilters(
                search_text="",
                status=validated.get("status", self.filter_status),
                entity_type=validated.get("entity_type"),
                min_confidence=validated.get("min_confidence", 0.0),
            )

            # Apply filters
            self._execute_search(filters)
            self.notify(f"Applied filters: {parsed.raw_input}", severity="information")

        except ValueError as e:
            self.notify(f"Filter error: {e}", severity="error", markup=False)

    def _command_sort(self, parsed: ParsedCommand) -> None:
        """Execute sort command.

        Args:
            parsed: Parsed command with sort field
        """
        from src.curation.interactive.command_parser import CommandParser

        parser = CommandParser()
        try:
            # Validate sort field
            field = parser.validate_sort_args(parsed.args)

            # Sorting is not yet implemented in the backend
            self.notify(
                f"Sort by '{field}' - Feature coming soon!",
                severity="information",
            )

        except ValueError as e:
            self.notify(f"Sort error: {e}", severity="error", markup=False)

    def _command_batch_approve(self, parsed: ParsedCommand) -> None:
        """Execute batch-approve command.

        Args:
            parsed: Parsed command with confidence threshold
        """
        from src.curation.interactive.command_parser import CommandParser

        parser = CommandParser()

        if not parsed.args:
            self.notify("batch-approve requires a threshold (e.g., >0.9)", severity="error")
            return

        try:
            # Parse threshold
            operator, threshold = parser.parse_confidence_threshold(parsed.args[0])

            # Filter candidates by confidence
            high_confidence = []
            for candidate in self.candidates:
                if candidate.status == CandidateStatus.PENDING:
                    if operator == ">" and candidate.confidence_score > threshold:
                        high_confidence.append(candidate)
                    elif operator == ">=" and candidate.confidence_score >= threshold:
                        high_confidence.append(candidate)
                    elif operator == "==" and candidate.confidence_score == threshold:
                        high_confidence.append(candidate)

            if not high_confidence:
                self.notify(
                    f"No pending candidates with confidence {operator}{threshold}",
                    severity="warning",
                )
                return

            # Show confirmation modal with closure to capture candidates
            def handle_result(confirmed: bool) -> None:
                if confirmed:
                    self.batch_approve_candidates(high_confidence)
                else:
                    self.notify("Batch approve cancelled", severity="information")

            self.push_screen(
                BatchOperationModal("approve", high_confidence),
                handle_result,
            )

        except ValueError as e:
            self.notify(f"batch-approve error: {e}", severity="error", markup=False)

    def _command_batch_reject(self, parsed: ParsedCommand) -> None:
        """Execute batch-reject command.

        Args:
            parsed: Parsed command with confidence threshold
        """
        from src.curation.interactive.command_parser import CommandParser

        parser = CommandParser()

        if not parsed.args:
            self.notify("batch-reject requires a threshold (e.g., <0.5)", severity="error")
            return

        try:
            # Parse threshold
            operator, threshold = parser.parse_confidence_threshold(parsed.args[0])

            # Filter candidates by confidence
            low_confidence = []
            for candidate in self.candidates:
                if candidate.status == CandidateStatus.PENDING:
                    if operator == "<" and candidate.confidence_score < threshold:
                        low_confidence.append(candidate)
                    elif operator == "<=" and candidate.confidence_score <= threshold:
                        low_confidence.append(candidate)
                    elif operator == "==" and candidate.confidence_score == threshold:
                        low_confidence.append(candidate)

            if not low_confidence:
                self.notify(
                    f"No pending candidates with confidence {operator}{threshold}",
                    severity="warning",
                )
                return

            # Show confirmation modal with closure to capture candidates
            def handle_result(confirmed: bool) -> None:
                if confirmed:
                    self.batch_reject_candidates(low_confidence)
                else:
                    self.notify("Batch reject cancelled", severity="information")

            self.push_screen(
                BatchOperationModal("reject", low_confidence),
                handle_result,
            )

        except ValueError as e:
            self.notify(f"batch-reject error: {e}", severity="error", markup=False)

    def _command_export(self, parsed: ParsedCommand) -> None:
        """Execute export command.

        Args:
            parsed: Parsed command with export filename
        """
        if not parsed.args:
            self.notify("export requires a filename (e.g., export results.json)", severity="error")
            return

        filename = parsed.args[0]

        # Export is not yet implemented
        self.notify(
            f"Export to '{filename}' - Feature coming soon!",
            severity="information",
        )

    def _command_help(self, parsed: ParsedCommand) -> None:
        """Execute help command.

        Args:
            parsed: Parsed command
        """
        help_text = (
            "\n📖 Command Mode Help\n"
            + "=" * 40
            + "\n"
            + "filter [key=value...]  - Apply filters\n"
            + "  Examples:\n"
            + "    :filter type=SYSTEM\n"
            + "    :filter status=pending confidence=0.8\n"
            + "    :filter type=COMPONENT status=approved\n"
            + "\n"
            + "sort <field>           - Sort by field\n"
            + "  Fields: confidence, name, type, mentions\n"
            + "  Example: :sort confidence\n"
            + "\n"
            + "batch-approve <threshold> - Approve high confidence\n"
            + "  Example: :batch-approve >0.9\n"
            + "\n"
            + "batch-reject <threshold>  - Reject low confidence\n"
            + "  Example: :batch-reject <0.5\n"
            + "\n"
            + "export <file>          - Export results\n"
            + "  Example: :export results.json\n"
            + "\n"
            + "help                   - Show this help\n"
            + "quit, q                - Quit application\n"
            + "=" * 40
            + "\n"
        )
        self.notify(help_text, severity="information", timeout=15)

    def _command_quit(self, parsed: ParsedCommand) -> None:
        """Execute quit command.

        Args:
            parsed: Parsed command
        """
        self.action_quit()

    def action_quit(self) -> None:
        """Handle quit action - show session summary before exiting."""
        # Save final session state
        self._save_session_state()

        # Show summary
        self._show_session_summary()

        # Wait for the notification to display, then quit
        self.set_timer(5.0, self.exit)

    # Selection mode actions

    def action_toggle_selection_mode(self) -> None:
        """Toggle visual selection mode (bound to 'v' key)."""
        self.selection_mode = not self.selection_mode

        if self.selection_mode:
            self.notify(
                "Selection mode ON - Use Space to select, A/R for batch operations, v to exit",
                severity="information",
                timeout=8,
            )
        else:
            # Exit selection mode - keep selections
            count = len(self.selected_candidate_ids)
            self.notify(
                f"Selection mode OFF - {count} candidates selected",
                severity="information",
            )

    def action_toggle_current_selection(self) -> None:
        """Toggle selection of current candidate (bound to Space key)."""
        if not self.selection_mode:
            self.notify("Press 'v' to enter selection mode first", severity="warning")
            return

        self.toggle_current_selection()

        # Show feedback
        if self.current_candidate:
            is_selected = self.current_candidate.id in self.selected_candidate_ids
            status = "selected" if is_selected else "deselected"
            self.notify(
                f"{status.capitalize()}: {self.current_candidate.canonical_name}",
                severity="information",
                timeout=2,
            )

    def action_select_all(self) -> None:
        """Select all visible candidates."""
        if not self.selection_mode:
            self.notify("Press 'v' to enter selection mode first", severity="warning")
            return

        self.select_all_visible()

    def action_deselect_all(self) -> None:
        """Deselect all candidates."""
        if not self.selection_mode:
            self.notify("Press 'v' to enter selection mode first", severity="warning")
            return

        self.deselect_all()

    def action_batch_approve(self) -> None:
        """Batch approve selected candidates (bound to 'A' key)."""
        if not self.selected_candidate_ids:
            self.notify(
                "No candidates selected. Press 'v' to enter selection mode.", severity="warning"
            )
            return

        # Get selected candidates
        selected = self.get_selected_candidates()
        if not selected:
            self.notify("No valid candidates selected", severity="warning")
            return

        # Show batch operation preview modal
        self.push_screen(
            BatchOperationModal("approve", selected),
            self._handle_batch_approve_result,
        )

    def _handle_batch_approve_result(self, confirmed: bool) -> None:
        """Handle the result from the batch approve modal.

        Args:
            confirmed: Whether the user confirmed the operation
        """
        if confirmed:
            selected = self.get_selected_candidates()
            self.batch_approve_candidates(selected)
        else:
            self.notify("Batch approve cancelled", severity="information")

    def action_batch_reject(self) -> None:
        """Batch reject selected candidates (bound to 'R' key)."""
        if not self.selected_candidate_ids:
            self.notify(
                "No candidates selected. Press 'v' to enter selection mode.", severity="warning"
            )
            return

        # Get selected candidates
        selected = self.get_selected_candidates()
        if not selected:
            self.notify("No valid candidates selected", severity="warning")
            return

        # Show batch operation preview modal
        self.push_screen(
            BatchOperationModal("reject", selected),
            self._handle_batch_reject_result,
        )

    def _handle_batch_reject_result(self, confirmed: bool) -> None:
        """Handle the result from the batch reject modal.

        Args:
            confirmed: Whether the user confirmed the operation
        """
        if confirmed:
            selected = self.get_selected_candidates()
            self.batch_reject_candidates(selected)
        else:
            self.notify("Batch reject cancelled", severity="information")

    def action_merge_candidates(self) -> None:
        """Handle merge candidates action (bound to 'M' key)."""
        if self.review_mode == "relationships":
            self.notify("Merging relationships is not supported", severity="warning")
            return

        if not self.selected_candidate_ids:
            self.notify(
                "No candidates selected. Press 'v' to enter selection mode.", severity="warning"
            )
            return

        # Get selected candidates
        selected = self.get_selected_candidates()
        if len(selected) < 2:
            self.notify("Select at least 2 candidates to merge", severity="warning")
            return

        # Filter to PENDING only
        pending_candidates = [c for c in selected if c.status == CandidateStatus.PENDING]
        if len(pending_candidates) < 2:
            filtered_count = len(selected) - len(pending_candidates)
            self.notify(
                f"⚠️ {filtered_count} candidates filtered (not PENDING). Need at least 2 PENDING candidates.",
                severity="warning",
            )
            return

        # Check for type conflicts (warn but allow) (ONLY FOR ENTITIES)
        if all(isinstance(c, EntityCandidate) for c in pending_candidates):
            types = {c.candidate_type for c in pending_candidates}
            if len(types) > 1:
                type_list = ", ".join(t.value for t in types)
                self.notify(
                    f"⚠️ Warning: Candidates have different types: {type_list}",
                    severity="warning",
                    timeout=5,
                )

        # Store candidates for modal handlers
        self._merge_candidates = pending_candidates

        # Show primary selection modal
        if all(isinstance(c, EntityCandidate) for c in pending_candidates):
            self.push_screen(
                PrimarySelectionModal(pending_candidates),
                self._handle_primary_selection_result,
            )

    def _handle_primary_selection_result(self, primary: Optional[EntityCandidate]) -> None:
        """Handle the result from the primary selection modal.

        Args:
            primary: The selected primary candidate, or None if cancelled
        """
        if primary is None:
            self.notify("Merge cancelled", severity="information")
            return

        # Get duplicates (all except primary)
        duplicates = [c for c in self._merge_candidates if c.id != primary.id]

        # Store for preview modal handler
        self._merge_primary = primary
        self._merge_duplicates = duplicates

        # Show merge preview modal
        self.push_screen(
            MergePreviewModal(primary, duplicates),
            self._handle_merge_preview_result,
        )

    def _handle_merge_preview_result(self, result: Optional[str]) -> None:
        """Handle the result from the merge preview modal.

        Args:
            result: 'confirm' to proceed, 'change_primary' to reselect, None to cancel
        """
        if result == "confirm":
            # Execute merge
            self.merge_candidates_workflow(self._merge_primary, self._merge_duplicates)
        elif result == "change_primary":
            # Go back to primary selection
            self.push_screen(
                PrimarySelectionModal(self._merge_candidates),
                self._handle_primary_selection_result,
            )
        else:
            # Cancelled
            self.notify("Merge cancelled", severity="information")

    def watch_current_index(self, old: int, new: int) -> None:
        """React to current index changes by updating detail panel and widget.

        This is a watch method that automatically runs when current_index changes.
        """
        if old != new:
            # Update detail panel with new candidate
            if self.current_candidate:
                try:
                    detail_panel = self.query_one(DetailPanel)
                    detail_panel.update_candidate(self.current_candidate)
                except Exception:
                    # Detail panel doesn't exist yet
                    pass

                # Update duplicate suggestions panel
                try:
                    dup_panel = self.query_one(DuplicateSuggestionsPanel)
                    if isinstance(self.current_candidate, EntityCandidate):
                        dup_panel.update_suggestions(self.current_candidate, self.candidates)
                    else:
                        dup_panel.clear_suggestions()  # Or implement relationship dup logic
                except Exception:
                    # Panel doesn't exist yet
                    pass

            # Sync the widget's index with the app's index
            try:
                candidate_list = self.query_one(CandidateList)
                if candidate_list.current_index != new:
                    candidate_list.current_index = new
            except Exception:
                # Widget doesn't exist yet
                pass

    def watch_filter_status(self, old: str, new: str) -> None:
        """React to filter status changes by reloading candidates.

        This is a watch method that automatically runs when filter_status changes.
        """
        if old != new:
            self.load_candidates()

    def watch_selection_mode(self, old: bool, new: bool) -> None:
        """React to selection mode changes by updating subtitle.

        This is a watch method that automatically runs when selection_mode changes.
        """
        self._update_subtitle()

    def watch_review_mode(self, old: str, new: str) -> None:
        """React to review mode changes by updating subtitle."""
        self._update_subtitle()

    def _update_subtitle(self) -> None:
        """Update the subtitle with current state information."""
        mode_str = "ENTITIES" if self.review_mode == "entities" else "RELATIONSHIPS"
        parts = [
            f"Mode: {mode_str}",
            f"Status: {self.filter_status}",
            f"Total: {len(self.candidates)}",
        ]

        if self.selection_mode:
            parts.append(f"[SELECTION MODE] {len(self.selected_candidate_ids)} selected")

        if self.review_mode == "entities":
            parts.append("Press 't' to switch to Relationships")
        else:
            parts.append("Press 't' to switch to Entities")

        self.sub_title = " | ".join(parts)


def run() -> None:
    """Entry point for the interactive review application."""
    app = ReviewApp()
    app.run()


if __name__ == "__main__":
    run()

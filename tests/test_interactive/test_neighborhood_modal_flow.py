"""Unit tests for interactive neighborhood modal flow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.curation.entity_approval import NeighborhoodIssue
from src.curation.interactive import ReviewApp


def _make_app() -> ReviewApp:
    with patch("src.utils.config.load_config") as mock_load:
        mock_config = MagicMock()
        mock_config.curation = MagicMock()
        mock_config.curation.enable_audit_trail = False
        mock_config.database = MagicMock()
        mock_load.return_value = mock_config

        with (
            patch("src.curation.interactive.app.Neo4jManager"),
            patch("src.curation.interactive.app.NormalizationTable"),
        ):
            return ReviewApp()


def test_entity_approval_shows_modal_before_reload() -> None:
    app = _make_app()
    app.notify = MagicMock()  # type: ignore[method-assign]
    app.push_screen = MagicMock()  # type: ignore[method-assign]
    app.load_candidates = MagicMock()  # type: ignore[method-assign]

    issue = MagicMock(spec=NeighborhoodIssue)
    app._handle_entity_approved_ui("ok", [issue])

    app.push_screen.assert_called_once()
    app.load_candidates.assert_not_called()


def test_entity_approval_without_issues_reloads_immediately() -> None:
    app = _make_app()
    app.notify = MagicMock()  # type: ignore[method-assign]
    app.push_screen = MagicMock()  # type: ignore[method-assign]
    app.load_candidates = MagicMock()  # type: ignore[method-assign]

    app._handle_entity_approved_ui("ok", [])

    app.load_candidates.assert_called_once()
    app.push_screen.assert_not_called()

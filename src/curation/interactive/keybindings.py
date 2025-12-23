"""Keyboard shortcut definitions for the interactive review interface."""

from textual.binding import Binding

# Navigation bindings (handled by CandidateList widget)
NAVIGATION_BINDINGS = [
    Binding("up", "cursor_up", "Up", show=False),
    Binding("down", "cursor_down", "Down", show=False),
    Binding("pageup", "page_up", "Page Up", show=False),
    Binding("pagedown", "page_down", "Page Down", show=False),
    Binding("home", "goto_first", "First", show=False),
    Binding("end", "goto_last", "Last", show=False),
]

# Action bindings (handled by ReviewApp)
ACTION_BINDINGS = [
    Binding("a", "approve_current", "Approve", show=True, priority=False),
    Binding("r", "reject_current", "Reject", show=True, priority=False),
    Binding("e", "edit_current", "Edit", show=True, priority=False),
    Binding("f", "flag_current", "Flag", show=True, priority=False),
    Binding("u", "undo_last", "Undo", show=True, priority=False),
    Binding("c", "compare_with_duplicate", "Compare", show=True, priority=False),
    Binding("m", "merge_into_entity", "Merge into Entity", show=True, priority=False),
]

# Search and filter bindings
SEARCH_BINDINGS = [
    Binding("/", "search", "Search", show=True, priority=False),
    Binding("n", "next_search", "Next", show=False),
    Binding("N", "prev_search", "Previous", show=False),
    Binding("colon", "command_mode", "Command", show=True, priority=False),
]

# Batch operation bindings (selection mode)
BATCH_BINDINGS = [
    Binding("v", "toggle_selection_mode", "Select Mode", show=True, priority=False),
    Binding("space", "toggle_current_selection", "Toggle Select", show=False),
    Binding("ctrl+a", "select_all", "Select All", show=False),
    Binding("ctrl+d", "deselect_all", "Deselect All", show=False),
    Binding("A", "batch_approve", "Batch Approve", show=True, priority=False),
    Binding("R", "batch_reject", "Batch Reject", show=True, priority=False),
    Binding("M", "merge_candidates", "Merge", show=True, priority=False),
]

# System bindings
SYSTEM_BINDINGS = [
    Binding("q", "quit", "Quit", show=True, priority=True),
    Binding("?", "help", "Help", show=True, priority=False),
]

# All bindings combined
ALL_BINDINGS = ACTION_BINDINGS + SEARCH_BINDINGS + BATCH_BINDINGS + SYSTEM_BINDINGS

"""Modal screen for searching existing entities for merging."""

from typing import Any, Dict, List, Optional

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Select

from src.storage.neo4j_manager import Neo4jManager
from src.storage.schemas import EntityStatus, EntityType
from src.utils.config import Config


class EntityResultItem(ListItem):
    """An item in the entity search results list."""

    def __init__(self, entity_data: Dict[str, Any]) -> None:
        super().__init__()
        self.entity_data = entity_data
        self.entity_id = entity_data.get("id")
        self.entity_name = entity_data.get("canonical_name", "Unknown")
        self.entity_type = entity_data.get("entity_type", "CONCEPT")

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]{self.entity_name}[/bold] ({self.entity_type})")
        if self.entity_data.get("description"):
            yield Label(f"[italic]{self.entity_data.get('description')[:80]}...[/italic]", classes="result-desc")


class EntitySearchModal(ModalScreen[Optional[Dict[str, Any]]]):
    """Modal screen for searching existing entities."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    CSS = """
    EntitySearchModal {
        align: center middle;
    }

    #search-container {
        width: 80;
        height: 35;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #search-title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        margin-bottom: 1;
    }

    #results-list {
        height: 1fr;
        border: solid $secondary;
        margin: 1 0;
    }

    .result-desc {
        color: $text-muted;
        margin-left: 2;
    }

    #button-container {
        height: auto;
        align: center middle;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, config: Config, initial_query: str = "") -> None:
        super().__init__()
        self.config = config
        self.initial_query = initial_query
        self.neo4j_manager = Neo4jManager(config.database)

    def compose(self) -> ComposeResult:
        with Container(id="search-container"):
            yield Label("Search Existing Entities", id="search-title")

            with Horizontal():
                yield Input(
                    value=self.initial_query,
                    placeholder="Search by name...",
                    id="entity-search-input"
                )
                yield Select(
                    options=[("All Types", None)] + [(t.value, t.value) for t in EntityType],
                    value=None,
                    id="type-filter",
                )

            yield ListView(id="results-list")

            with Horizontal(id="button-container"):
                yield Button("Select", variant="primary", id="select-btn")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        self.neo4j_manager.connect()
        if self.initial_query:
            self.perform_search(self.initial_query)
        self.query_one("#entity-search-input").focus()

    def on_unmount(self) -> None:
        self.neo4j_manager.close()

    @on(Input.Changed, "#entity-search-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        if len(event.value) >= 2:
            self.perform_search(event.value)

    @work(thread=True)
    def perform_search(self, query: str) -> None:
        type_filter = self.query_one("#type-filter", Select).value

        try:
            results = self.neo4j_manager.search_entities(
                query=query,
                entity_types=[EntityType(type_filter)] if type_filter else None,
                status=EntityStatus.APPROVED,
                limit=20
            )
            self.app.call_from_thread(self.update_results, results)
        except Exception as e:
            logger.error(f"Search failed: {e}")

    def update_results(self, results: List[Dict[str, Any]]) -> None:
        list_view = self.query_one("#results-list", ListView)
        list_view.clear()
        for res in results:
            list_view.append(EntityResultItem(res))

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, EntityResultItem):
            self.dismiss(event.item.entity_data)

    @on(Button.Pressed, "#select-btn")
    def on_select_pressed(self) -> None:
        list_view = self.query_one("#results-list", ListView)
        if list_view.highlighted_child and isinstance(list_view.highlighted_child, EntityResultItem):
            self.dismiss(list_view.highlighted_child.entity_data)

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

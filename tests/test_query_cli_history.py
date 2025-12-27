from unittest.mock import MagicMock, patch

import pytest

from scripts.query_system import QueryInterface


@pytest.fixture
def mock_dependencies():
    with (
        patch("scripts.query_system.Neo4jManager") as mock_neo4j,
        patch("scripts.query_system.HybridRetriever") as mock_retriever,
        patch("scripts.query_system.QueryParser") as mock_parser,
        patch("scripts.query_system.Config") as mock_config,
        patch("scripts.query_system.console") as mock_console,
    ):

        mock_config.from_yaml.return_value = MagicMock()
        mock_config.from_yaml.return_value.llm.resolve.return_value.model = "test-model"

        yield {
            "neo4j": mock_neo4j,
            "retriever": mock_retriever,
            "parser": mock_parser,
            "config": mock_config,
            "console": mock_console,
        }


def test_history_clear_command(mock_dependencies):
    mock_console = mock_dependencies["console"]

    # Sequence of commands:
    # 1. Run a query -> adds to history
    # 2. Check history -> should show 1 item
    # 3. Clear history -> should empty list
    # 4. Check history -> should show empty message
    # 5. Exit
    mock_console.input.side_effect = ["test query", "history", "clear", "history", "exit"]

    # Initialize interface
    interface = QueryInterface()

    # Mock process_query to verify it's called and to simulate history addition
    with patch.object(interface, "process_query") as mock_process:

        def side_effect(query, top_k=5):
            interface.history.append(
                {
                    "timestamp": "2024-01-01T10:00:00.000",
                    "query": query,
                    "answer": "answer",
                    "result": {},
                    "mode": "standard",
                }
            )

        mock_process.side_effect = side_effect

        interface.run_interactive()

        # Verification
        assert mock_process.call_count == 1

        # IMPORTANT: The history should be empty at the end of the run because we called 'clear'
        assert len(interface.history) == 0

        # Verify 'clear' printed success message
        # We can inspect mock_console.print calls to ensure "Query history cleared." was printed
        assert any(
            "Query history cleared." in str(call) for call in mock_console.print.call_args_list
        )


def test_history_clear_alias_command(mock_dependencies):
    mock_console = mock_dependencies["console"]

    # Test "clear history" alias
    mock_console.input.side_effect = ["test query", "clear history", "exit"]

    interface = QueryInterface()

    with patch.object(interface, "process_query") as mock_process:

        def side_effect(query, top_k=5):
            interface.history.append({"timestamp": "...", "query": query})

        mock_process.side_effect = side_effect

        interface.run_interactive()

        # Should be cleared
        assert len(interface.history) == 0

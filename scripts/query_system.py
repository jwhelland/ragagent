"""Interactive CLI for querying the Graph RAG system (Task 4.7).

This script provides an interactive loop for querying the system,
displaying results with formatting, and generating natural language answers.

Usage:
    uv run python scripts/query_system.py [--verbose] [--export PATH] [--deep]
"""

import argparse
import json
import os
import sys

# Suppress HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime
from typing import Any, Dict, List


from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import HybridRetrievalResult, ResearchResult
from src.retrieval.query_parser import QueryParser
from src.retrieval.research_agent import ResearchAgent
from src.storage.neo4j_manager import Neo4jManager
from src.utils.config import Config

# Initialize rich console
console = Console()


class QueryInterface:
    """Interactive CLI for querying the system."""

    def __init__(self, verbose: bool = False, use_deep_mode: bool = False, auto_mode: bool = False):
        """Initialize query interface.

        Args:
            verbose: Whether to show detailed retrieval info
            use_deep_mode: Whether to use Deep Research Mode
            auto_mode: Whether to automatically determine mode
        """
        self.verbose = verbose
        self.use_deep_mode = use_deep_mode
        self.auto_mode = auto_mode
        self.config = Config.from_yaml()

        # Configure logging
        logger.remove()
        if verbose:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="WARNING")

        console.print("[bold blue]Initializing Graph RAG Query System...[/bold blue]")

        # Initialize managers
        try:
            self.neo4j = Neo4jManager(config=self.config.database)
            self.neo4j.connect(debug=verbose)

            self.query_parser = QueryParser(config=self.config)
            self.hybrid_retriever = HybridRetriever(config=self.config, neo4j_manager=self.neo4j)
            
            # Display model being used
            chat_model = self.config.llm.resolve("chat").model
            console.print(f"[dim]Standard Model: {chat_model}[/dim]")

            # Initialize ResearchAgent if needed (or if in auto mode)
            self.research_agent = None
            if self.use_deep_mode or self.auto_mode:
                research_model = self.config.llm.resolve("research").model
                mode_msg = "Deep Research Mode Enabled" if self.use_deep_mode else "Auto Mode Enabled"
                console.print(f"[bold purple]{mode_msg}[/bold purple] [dim](Model: {research_model})[/dim]")
                self.research_agent = ResearchAgent(
                    config=self.config,
                    retriever=self.hybrid_retriever,
                    response_generator=self.hybrid_retriever.response_generator,
                    query_parser=self.query_parser
                )

            console.print("[bold green]System ready![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Initialization failed: {e}[/bold red]")
            sys.exit(1)

    self.history: List[Dict[str, Any]] = []

    def run_interactive(self):
        """Run the interactive query loop."""
        console.print("\n[bold]Welcome to the Graph RAG Query System[/bold]")
        console.print("Type your query or 'exit' to quit, 'help' for commands.\n")

        while True:
            try:
                query_text = console.input("[bold cyan]Query > [/bold cyan]").strip()

                if not query_text:
                    continue

                if query_text.lower() in ["exit", "quit", "q"]:
                    break

                if query_text.lower() == "help":
                    self._show_help()
                    continue

                if query_text.lower() == "history":
                    self._show_history()
                    continue

                # Process query
                self.process_query(query_text)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit.[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                if self.verbose:
                    import traceback

                    traceback.print_exc()

    def process_query(self, query_text: str, top_k: int = 5):
        """Process a single query.

        Args:
            query_text: User query string
            top_k: Number of results to retrieve
        """
        with console.status("[bold yellow]Analyzing query...[/bold yellow]"):
            parsed_query = self.query_parser.parse(query_text)

        if self.verbose:
            self._display_parsed_query(parsed_query)

        # Determine if we should use deep research
        use_deep = self.use_deep_mode
        if self.auto_mode and not use_deep:
            is_complex = self.query_parser.analyze_complexity(parsed_query)
            if is_complex:
                use_deep = True
                logger.info(f"Auto-mode: Complex query detected, switching to Deep Research. Query: {query_text}")
                console.print("[dim italic]Auto-switching to Deep Research Mode due to query complexity[/dim italic]")
            else:
                logger.info(f"Auto-mode: Simple query detected, using Standard Retrieval. Query: {query_text}")
                if self.verbose:
                    console.print("[dim italic]Auto-mode: Using Standard Retrieval for simple query[/dim italic]")

        if use_deep and self.research_agent:
            # Deep Research Path
            with console.status(
                "[bold purple]Starting Deep Research...[/bold purple]"
            ) as status:
                def update_status(msg: str):
                    status.update(f"[bold purple]{msg}[/bold purple]")
                    
                result = self.research_agent.research(
                    query_text, 
                    status_callback=update_status
                )
            
            self._display_research_result(result)
            
            # Add to history
            self.history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "query": query_text,
                    "answer": result.final_answer.answer,
                    "result": result.to_dict(),
                    "mode": "deep_research"
                }
            )
            
        else:
            # Standard Retrieval Path
            with console.status(
                "[bold yellow]Retrieving context and generating answer...[/bold yellow]"
            ):
                result = self.hybrid_retriever.retrieve(parsed_query, top_k=top_k, generate_answer=True)

            # Display answer
            self._display_answer(result)

            # Display chunks in verbose mode
            if self.verbose:
                self._display_chunks(result)

            # Add to history
            self.history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "query": query_text,
                    "answer": result.answer.answer if result.answer else None,
                    "result": result.to_dict(),
                    "mode": "standard"
                }
            )

    def _display_parsed_query(self, parsed_query: Any):
        """Display details of the parsed query."""
        table = Table(title="Query Analysis", show_header=False)
        table.add_row("Intent", str(parsed_query.intent.value))
        table.add_row("Confidence", f"{parsed_query.intent_confidence:.2f}")
        table.add_row("Requires Graph", str(parsed_query.requires_graph_traversal))

        entities = ", ".join([m.text for m in parsed_query.entity_mentions])
        table.add_row("Entities Found", entities if entities else "None")

        console.print(table)

    def _display_answer(self, result: HybridRetrievalResult):
        """Display the generated answer."""
        if not result.answer:
            console.print("[yellow]No answer was generated.[/yellow]")
            return

        console.print("\n")
        console.print(
            Panel(
                Markdown(result.answer.answer),
                title=f"[bold green]Answer[/bold green] (Confidence: {result.answer.confidence_score:.2f})",
                border_style="green",
                expand=False,
            )
        )

        console.print(
            f"[dim]Strategy: {result.strategy_used.value} | Time: {result.retrieval_time_ms:.0f}ms[/dim]\n"
        )

    def _display_research_result(self, result: ResearchResult):
        """Display the deep research result."""
        
        # Display Steps
        console.print("\n[bold purple]Research Steps:[/bold purple]")
        for step in result.steps:
            missing_info = ", ".join(step.sufficiency_check.missing_information) if step.sufficiency_check else "None"
            is_sufficient = "✅" if step.sufficiency_check and step.sufficiency_check.is_sufficient else "❌"
            
            console.print(f"Step {step.step_number}: Sufficiency: {is_sufficient} | New Chunks: {step.new_chunks_found}")
            if not step.sufficiency_check.is_sufficient:
                console.print(f"  [dim]Missing: {missing_info}[/dim]")
                if step.sub_queries:
                    console.print(f"  [dim]Generated {len(step.sub_queries)} sub-queries[/dim]")

        # Display Final Answer
        console.print("\n")
        console.print(
            Panel(
                Markdown(result.final_answer.answer),
                title=f"[bold purple]Deep Research Answer[/bold purple]",
                border_style="purple",
                expand=False,
            )
        )
        
        console.print(
            f"[dim]Total Chunks: {result.total_chunks} | Time: {result.total_time_ms:.0f}ms[/dim]\n"
        )

    def _display_chunks(self, result: HybridRetrievalResult):
        """Display retrieved chunks."""
        if not result.chunks:
            console.print("[yellow]No chunks retrieved.[/yellow]")
            return

        table = Table(title="Retrieved Chunks", box=None)
        table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
        table.add_column("Score", justify="right", style="magenta")
        table.add_column("ID", style="dim")
        table.add_column("Source", style="green")
        table.add_column("Content Preview", ratio=1)

        for chunk in result.chunks:
            table.add_row(
                str(chunk.rank),
                f"{chunk.final_score:.3f}",
                chunk.chunk_id[:8],
                chunk.source,
                chunk.content[:100].replace("\n", " ") + "...",
            )

        console.print(table)

    def _show_help(self):
        """Show help information."""
        help_text = """
[bold]Commands:[/bold]
  [cyan]history[/cyan] - Show query history
  [cyan]exit[/cyan]    - Quit the system
  [cyan]help[/cyan]    - Show this help message

[bold]Querying Tips:[/bold]
  - Ask natural language questions about the system.
  - For structural questions, use "What are the components of..."
  - For procedural questions, use "How to perform..." or "What steps are needed for..."
  - For dependency questions, use "What does X depend on?"
"""
        console.print(Panel(help_text, title="Help"))

    def _show_history(self):
        """Show query history."""
        if not self.history:
            console.print("[yellow]No history yet.[/yellow]")
            return

        table = Table(title="Query History")
        table.add_column("#", justify="right")
        table.add_column("Time")
        table.add_column("Query")
        table.add_column("Mode")

        for i, entry in enumerate(self.history, 1):
            time_str = entry["timestamp"].split("T")[1].split(".")[0]
            mode = entry.get("mode", "standard")
            table.add_row(str(i), time_str, entry["query"], mode)

        console.print(table)

    def close(self):
        """Close managers."""
        if hasattr(self, "neo4j"):
            self.neo4j.close()


def main():
    parser = argparse.ArgumentParser(description="Graph RAG Query System")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed retrieval info")
    parser.add_argument("--query", "-q", type=str, help="Run a single query and exit")
    parser.add_argument("--export", "-e", type=str, help="Path to export results (JSON)")
    parser.add_argument("--deep", action="store_true", help="Enable Deep Research Mode (Iterative Refinement)")
    parser.add_argument("--auto", action="store_true", help="Automatically enable Deep Research for complex queries")

    args = parser.parse_args()

    interface = QueryInterface(verbose=args.verbose, use_deep_mode=args.deep, auto_mode=args.auto)

    try:
        if args.query:
            interface.process_query(args.query)
            if args.export:
                with open(args.export, "w") as f:
                    json.dump(interface.history, f, indent=2)
                console.print(f"[green]Results exported to {args.export}[/green]")
        else:
            interface.run_interactive()
    finally:
        interface.close()


if __name__ == "__main__":
    main()
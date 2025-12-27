"""Response generation from retrieved context (Task 4.6).

This module implements response generation using LLMs based on retrieved
chunks and graph relationships. It handles:
- Context formatting
- Prompt rendering
- Source citation integration
- LLM interaction
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from loguru import logger

from src.retrieval.models import GeneratedResponse, HybridChunk, HybridRetrievalResult
from src.utils.config import Config
from src.utils.llm_client import create_openai_client


class ResponseGenerator:
    """Generates natural language responses from retrieved context."""

    def __init__(
        self,
        config: Optional[Config] = None,
        prompts_path: str | Path = "config/response_prompts.yaml",
        *,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize response generator.

        Args:
            config: Configuration object
            prompts_path: Path to response prompts YAML
            sleep_fn: Function for sleep during retries
        """
        self.config = config or Config.from_yaml()
        self.llm_config = self.config.llm.resolve("chat")
        self.prompts_path = Path(prompts_path)
        self.prompts = self._load_prompts(self.prompts_path)
        self._sleep = sleep_fn or time.sleep

        logger.info(
            "Initialized ResponseGenerator",
            provider=self.llm_config.provider,
            model=self.llm_config.model,
            prompts=str(self.prompts_path),
        )

    def generate(
        self,
        query_text: str,
        retrieval_result: HybridRetrievalResult,
        prompt_key: str = "default_response",
    ) -> GeneratedResponse:
        """Generate response for a query given retrieval results.

        Args:
            query_text: Original user query
            retrieval_result: Results from hybrid retrieval
            prompt_key: Key for prompt template in YAML

        Returns:
            GeneratedResponse object
        """
        start_time = time.time()

        # Limit context to prevent token overflow
        # We take the top 5 chunks which should be safe for most contexts
        context_chunks_list = retrieval_result.chunks[:5]

        # Format context
        context_chunks = self._format_chunks(context_chunks_list)
        graph_paths = self._format_graph_paths(retrieval_result)

        # Prepare prompt context
        prompt_context = {
            "query_text": query_text,
            "context_chunks": context_chunks,
            "graph_paths": graph_paths,
        }

        # Render prompt
        system, user = self._render_prompt(prompt_key, prompt_context)

        # Call LLM
        try:
            answer = self._call_llm(system=system, user=user)
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            answer = "I'm sorry, I encountered an error while generating a response."

        # Extract used chunk IDs based on numbered citations [1], [2], etc.
        chunks_used = []
        used_indices = []
        for i, chunk in enumerate(retrieval_result.chunks, 1):
            if f"[{i}]" in answer:
                chunks_used.append(chunk.chunk_id)
                used_indices.append(i)

        # Append Sources section if citations were used
        if used_indices:
            sources_list = []
            for idx in sorted(used_indices):
                chunk = retrieval_result.chunks[idx - 1]
                # Try to get document name from metadata or use document_id
                doc_name = (
                    chunk.metadata.get("document_title")
                    or chunk.metadata.get("title")
                    or chunk.metadata.get("filename")
                    or chunk.metadata.get("file_name")
                    or chunk.document_id
                    or "Unknown Document"
                )
                sources_list.append(f"{idx}. {doc_name} (Chunk: {chunk.chunk_id[:8]})")

            answer += "\n\n### Sources\n" + "\n".join(sources_list)

        generation_time = time.time() - start_time

        return GeneratedResponse(
            answer=answer,
            query_id=retrieval_result.query_id,
            chunks_used=chunks_used,
            confidence_score=self._calculate_confidence(retrieval_result, answer),
            metadata={
                "generation_time_s": generation_time,
                "model": self.llm_config.model,
                "provider": self.llm_config.provider,
            },
        )

    def _load_prompts(self, path: Path) -> Dict[str, Any]:
        """Load prompts from YAML file."""
        if not path.exists():
            logger.warning(f"Response prompts file not found: {path}")
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load response prompts: {e}")
            return {}

    def _render_prompt(self, key: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Render prompt template with context."""
        template = self.prompts.get(key, {})
        system = template.get("system", "")
        user_template = template.get("user_template", "{query_text}")

        user = user_template.format(**context)
        return system, user

    def _format_chunks(self, chunks: List[HybridChunk]) -> str:
        """Format chunks for inclusion in prompt."""
        formatted = []
        max_char_per_chunk = 1500

        for i, chunk in enumerate(chunks, 1):
            content = chunk.content
            if len(content) > max_char_per_chunk:
                content = content[:max_char_per_chunk] + "... [truncated]"

            formatted.append(
                f"--- Chunk [{i}] (ID: {chunk.chunk_id}, Doc: {chunk.document_id}) ---\n"
                f"{content}\n"
            )
        return "\n".join(formatted) if formatted else "No relevant context chunks found."

    def _format_graph_paths(self, result: HybridRetrievalResult) -> str:
        """Format graph paths for inclusion in prompt."""
        if not result.graph_paths:
            return "No relevant graph relationships found."

        formatted = []
        for i, path in enumerate(result.graph_paths, 1):
            # Format path as: Entity A -[REL]-> Entity B -[REL]-> Entity C
            path_str = " -> ".join(
                f"{path.nodes[j]['id']} -[{path.relationships[j]['type']}]"
                for j in range(len(path.relationships))
            )
            path_str += f" -> {path.nodes[-1]['id']}"
            formatted.append(f"{i}. {path_str}")

        return "\n".join(formatted)

    def _call_llm(self, system: str, user: str) -> str:
        """Call LLM provider (OpenAI or Anthropic)."""
        attempts = max(1, self.llm_config.retry_attempts)
        last_error = None

        logger.info(
            f"Calling LLM for response generation using {self.llm_config.provider}: {self.llm_config.model}"
        )

        for attempt in range(1, attempts + 1):
            try:
                if self.llm_config.provider == "openai":
                    return self._call_openai(system, user)
                elif self.llm_config.provider == "anthropic":
                    return self._call_anthropic(system, user)
                else:
                    raise ValueError(f"Unsupported LLM provider: {self.llm_config.provider}")
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt} failed: {e}")
                if attempt < attempts:
                    self._sleep(2**attempt)

        if last_error:
            raise last_error
        return "Failed to generate response."

    def _call_openai(self, system: str, user: str) -> str:
        """Call OpenAI API."""
        client = create_openai_client(
            api_key=self.config.openai_api_key,
            base_url=self.llm_config.base_url,
            timeout=self.llm_config.timeout,
        )
        response = client.chat.completions.create(
            model=self.llm_config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, system: str, user: str) -> str:
        """Call Anthropic API."""
        import anthropic

        client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        response = client.messages.create(
            model=self.llm_config.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=4096,
        )
        # Handle the list of content blocks
        return "".join([block.text for block in response.content if hasattr(block, "text")])

    def _calculate_confidence(self, result: HybridRetrievalResult, answer: str) -> float:
        """Calculate confidence in generated answer."""
        if not result.chunks:
            return 0.0

        # Base confidence from mean chunk final score
        mean_score = sum(c.final_score for c in result.chunks) / len(result.chunks)

        # Penalize if answer is very short or contains "don't know" phrases
        if len(answer.split()) < 10:
            mean_score *= 0.5
        if "don't have enough information" in answer.lower() or "cannot answer" in answer.lower():
            mean_score *= 0.2

        return max(0.0, min(1.0, mean_score))

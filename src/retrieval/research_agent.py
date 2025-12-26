"Deep research agent for iterative information gathering and refinement."

import json
import time
from typing import Callable, List, Optional, Set, Tuple

from loguru import logger

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import (
    GeneratedResponse,
    HybridChunk,
    HybridRetrievalResult,
    ResearchResult,
    ResearchStep,
    SufficiencyResult,
)
from src.retrieval.query_parser import QueryParser
from src.retrieval.response_generator import ResponseGenerator
from src.utils.config import Config


class ResearchAgent:
    """Agent that performs iterative research to answer complex queries."""

    def __init__(
        self,
        config: Config,
        retriever: HybridRetriever,
        response_generator: ResponseGenerator,
        query_parser: QueryParser,
    ):
        """Initialize ResearchAgent.

        Args:
            config: System configuration
            retriever: Initialized HybridRetriever
            response_generator: Initialized ResponseGenerator
            query_parser: Initialized QueryParser
        """
        self.config = config
        self.retriever = retriever
        self.response_generator = response_generator
        self.query_parser = query_parser
        
        # Resolve research LLM config if available, otherwise fallback to chat/default
        # Note: We use this purely for the "thinking" steps (sufficiency, sub-queries)
        self.llm_config = self.config.llm.resolve("research")
        
        logger.info(
            "Initialized ResearchAgent",
            model=self.llm_config.model,
            provider=self.llm_config.provider
        )

    def research(
        self,
        query_text: str,
        max_iterations: int = 3,
        initial_top_k: int = 10,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> ResearchResult:
        """Perform deep research on a query.

        Args:
            query_text: User query
            max_iterations: Maximum number of refinement loops
            initial_top_k: Number of chunks to retrieve in first pass
            status_callback: Optional callback for status updates

        Returns:
            ResearchResult containing final answer and steps
        """
        start_time = time.time()
        query_id = f"research_{int(start_time)}"
        
        # Track state
        accumulated_chunks: List[HybridChunk] = []
        seen_chunk_ids: Set[str] = set()
        steps: List[ResearchStep] = []
        
        if status_callback:
            status_callback(f"Starting deep research...")
        
        logger.info(f"Starting deep research for query: {query_text}")

        # --- Step 0: Initial Retrieval ---
        if status_callback:
            status_callback("Executing initial retrieval...")
        logger.info("Executing initial retrieval...")
        parsed_query = self.query_parser.parse(query_text)
        initial_result = self.retriever.retrieve(
            parsed_query, top_k=initial_top_k, generate_answer=False
        )
        
        new_chunks = self._deduplicate_chunks(initial_result.chunks, seen_chunk_ids)
        accumulated_chunks.extend(new_chunks)
        
        # --- Research Loop ---
        for i in range(1, max_iterations + 1):
            if status_callback:
                status_callback(f"Research iteration {i}/{max_iterations}: Assessing sufficiency...")
            logger.info(f"Research iteration {i}/{max_iterations}")
            
            # 1. Assess Sufficiency
            sufficiency = self._assess_sufficiency(query_text, accumulated_chunks)
            
            step_record = ResearchStep(
                step_number=i,
                sub_queries=[],
                sufficiency_check=sufficiency,
                new_chunks_found=len(new_chunks) if i == 1 else 0 
            )
            
            if sufficiency.is_sufficient:
                if status_callback:
                    status_callback("Information sufficient. Synthesizing answer...")
                logger.info("Information deemed sufficient.")
                steps.append(step_record)
                break
            
            if status_callback:
                status_callback(f"Gap detected: {sufficiency.missing_information[0] if sufficiency.missing_information else 'Unknown'}")
                
            # 2. Generate Sub-queries
            sub_queries = self._generate_sub_queries(query_text, sufficiency.missing_information)
            
            if status_callback:
                status_callback(f"Generated {len(sub_queries)} sub-queries...")

            # Update step record with generated queries
            # We create a new record because ResearchStep is frozen
            step_record = ResearchStep(
                step_number=i,
                sub_queries=sub_queries,
                sufficiency_check=sufficiency,
                new_chunks_found=len(new_chunks) if i == 1 else 0
            )
            
            if not sub_queries:
                logger.info("No sub-queries generated, stopping research.")
                steps.append(step_record)
                break
                
            # 3. Execute Sub-queries
            iteration_new_chunks = 0
            for idx, sub_q in enumerate(sub_queries, 1):
                if status_callback:
                    status_callback(f"Executing sub-query {idx}/{len(sub_queries)}: {sub_q}")
                logger.debug(f"Executing sub-query: {sub_q}")
                try:
                    sub_parsed = self.query_parser.parse(sub_q)
                    sub_result = self.retriever.retrieve(
                        sub_parsed, top_k=5, generate_answer=False
                    )
                    
                    unique_chunks = self._deduplicate_chunks(sub_result.chunks, seen_chunk_ids)
                    accumulated_chunks.extend(unique_chunks)
                    iteration_new_chunks += len(unique_chunks)
                    
                except Exception as e:
                    logger.warning(f"Sub-query failed: {sub_q} - {e}")
            
            # Update chunks found count in step record
            step_record = ResearchStep(
                step_number=i,
                sub_queries=sub_queries,
                sufficiency_check=sufficiency,
                new_chunks_found=iteration_new_chunks
            )
            steps.append(step_record)
            
            if iteration_new_chunks == 0:
                if status_callback:
                    status_callback("No new info found. Stopping.")
                logger.info("No new information found in this iteration, stopping.")
                break

        # --- Final Synthesis ---
        if status_callback:
            status_callback("Synthesizing final answer...")
        logger.info("Synthesizing final answer...")
        final_answer = self._synthesize_final_answer(query_text, accumulated_chunks, query_id)
        
        total_time = (time.time() - start_time) * 1000
        
        return ResearchResult(
            query_id=query_id,
            original_query=query_text,
            final_answer=final_answer,
            steps=steps,
            total_chunks=len(accumulated_chunks),
            total_time_ms=total_time,
            accumulated_context=accumulated_chunks
        )

    def _assess_sufficiency(
        self,
        query: str,
        context: List[HybridChunk]
    ) -> SufficiencyResult:
        """Ask LLM if current context is sufficient."""
        # Summarize context for the LLM (to save tokens)
        context_summary = "\n".join(
            [f"- {c.content[:200]}..." for c in context[:20]]
        )  # Limit to 20 chunks preview
        
        # Prepare Prompt
        system_template = self.response_generator.prompts.get("sufficiency_check", {}).get("system", "")
        user_template = self.response_generator.prompts.get("sufficiency_check", {}).get("user_template", "")
        
        user_prompt = user_template.format(
            query_text=query,
            context_summary=context_summary or "No context gathered yet."
        )
        
        # Call LLM
        try:
            response_text = self._call_llm(system_template, user_prompt)
            # Clean json
            json_str = self._extract_json(response_text)
            data = json.loads(json_str)
            
            return SufficiencyResult(
                is_sufficient=data.get("is_sufficient", False),
                missing_information=data.get("missing_information", []),
                reasoning=data.get("reasoning", "No reasoning provided.")
            )
        except Exception as e:
            logger.error(f"Sufficiency check failed: {e}")
            # Fallback to avoid crashing
            return SufficiencyResult(
                is_sufficient=False,
                missing_information=["Unable to verify sufficiency due to error"],
                reasoning=str(e)
            )

    def _generate_sub_queries(self, original_query: str, missing_info: List[str]) -> List[str]:
        """Generate search queries for missing information."""
        if not missing_info:
            return []
            
        system_template = self.response_generator.prompts.get("sub_query_generation", {}).get("system", "")
        user_template = self.response_generator.prompts.get("sub_query_generation", {}).get("user_template", "")
        
        user_prompt = user_template.format(
            query_text=original_query,
            missing_info_list="\n".join(f"- {item}" for item in missing_info)
        )
        
        try:
            response_text = self._call_llm(system_template, user_prompt)
            json_str = self._extract_json(response_text)
            data = json.loads(json_str)
            return data.get("sub_queries", [])
        except Exception as e:
            logger.error(f"Sub-query generation failed: {e}")
            return []

    def _synthesize_final_answer(
        self,
        query: str,
        chunks: List[HybridChunk],
        query_id: str
    ) -> GeneratedResponse:
        """Synthesize final answer from all accumulated chunks."""
        # Use the standard response generator logic but with a specialized prompt
        # We manually construct the context string
        
        # Limit context to prevent token overflow
        # We sort by final_score (descending) to keep the best chunks
        # If final_score is None, we treat it as 0
        sorted_chunks = sorted(
            chunks, 
            key=lambda c: c.final_score if c.final_score is not None else 0.0, 
            reverse=True
        )
        
        # Hard limit to top 5 chunks to avoid context window issues with smaller models
        # (Assuming avg chunk size ~300-500 tokens, 5 chunks is ~1.5k-2.5k tokens)
        max_chunks = 5
        if len(sorted_chunks) > max_chunks:
            logger.warning(
                f"Truncating research context from {len(sorted_chunks)} to {max_chunks} chunks for synthesis."
            )
            context_chunks = sorted_chunks[:max_chunks]
        else:
            context_chunks = sorted_chunks
            
        # Create copies of chunks with truncated content to ensure we don't blow up the prompt
        safe_chunks = []
        max_char_per_chunk = 1000  # Approx 250 tokens
        
        for chunk in context_chunks:
            if len(chunk.content) > max_char_per_chunk:
                # Create a safe copy (pydantic model copy)
                safe_chunk = chunk.model_copy()
                safe_chunk.content = chunk.content[:max_char_per_chunk] + "... [truncated]"
                safe_chunks.append(safe_chunk)
            else:
                safe_chunks.append(chunk)
        
        context_str = self.response_generator._format_chunks(safe_chunks)
        
        # Prepare Prompt
        system_template = self.response_generator.prompts.get("synthesis", {}).get("system", "")
        user_template = self.response_generator.prompts.get("synthesis", {}).get("user_template", "")
        
        user_prompt = user_template.format(
            query_text=query,
            context_chunks=context_str
        )
        
        # Use standard response generator's LLM call to get the final text
        # But we might want to use the research model configuration?
        # For synthesis, the standard chat model might be better/faster, 
        # but let's stick to the research model for consistency if configured,
        # or fall back to the ResponseGenerator's default method which uses 'chat'.
        
        # Actually, let's reuse ResponseGenerator's logic to reuse its robust calling/retry
        # but we need to inject the prompt.
        
        # Create a temporary retrieval result to pass to generate
        # This is a bit of a hack to reuse the method signature
        result = HybridRetrievalResult(
            query_id=query_id,
            query_text=query,
            strategy_used="hybrid_parallel", # Placeholder
            chunks=context_chunks,
            total_results=len(chunks),
            retrieval_time_ms=0
        )
        
        # We call generate with our specific prompt key
        return self.response_generator.generate(
            query_text=query,
            retrieval_result=result,
            prompt_key="synthesis"
        )

    def _deduplicate_chunks(
        self,
        chunks: List[HybridChunk],
        seen_ids: Set[str]
    ) -> List[HybridChunk]:
        """Filter out chunks that have already been seen."""
        new_chunks = []
        for chunk in chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                new_chunks.append(chunk)
        return new_chunks

    def _call_llm(self, system: str, user: str) -> str:
        """Call LLM using the ResearchAgent's specific configuration."""
        # We essentially duplicate the logic from ResponseGenerator but using self.llm_config
        # This allows using a different model for research steps
        
        from src.utils.llm_client import create_openai_client
        import anthropic
        
        attempts = max(1, self.llm_config.retry_attempts)
        
        for attempt in range(1, attempts + 1):
            try:
                if self.llm_config.provider == "openai":
                    client = create_openai_client(
                        api_key=self.config.openai_api_key,
                        base_url=self.llm_config.base_url,
                        timeout=self.llm_config.timeout
                    )
                    response = client.chat.completions.create(
                        model=self.llm_config.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        response_format={"type": "json_object"} # Enforce JSON
                    )
                    return response.choices[0].message.content or "{}"
                    
                elif self.llm_config.provider == "anthropic":
                    client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
                    response = client.messages.create(
                        model=self.llm_config.model,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                        max_tokens=4096,
                    )
                    return "".join([block.text for block in response.content if hasattr(block, "text")])
                    
            except Exception as e:
                logger.warning(f"Research LLM call failed (attempt {attempt}): {e}")
                if attempt == attempts:
                    raise e
                time.sleep(1)
                
        return "{}"

    def _extract_json(self, text: str) -> str:
        """Extract JSON substring from text."""
        try:
            # If already valid JSON
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
            
        # Try to find { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
            
        return "{}"

# Deep Research Mode (Iterative Refinement) Plan

## Objective
Enable a "Deep Research Mode" in the query system that allows for iterative retrieval and refinement. This mode is designed for complex queries where a single retrieval pass is insufficient. The system will self-assess the quality of retrieved information and autonomously generate follow-up queries to fill information gaps before synthesizing a final answer.

## Core Concept
Instead of the standard `Query -> Retrieve -> Generate` linear flow, Deep Research Mode implements a loop:
1.  **Retrieve:** Initial standard retrieval.
2.  **Assess:** LLM evaluates if retrieved context is sufficient to answer the query comprehensively.
3.  **Refine:** If gaps exist, generate targeted sub-queries.
4.  **Iterate:** Execute sub-queries and accumulate context.
5.  **Synthesize:** Generate final comprehensive answer from all accumulated evidence.

## Architecture

### 1. Configuration (`src/utils/config.py` and `config/config.yaml`)
Update `LLMSection` in `src/utils/config.py` to include a `research` sub-section. This allows the Deep Research mode to use a different model/provider than the standard chat or extraction tasks.

```python
class LLMSection(BaseSettings):
    defaults: LLMConfig = Field(default_factory=LLMConfig)
    extraction: PartialLLMConfig = Field(default_factory=PartialLLMConfig)
    rewriting: PartialLLMConfig = Field(default_factory=PartialLLMConfig)
    chat: PartialLLMConfig = Field(default_factory=PartialLLMConfig)
    research: PartialLLMConfig = Field(default_factory=PartialLLMConfig) # New
```

### 2. New Component: `ResearchAgent` (`src/retrieval/research_agent.py`)
A new class that orchestrates the iterative process. It will wrap the existing `HybridRetriever` and `ResponseGenerator`.

**Responsibilities:**
- Managing the research loop (max steps, token limits).
- Maintaining the "Context Window" (accumulated chunks/facts).
- Prompting the LLM for assessment and sub-query generation using task-specific model overrides if configured (e.g., using a larger/smarter model like Claude 3.5 Sonnet for sufficiency checking while using GPT-4o-mini for standard retrieval).

**Key Methods:**
- `research(query_text: str, max_iterations: int = 3) -> ResearchResult`
- `_assess_sufficiency(query, current_context) -> SufficiencyResult`
- `_generate_sub_queries(query, missing_info) -> List[str]`
- `_synthesize_final_answer(query, accumulated_context) -> GeneratedResponse`

### 3. Data Models (`src/retrieval/models.py`)
Update or create models to support the intermediate states.
- `SufficiencyResult`: Boolean `is_sufficient`, List[str] `missing_information`.
- `ResearchResult`: Final answer + trace of steps taken (for transparency/debugging).

### 4. Configuration (`config/response_prompts.yaml`)
Add new prompts:
- `sufficiency_check`: "Given this query and context, is the information sufficient? If not, what is missing?"
- `sub_query_generation`: "Given these missing details, generate 1-3 search queries to find them."
- `synthesis`: "Synthesize a comprehensive answer using these multiple context sources..."

### 4. Integration (`scripts/query_system.py`)
- Add `--deep` or `--research` flag to the CLI.
- When enabled, instantiate and use `ResearchAgent` instead of calling `HybridRetriever.retrieve` directly for the final answer.

## Implementation Plan

### Step 1: Prompt Engineering
- Design and test the `sufficiency_check` and `sub_query_generation` prompts.
- Ensure they return structured or easily parsable outputs (e.g., JSON or strict formatting).

### Step 2: `ResearchAgent` Skeleton
- Create the class structure.
- Inject `HybridRetriever` and `ResponseGenerator` dependencies.
- Implement the main loop logic without the actual LLM calls first (mocking).

### Step 3: Implement Logic
- Connect the LLM calls for assessment and sub-query generation.
- Implement the context aggregation strategy (deduplicating chunks vs. summarizing).
    - *Simplicity Note:* Start by just appending new non-duplicate chunks to the list.

### Step 4: CLI Integration
- Update `scripts/query_system.py` to accept the flag.
- Add visualization for the research steps (e.g., "Thinking... found gap in X... searching for Y...").

### Step 5: Testing & Tuning
- Test with multi-hop questions (e.g., "Compare the power systems of Starship and the ISS").
- Tune the "Sufficiency" threshold to prevent infinite loops or over-researching simple questions.

## Success Criteria
- The system can answer questions that require combining information not found in a single top-k retrieval batch.
- The user can see the "thought process" (e.g., "I need to look up X first").
- Standard queries remain fast; Deep Mode is explicitly opt-in.

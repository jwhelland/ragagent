# Task 2.3 - Prompt Engineering Notes

## Goals
- Improve entity and relationship extraction accuracy for technical docs.
- Enforce JSON-only output with schema alignment to `config/entity_types` and `relationship_types`.
- Make prompts robust to both OpenAI and Anthropic chat APIs.

## Design Decisions
- **Normalization:** Canonical entity names normalized to `lower_snake_case`; abbreviations preserved in `aliases`.
- **Evidence capture:** Optional `source_sentence` (entities) and `evidence` (relationships) to aid curation and confidence review.
- **Schema guardrails:** Added explicit schemas (`output_schemas` in `config/extraction_prompts.yaml`) defining required fields, enums, and confidence bounds (0-1).
- **Domain grounding:** Expanded label descriptions and few-shot examples for subsystem/component/procedure patterns plus table/figure references.
- **Failure mode:** When no valid items exist, prompts instruct models to return empty lists instead of hallucinating placeholders.

## Updated Artifacts
- `config/extraction_prompts.yaml`
  - Strengthened system and user prompts for `entity_extraction` and `relationship_extraction`.
  - Added few-shot examples for procedures, tables, and data flow.
  - Added `output_schemas` documenting entity/relationship JSON structure.

## Testing Plan
- **Offline validation:** Format check via YAML parse and prompt rendering (done locally without LLM calls).
- **LLM dry runs (pending network):**
  - Run `LLMExtractor.extract_entities` and `extract_relationships` on sample EPS and deployment chunks; verify JSON parses without regex fallback.
  - Confirm adherence to enums and that empty outputs return `{"entities":[]}` / `{"relationships":[]}` when no signals exist.
  - Measure latency targets (<2s OpenAI, <5s Anthropic) once network access is available.
- **Regression hooks:** Use stored few-shot examples as fixtures for parsing tests to ensure schema stability across prompt updates.

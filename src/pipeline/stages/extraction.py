import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from src.extraction import EntityMerger, ExtractedEntity, LLMExtractor, SpacyExtractor
from src.extraction.dependency_extractor import DependencyRelationshipExtractor
from src.extraction.pattern_extractor import PatternRelationshipExtractor
from src.extraction.relationship_validator import RelationshipValidator
from src.normalization import EntityDeduplicator, EntityRecord, MergeSuggestion, StringNormalizer
from src.pipeline.base import PipelineContext, PipelineStage
from src.utils.candidate_keys import normalize_candidate_key_fragment
from src.utils.embeddings import EmbeddingGenerator
from src.utils.progress import ExtractionProgress

logger = logging.getLogger(__name__)


class ExtractionStage(PipelineStage):
    """Stage for entity and relationship extraction."""

    def __init__(self, config):
        super().__init__("Extraction")
        self.config = config

        self.embeddings = EmbeddingGenerator(config.database)
        self.string_normalizer = StringNormalizer(config.normalization)

        # Initialize extractors
        self.spacy_extractor: Optional[SpacyExtractor] = None
        try:
            self.spacy_extractor = SpacyExtractor(config.extraction.spacy)
        except Exception as exc:
            logger.warning(f"spaCy extractor initialization failed: {exc}")

        self.llm_extractor: Optional[LLMExtractor] = None
        if config.extraction.enable_llm:
            try:
                self.llm_extractor = LLMExtractor(
                    config.llm.resolve("extraction"),
                    prompts_path=config.extraction.llm_prompt_template,
                )
            except Exception as exc:
                logger.warning(f"LLM extractor initialization failed: {exc}")

        self.pattern_extractor: Optional[PatternRelationshipExtractor] = None
        try:
            self.pattern_extractor = PatternRelationshipExtractor()
        except Exception as exc:
            logger.warning(f"Pattern extractor initialization failed: {exc}")

        self.dependency_extractor: Optional[DependencyRelationshipExtractor] = None
        try:
            nlp = self.spacy_extractor.nlp if self.spacy_extractor else None
            self.dependency_extractor = DependencyRelationshipExtractor(nlp=nlp)
        except Exception as exc:
            logger.warning(f"Dependency extractor initialization failed: {exc}")

        self.relationship_validator = RelationshipValidator(
            config=config.extraction.relationship_validation,
            normalizer=self.string_normalizer,
        )

        self.entity_merger = EntityMerger(
            allowed_types=config.extraction.entity_types,
            normalizer=self.string_normalizer,
        )

        self.entity_deduplicator: Optional[EntityDeduplicator] = None
        if config.normalization.enable_semantic_matching:
            self.entity_deduplicator = EntityDeduplicator(
                config=config.normalization,
                embedder=self.embeddings,
                database_config=config.database,
            )

        self.acronym_resolver = None
        if config.normalization.enable_acronym_resolution:
            from src.normalization.acronym_resolver import AcronymResolver

            self.acronym_resolver = AcronymResolver(
                config=config.normalization,
                normalizer=self.string_normalizer,
            )

    def run(self, context: PipelineContext) -> PipelineContext:
        chunks = context.chunks
        if not chunks:
            return context

        progress = ExtractionProgress(len(chunks))
        if self.spacy_extractor:
            progress.enable_stage("spacy", "spaCy entities")
        if self.config.extraction.enable_llm and self.llm_extractor:
            progress.enable_stage("llm_entities", "LLM entities")
            progress.enable_stage("llm_relationships", "LLM relationships")

        # Step 4: Extract entities with spaCy + LLM (in parallel when possible)
        logger.debug("Extracting entities (spaCy + LLM)")
        spacy_entities_created = 0
        llm_entities_created = 0
        llm_relationships_created = 0

        # Filter chunks for LLM extraction
        llm_extraction_chunks = self._get_extraction_chunks(chunks)
        llm_progress = ExtractionProgress(len(llm_extraction_chunks))

        can_parallelize = (
            self.spacy_extractor is not None
            and self.llm_extractor is not None
            and self.config.extraction.enable_llm
        )

        # We need to access these from context later
        context.spacy_entities_by_chunk = {}
        context.llm_entities_by_chunk = {}

        if can_parallelize:
            with ThreadPoolExecutor(max_workers=2) as executor:
                spacy_future = executor.submit(
                    self._extract_spacy_entities, chunks, context, progress
                )
                llm_future = executor.submit(
                    self._extract_llm_entities, llm_extraction_chunks, context, llm_progress
                )
                spacy_entities_created = spacy_future.result()
                llm_entities_created = llm_future.result()
        else:
            spacy_entities_created = self._extract_spacy_entities(chunks, context, progress)
            if self.config.extraction.enable_llm and self.llm_extractor:
                llm_entities_created = self._extract_llm_entities(
                    llm_extraction_chunks, context, llm_progress
                )

        if self.config.extraction.enable_llm and self.llm_extractor:
            logger.debug("Extracting relationships with LLM")
            llm_relationships_created = self._extract_llm_relationships(
                llm_extraction_chunks, context, llm_progress
            )
            # Propagate entities to child chunks
            self._propagate_llm_entities(chunks, context)

        logger.debug("Extracting rule-based relationships")
        rule_based_relationships_created = self._extract_rule_based_relationships(
            chunks, context, progress
        )

        logger.debug("Merging extracted entities")
        merged_entities_created = self._merge_entities(chunks, context)
        self._enrich_merged_entities_with_acronyms(chunks)

        entities_created = (
            merged_entities_created
            if self.entity_merger
            else spacy_entities_created + llm_entities_created
        )

        logger.debug("Deduplicating merged entities with embeddings")
        dedup_suggestions = self._deduplicate_merged_entities(chunks)

        # Update stats
        context.update_stats("entities_created", entities_created)
        context.update_stats("llm_entities_extracted", llm_entities_created)
        context.update_stats("llm_relationships_extracted", llm_relationships_created)
        context.update_stats("rule_based_relationships_extracted", rule_based_relationships_created)
        context.update_stats("merged_entities_created", merged_entities_created)
        context.update_stats("dedup_merge_suggestions", dedup_suggestions)

        return context

    def _get_extraction_chunks(self, chunks: List[Any]) -> List[Any]:
        # 1. Try Level 3 (Subsections)
        l3_chunks = [c for c in chunks if getattr(c, "level", 0) == 3]
        if l3_chunks:
            return l3_chunks
        # 2. Try Level 2 (Sections)
        l2_chunks = [c for c in chunks if getattr(c, "level", 0) == 2]
        if l2_chunks:
            return l2_chunks
        # 3. Fallback to Level 4 (Paragraphs)
        l4_chunks = [c for c in chunks if getattr(c, "level", 0) == 4]
        if l4_chunks:
            return l4_chunks
        # 4. Final fallback to Level 1
        l1_chunks = [c for c in chunks if getattr(c, "level", 0) == 1]
        return l1_chunks

    def _extract_spacy_entities(
        self,
        chunks: List[Any],
        context: PipelineContext,
        progress: ExtractionProgress | None = None,
    ) -> int:
        if not self.spacy_extractor:
            return 0
        try:
            by_chunk = self.spacy_extractor.extract_from_chunks(chunks)
            context.spacy_entities_by_chunk = by_chunk
        except Exception as exc:
            logger.warning(f"spaCy extraction failed: {exc}")
            return 0

        total = 0
        for chunk in chunks:
            entities = by_chunk.get(getattr(chunk, "chunk_id", None), [])
            if not entities:
                if progress:
                    progress.update("spacy")
                continue

            total += len(entities)
            chunk.metadata.setdefault("spacy_entities", [])
            for ent in entities:
                chunk.metadata["spacy_entities"].append(
                    {
                        "name": ent.name,
                        "type": ent.type,
                        "confidence": ent.confidence,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "sentence": ent.sentence,
                        "context": ent.context,
                        "source": (ent.metadata or {}).get("source", ent.source),
                    }
                )
            if progress:
                progress.update("spacy")

        if progress and not chunks:
            progress.update("spacy")
        return total

    def _process_single_chunk_entities(
        self, chunk: Any
    ) -> Tuple[str | None, List[ExtractedEntity]]:
        metadata = getattr(chunk, "metadata", {}) or {}
        chunk_id = getattr(chunk, "chunk_id", None)
        try:
            if not self.llm_extractor:
                return chunk_id, []
            entities = self.llm_extractor.extract_entities(
                chunk,
                document_context={
                    "document_title": metadata.get("document_title"),
                    "section_title": metadata.get("section_title")
                    or metadata.get("hierarchy_path"),
                    "page_numbers": metadata.get("page_numbers"),
                },
            )
            return chunk_id, entities
        except Exception as exc:
            logger.warning(f"LLM entity extraction failed for chunk {chunk_id}: {exc}")
            return chunk_id, []

    def _extract_llm_entities(
        self,
        chunks: List[Any],
        context: PipelineContext,
        progress: ExtractionProgress | None = None,
    ) -> int:
        if not self.llm_extractor:
            return 0

        total = 0
        llm_entities_by_chunk: DefaultDict[str | None, List[Any]] = defaultdict(list)
        chunk_map = {getattr(c, "chunk_id"): c for c in chunks}

        max_workers = (
            self.config.extraction.max_parallel_calls
            if self.config.extraction.parallel_extraction
            else 1
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_single_chunk_entities, chunk): chunk
                for chunk in chunks
            }
            for future in as_completed(future_to_chunk):
                chunk_id, entities = future.result()
                if entities:
                    total += len(entities)
                    llm_entities_by_chunk[chunk_id].extend(entities)
                    chunk = chunk_map.get(chunk_id)
                    if chunk:
                        metadata = getattr(chunk, "metadata", {}) or {}
                        metadata.setdefault("llm_entities", [])
                        for ent in entities:
                            metadata["llm_entities"].append(
                                {
                                    "name": ent.name,
                                    "type": ent.type,
                                    "description": ent.description,
                                    "aliases": ent.aliases,
                                    "confidence": ent.confidence,
                                    "source": ent.source,
                                    "chunk_id": ent.chunk_id or chunk_id,
                                    "document_id": ent.document_id
                                    or getattr(chunk, "document_id", None),
                                }
                            )
                        chunk.metadata = metadata
                if progress:
                    progress.update("llm_entities")

        context.llm_entities_by_chunk = dict(llm_entities_by_chunk)
        if progress and not chunks:
            progress.update("llm_entities")
        return total

    def _process_single_chunk_relationships(
        self, chunk: Any, context: PipelineContext
    ) -> Tuple[str | None, List[Any]]:
        metadata = getattr(chunk, "metadata", {}) or {}
        chunk_id = getattr(chunk, "chunk_id", None)

        known_entities = []
        known_entities.extend(metadata.get("llm_entities", []))
        known_entities.extend(metadata.get("spacy_entities", []))

        try:
            if not self.llm_extractor:
                return chunk_id, []
            relationships = self.llm_extractor.extract_relationships(
                chunk,
                known_entities=known_entities,
                document_context={
                    "document_title": metadata.get("document_title"),
                    "section_title": metadata.get("section_title")
                    or metadata.get("hierarchy_path"),
                },
            )
            return chunk_id, relationships
        except Exception as exc:
            logger.warning(f"LLM relationship extraction failed for chunk {chunk_id}: {exc}")
            return chunk_id, []

    def _extract_llm_relationships(
        self,
        chunks: List[Any],
        context: PipelineContext,
        progress: ExtractionProgress | None = None,
    ) -> int:
        if not self.llm_extractor:
            return 0
        total = 0
        total_filtered = 0
        chunk_map = {getattr(c, "chunk_id"): c for c in chunks}
        max_workers = (
            self.config.extraction.max_parallel_calls
            if self.config.extraction.parallel_extraction
            else 1
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_single_chunk_relationships, chunk, context): chunk
                for chunk in chunks
            }

            for future in as_completed(future_to_chunk):
                chunk_id, relationships = future.result()
                chunk = chunk_map.get(chunk_id)
                if not chunk:
                    if progress:
                        progress.update("llm_relationships")
                    continue

                metadata = getattr(chunk, "metadata", {}) or {}
                known_entities = []
                known_entities.extend(metadata.get("llm_entities", []))
                known_entities.extend(metadata.get("spacy_entities", []))

                if relationships:
                    rel_dicts = []
                    new_entities_discovered = []

                    for rel in relationships:
                        rel_dict = {
                            "source": rel.source,
                            "source_type": rel.source_type,
                            "type": rel.type,
                            "target": rel.target,
                            "target_type": rel.target_type,
                            "description": rel.description,
                            "confidence": rel.confidence,
                            "bidirectional": rel.bidirectional,
                            "chunk_id": rel.chunk_id or getattr(chunk, "chunk_id", None),
                            "document_id": rel.document_id or getattr(chunk, "document_id", None),
                            "source_extractor": rel.source_extractor,
                        }
                        rel_dicts.append(rel_dict)

                        # Auto-discovery logic
                        for side_name, side_type in [
                            (rel.source, rel.source_type),
                            (rel.target, rel.target_type),
                        ]:
                            if not side_name or not side_type:
                                continue
                            is_known = False
                            side_norm = self.string_normalizer.normalize(side_name).normalized
                            for ent in known_entities:
                                ent_name = str(ent.get("name") or ent.get("canonical_name") or "")
                                ent_norm = self.string_normalizer.normalize(ent_name).normalized
                                if side_norm == ent_norm:
                                    is_known = True
                                    break
                            if not is_known:
                                new_ent = {
                                    "name": side_name,
                                    "type": side_type,
                                    "description": f"Discovered via relationship to {rel.target if side_name == rel.source else rel.source}",
                                    "confidence": rel.confidence * 0.8,
                                    "source": "llm_discovery",
                                    "chunk_id": getattr(chunk, "chunk_id", None),
                                    "document_id": getattr(chunk, "document_id", None),
                                }
                                new_entities_discovered.append(new_ent)
                                known_entities.append(new_ent)
                                metadata.setdefault("llm_entities", []).append(new_ent)

                                chunk_id = getattr(chunk, "chunk_id", None)
                                if chunk_id not in context.llm_entities_by_chunk:
                                    context.llm_entities_by_chunk[chunk_id] = []
                                context.llm_entities_by_chunk[chunk_id].append(
                                    ExtractedEntity(**new_ent)
                                )

                    if new_entities_discovered:
                        logger.info(
                            f"Discovered {len(new_entities_discovered)} new entities via relationships in chunk {chunk_id}"
                        )

                    if self.relationship_validator:
                        valid_rels, rejected_rels = (
                            self.relationship_validator.filter_relationships(
                                rel_dicts, known_entities
                            )
                        )
                        total_filtered += len(rejected_rels)
                        if valid_rels:
                            metadata.setdefault("llm_relationships", []).extend(valid_rels)
                            total += len(valid_rels)
                    else:
                        metadata.setdefault("llm_relationships", []).extend(rel_dicts)
                        total += len(rel_dicts)

                    chunk.metadata = metadata

                if progress:
                    progress.update("llm_relationships")

        context.update_stats("relationships_filtered", total_filtered)
        if progress and not chunks:
            progress.update("llm_relationships")
        return total

    def _propagate_llm_entities(self, chunks: List[Any], context: PipelineContext) -> int:
        propagated_count = 0
        chunk_map = {getattr(c, "chunk_id", None): c for c in chunks}

        for parent in chunks:
            parent_id = getattr(parent, "chunk_id", None)
            child_ids = getattr(parent, "child_chunk_ids", []) or []
            if not child_ids:
                continue

            source_entities = context.llm_entities_by_chunk.get(parent_id, [])
            if not source_entities:
                continue

            for child_id in child_ids:
                child = chunk_map.get(child_id)
                if not child:
                    continue
                content = getattr(child, "content", "").lower()
                child_metadata = getattr(child, "metadata", {}) or {}

                if "llm_entities" not in child_metadata:
                    child_metadata["llm_entities"] = []
                    if child_id not in context.llm_entities_by_chunk:
                        context.llm_entities_by_chunk[child_id] = []

                for ent in source_entities:
                    name_match = str(ent.name).lower() in content
                    alias_match = any(str(alias).lower() in content for alias in ent.aliases)

                    if name_match or alias_match:
                        propagated_entity = ent.model_copy()
                        propagated_entity.chunk_id = child_id
                        context.llm_entities_by_chunk[child_id].append(propagated_entity)

                        child_metadata["llm_entities"].append(
                            {
                                "name": ent.name,
                                "type": ent.type,
                                "description": ent.description,
                                "aliases": ent.aliases,
                                "confidence": ent.confidence,
                                "source": ent.source,
                                "chunk_id": child_id,
                                "document_id": ent.document_id,
                                "propagated_from": parent_id,
                            }
                        )
                        propagated_count += 1
                child.metadata = child_metadata
        return propagated_count

    def _extract_rule_based_relationships(
        self,
        chunks: List[Any],
        context: PipelineContext,
        progress: ExtractionProgress | None = None,
    ) -> int:
        total = 0
        total_filtered = 0

        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            all_rels = []

            if self.pattern_extractor:
                rels = self.pattern_extractor.extract_relationships(chunk)
                if rels:
                    all_rels.extend([r.model_dump() for r in rels])

            if self.dependency_extractor:
                rels = self.dependency_extractor.extract_relationships(chunk)
                if rels:
                    all_rels.extend([r.model_dump() for r in rels])

            if not all_rels:
                continue

            if self.relationship_validator:
                known_entities = []
                known_entities.extend(metadata.get("llm_entities", []))
                known_entities.extend(metadata.get("spacy_entities", []))

                valid_rels, rejected_rels = self.relationship_validator.filter_relationships(
                    all_rels, known_entities
                )
                total_filtered += len(rejected_rels)
                if valid_rels:
                    metadata.setdefault("rule_based_relationships", []).extend(valid_rels)
                    total += len(valid_rels)
            else:
                metadata.setdefault("rule_based_relationships", []).extend(all_rels)
                total += len(all_rels)

            chunk.metadata = metadata

        context.update_stats("relationships_filtered", total_filtered)
        return total

    def _merge_entities(self, chunks: List[Any], context: PipelineContext) -> int:
        if not self.entity_merger:
            return 0
        total = 0
        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", None)
            metadata = getattr(chunk, "metadata", None) or {}

            spacy_entities = context.spacy_entities_by_chunk.get(chunk_id, [])
            llm_entities = context.llm_entities_by_chunk.get(chunk_id, [])

            merged = self.entity_merger.merge(spacy_entities, llm_entities)
            if not merged:
                continue

            metadata.setdefault("merged_entities", [])
            for candidate in merged:
                candidate_key = self._candidate_key(
                    candidate.resolved_type,
                    candidate.canonical_normalized,
                    candidate.canonical_name,
                )
                metadata["merged_entities"].append(
                    {
                        "canonical_name": candidate.canonical_name,
                        "canonical_normalized": candidate.canonical_normalized,
                        "type": candidate.resolved_type,
                        "candidate_key": candidate_key,
                        "confidence": candidate.combined_confidence,
                        "aliases": candidate.aliases,
                        "description": candidate.description,
                        "mention_count": candidate.mention_count,
                        "conflicting_types": candidate.conflicting_types,
                        "provenance": [prov.model_dump() for prov in candidate.provenance],
                    }
                )
            chunk.metadata = metadata
            total += len(merged)
        return total

    def _enrich_merged_entities_with_acronyms(self, chunks: List[Any]) -> None:
        if not (self.acronym_resolver and self.config.normalization.enable_acronym_resolution):
            return
        acronym_re = re.compile(r"\b[A-Z][A-Z0-9&/\-]{1,10}\b")

        for chunk in chunks:
            metadata = getattr(chunk, "metadata", None) or {}
            merged = metadata.get("merged_entities") or []
            if not merged:
                continue
            context = getattr(chunk, "content", "") or ""

            for candidate in merged:
                canonical_name = str(candidate.get("canonical_name") or "").strip()
                alias_list = list(candidate.get("aliases") or [])
                seen: set[str] = set()
                for value in [canonical_name, *alias_list]:
                    if value:
                        seen.add(str(value))

                acronyms: set[str] = set()
                for value in [canonical_name, *alias_list]:
                    for match in acronym_re.finditer(str(value or "")):
                        token = match.group(0)
                        if len(token) > 1:
                            acronyms.add(token)

                if not acronyms:
                    continue

                for acronym in sorted(acronyms):
                    resolution = self.acronym_resolver.resolve(acronym, context=context)
                    if not resolution:
                        continue

                    for alias in resolution.aliases:
                        if alias and alias not in seen:
                            alias_list.append(alias)
                            seen.add(alias)

                    for mention in [canonical_name, *list(candidate.get("aliases") or [])]:
                        if not mention or acronym not in str(mention):
                            continue
                        expanded = str(mention).replace(acronym, resolution.expansion)
                        if expanded and expanded not in seen:
                            alias_list.append(expanded)
                            seen.add(expanded)
                candidate["aliases"] = alias_list
            metadata["merged_entities"] = merged
            chunk.metadata = metadata

    def _deduplicate_merged_entities(self, chunks: List[Any]) -> int:
        if not (self.entity_deduplicator and self.config.normalization.enable_semantic_matching):
            return 0
        aggregated: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            merged = metadata.get("merged_entities") or []
            if not merged:
                continue

            for candidate in merged:
                canonical_name = str(candidate.get("canonical_name") or "").strip()
                if not canonical_name:
                    continue
                type_label = str(candidate.get("type") or "UNKNOWN").upper()
                canonical_normalized = str(
                    candidate.get("canonical_normalized") or canonical_name
                ).strip()
                candidate_key = str(
                    candidate.get("candidate_key")
                    or self._candidate_key(type_label, canonical_normalized, canonical_name)
                )
                candidate["candidate_key"] = candidate_key

                mention_count = int(candidate.get("mention_count") or 1)
                aliases = [alias for alias in candidate.get("aliases") or [] if alias]
                description = str(candidate.get("description") or "").strip()

                record = aggregated.get(candidate_key)
                if record:
                    record["mention_count"] += mention_count
                    for alias in aliases:
                        if alias not in record["aliases"]:
                            record["aliases"].append(alias)
                    if not record["description"] and description:
                        if len(description) > len(record["description"]):
                            record["description"] = description
                else:
                    aggregated[candidate_key] = {
                        "entity_id": candidate_key,
                        "name": canonical_name,
                        "entity_type": type_label,
                        "description": description,
                        "aliases": aliases,
                        "mention_count": max(1, mention_count),
                    }

        if not aggregated:
            return 0
        try:
            records = [EntityRecord(**payload) for payload in aggregated.values()]
            result = self.entity_deduplicator.deduplicate(records)
        except Exception as exc:
            logger.warning(f"Deduplication failed: {exc}")
            return 0

        auto_merge_suggestions = [s for s in result.merge_suggestions if s.auto_merge]
        if auto_merge_suggestions:
            merges_performed = self._perform_auto_merges(chunks, auto_merge_suggestions)
            if merges_performed > 0:
                logger.info(f"Automatically merged {merges_performed} duplicate entities")
                return self._deduplicate_merged_entities(chunks)

        suggestions_by_key: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        for suggestion in result.merge_suggestions:
            payload = suggestion.model_dump()
            suggestions_by_key[suggestion.source_id].append(payload)
            suggestions_by_key[suggestion.target_id].append(payload)

        suggestion_count = len(result.merge_suggestions)
        if suggestion_count == 0:
            return 0

        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            merged = metadata.get("merged_entities") or []
            if not merged:
                continue
            changed = False
            for candidate in merged:
                candidate_key = candidate.get("candidate_key")
                if candidate_key and candidate_key in suggestions_by_key:
                    candidate["dedup_suggestions"] = suggestions_by_key[candidate_key]
                    changed = True
            if changed:
                metadata["merged_entities"] = merged
                chunk.metadata = metadata
        return suggestion_count

    def _perform_auto_merges(self, chunks: List[Any], suggestions: List[MergeSuggestion]) -> int:
        if not suggestions:
            return 0
        parent: Dict[str, str] = {}

        def find(i: str) -> str:
            if i not in parent:
                parent[i] = i
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i: str, j: str) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        involved_keys = set()
        for s in suggestions:
            union(s.source_id, s.target_id)
            involved_keys.add(s.source_id)
            involved_keys.add(s.target_id)

        candidate_stats: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            merged = chunk.metadata.get("merged_entities") or []
            for cand in merged:
                key = cand.get("candidate_key")
                if key and key in involved_keys:
                    if key not in candidate_stats:
                        candidate_stats[key] = {
                            "canonical_name": cand.get("canonical_name"),
                            "mention_count": 0,
                            "type": cand.get("type"),
                            "key": key,
                        }
                    candidate_stats[key]["mention_count"] += int(cand.get("mention_count", 1))

        groups: DefaultDict[str, List[str]] = defaultdict(list)
        for key in involved_keys:
            if key in candidate_stats:
                root = find(key)
                groups[root].append(key)

        replacement_map: Dict[str, str] = {}
        for members in groups.values():
            if len(members) < 2:
                continue
            survivor_key = sorted(
                members,
                key=lambda k: (
                    candidate_stats[k]["mention_count"],
                    len(candidate_stats[k]["canonical_name"] or ""),
                    k,
                ),
                reverse=True,
            )[0]
            for m in members:
                if m != survivor_key:
                    replacement_map[m] = survivor_key

        if not replacement_map:
            return 0

        merged_data_cache: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            merged = chunk.metadata.get("merged_entities") or []
            for cand in merged:
                key = cand.get("candidate_key")
                if not key:
                    continue

                is_survivor = key in merged_data_cache or (
                    key in involved_keys and key not in replacement_map
                )
                is_victim = key in replacement_map
                if not (is_survivor or is_victim):
                    continue

                survivor_key = replacement_map.get(key, key)
                if survivor_key not in merged_data_cache:
                    base_stats = candidate_stats.get(survivor_key)
                    if not base_stats:
                        continue
                    merged_data_cache[survivor_key] = {
                        "canonical_name": base_stats["canonical_name"],
                        "canonical_normalized": "",
                        "type": base_stats["type"],
                        "candidate_key": survivor_key,
                        "confidence": 0.0,
                        "aliases": set(),
                        "description": "",
                        "mention_count": 0,
                        "conflicting_types": set(),
                        "provenance": [],
                    }
                target = merged_data_cache[survivor_key]
                target["confidence"] = max(
                    float(target["confidence"]), float(cand.get("confidence", 0.0))
                )
                target["mention_count"] += int(cand.get("mention_count", 1))
                for alias in cand.get("aliases") or []:
                    if alias:
                        target["aliases"].add(alias)
                if (
                    cand.get("canonical_name")
                    and cand.get("canonical_name") != target["canonical_name"]
                ):
                    target["aliases"].add(cand.get("canonical_name"))
                desc = str(cand.get("description", ""))
                if len(desc) > len(str(target["description"])):
                    target["description"] = desc
                for ct in cand.get("conflicting_types") or []:
                    target["conflicting_types"].add(ct)
                if cand.get("type") and cand.get("type") != target["type"]:
                    target["conflicting_types"].add(cand.get("type"))
                target["provenance"].extend(cand.get("provenance") or [])

        merges_count = len(replacement_map)
        for chunk in chunks:
            merged = chunk.metadata.get("merged_entities") or []
            new_merged = []
            seen_keys_in_chunk = set()
            for cand in merged:
                key = cand.get("candidate_key")
                if not key or (key not in replacement_map and key not in merged_data_cache):
                    new_merged.append(cand)
                    continue
                survivor_key = replacement_map.get(key, key)
                if survivor_key not in seen_keys_in_chunk:
                    if survivor_key in merged_data_cache:
                        data = merged_data_cache[survivor_key]
                        final_cand = data.copy()
                        final_cand["aliases"] = list(data["aliases"])
                        final_cand["conflicting_types"] = list(data["conflicting_types"])
                        new_merged.append(final_cand)
                        seen_keys_in_chunk.add(survivor_key)
            chunk.metadata["merged_entities"] = new_merged
        return merges_count

    def _candidate_key(self, type_label: str, canonical_normalized: str, fallback: str) -> str:
        base = canonical_normalized or fallback
        normalized = normalize_candidate_key_fragment(base, normalizer=self.string_normalizer)
        return f"{type_label}:{normalized}" if normalized else f"{type_label}:{fallback}"

#!/usr/bin/env python3
"""Backfill entity embeddings for existing approved entities.

This script fetches all APPROVED entities from Neo4j, generates embeddings
for them (canonical name + description + aliases), and upserts them to Qdrant.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from tqdm import tqdm

from src.storage.neo4j_manager import Neo4jManager
from src.storage.qdrant_manager import QdrantManager
from src.utils.config import load_config
from src.utils.embeddings import EmbeddingGenerator


def main() -> None:
    """Run the backfill process."""
    config = load_config(Path("config/config.yaml"))

    # Initialize managers
    neo4j = Neo4jManager(config.database)
    neo4j.connect()

    qdrant = QdrantManager(config.database)
    # Ensure collection exists
    qdrant.create_collections(recreate=False)

    embedding_generator = EmbeddingGenerator(config.database)

    try:
        logger.info("Fetching approved entities from Neo4j...")
        # We need a method to get all approved entities.
        # Since Neo4jManager might not have a direct "get_all_approved_entities",
        # we can execute a Cypher query.

        query = """
        MATCH (e:Entity)
        WHERE e.status = 'approved'
        RETURN e
        """

        results = neo4j.execute_cypher(query)
        entities = [record["e"] for record in results]

        logger.info(f"Found {len(entities)} approved entities.")

        if not entities:
            return

        batch_size = 50
        total_upserted = 0

        # Process in batches
        for i in tqdm(range(0, len(entities), batch_size), desc="Embedding entities"):
            batch_entities = entities[i : i + batch_size]

            # Prepare texts and payloads
            texts_to_embed = []
            payloads = []

            for entity in batch_entities:
                # Prepare text for embedding
                text_parts = [entity.get("canonical_name", "")]
                if entity.get("description"):
                    text_parts.append(entity["description"])
                if entity.get("aliases"):
                    aliases = entity["aliases"]
                    if isinstance(aliases, list):
                        text_parts.extend(aliases)

                text_to_embed = ". ".join(text_parts)
                texts_to_embed.append(text_to_embed)

                # Prepare payload (matching what's expected in QdrantManager.upsert_entities)
                # The entity dict from Neo4j needs to be mapped correctly
                # keys: canonical_name, entity_type, description, aliases, id (mapped to entity_id)
                payload = {
                    "entity_id": entity.get("id"),
                    "canonical_name": entity.get("canonical_name"),
                    "entity_type": entity.get("entity_type"),
                    "description": entity.get("description"),
                    "aliases": entity.get("aliases", []),
                    # Add related entity ids if available
                    "related_entity_ids": entity.get("related_entity_ids", []),
                }
                payloads.append(payload)

            # Generate embeddings
            embeddings = embedding_generator.generate(texts_to_embed)

            # Upsert to Qdrant
            # Convert numpy arrays to lists
            vectors = [e.tolist() for e in embeddings]

            count = qdrant.upsert_entities(payloads, vectors)
            total_upserted += count

        logger.success(f"Successfully backfilled {total_upserted} entity embeddings.")

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        raise
    finally:
        neo4j.close()
        qdrant.close()


if __name__ == "__main__":
    main()

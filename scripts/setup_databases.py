#!/usr/bin/env python3
"""Database setup script for initializing Neo4j and Qdrant databases.

This script creates the required schemas, constraints, indexes, and collections
for the Graph RAG system. It can be run multiple times safely (idempotent).

Usage:
    python scripts/setup_databases.py

Environment variables:
    NEO4J_URI - Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USER - Neo4j username (default: neo4j)
    NEO4J_PASSWORD - Neo4j password (default: ragagent2024)
    QDRANT_HOST - Qdrant host (default: localhost)
    QDRANT_PORT - Qdrant port (default: 6333)
    QDRANT_API_KEY - Qdrant API key (optional)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from src.storage.neo4j_manager import Neo4jManager
from src.storage.qdrant_manager import QdrantManager
from src.utils.config import load_config


def setup_neo4j(config) -> bool:
    """Set up Neo4j database with schema and constraints.

    Args:
        config: Application configuration

    Returns:
        True if setup successful, False otherwise
    """
    logger.info("Setting up Neo4j database...")

    try:
        # Initialize Neo4j manager
        neo4j_manager = Neo4jManager(config.database)
        neo4j_manager.connect()

        # Create schema (constraints and indexes)
        neo4j_manager.create_schema()

        # Verify connection and schema
        if neo4j_manager.health_check():
            logger.success("Neo4j setup completed successfully")
            return True
        else:
            logger.error("Neo4j health check failed after setup")
            return False

    except Exception as e:
        logger.error(f"Neo4j setup failed: {e}")
        return False
    finally:
        if "neo4j_manager" in locals():
            neo4j_manager.close()


def setup_qdrant(config) -> bool:
    """Set up Qdrant database with collections.

    Args:
        config: Application configuration

    Returns:
        True if setup successful, False otherwise
    """
    logger.info("Setting up Qdrant database...")

    try:
        # Initialize Qdrant manager
        qdrant_manager = QdrantManager(config.database)

        # Create collections
        qdrant_manager.create_collections(recreate=False)

        # Verify setup
        is_healthy, message = qdrant_manager.health_check()
        if is_healthy:
            logger.success("Qdrant setup completed successfully")
            return True
        else:
            logger.error(f"Qdrant health check failed: {message}")
            return False

    except Exception as e:
        logger.error(f"Qdrant setup failed: {e}")
        return False
    finally:
        if "qdrant_manager" in locals():
            qdrant_manager.close()


def main():
    """Main setup function."""
    logger.info("Starting database setup...")

    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Setup databases
        neo4j_success = setup_neo4j(config)
        qdrant_success = setup_qdrant(config)

        # Report results
        if neo4j_success and qdrant_success:
            logger.success("All databases setup completed successfully!")
            logger.info("You can now run the ingestion pipeline")
            return 0
        else:
            logger.error("Database setup failed. Check logs above for details.")
            return 1

    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        return 1


if __name__ == "__main__":
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # Add file logging
    log_file = Path("logs/setup_databases.log")
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, rotation="10 MB", retention="1 week", level="DEBUG")

    sys.exit(main())

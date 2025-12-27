#!/usr/bin/env python3
"""Database reset script for clearing Neo4j and Qdrant databases.

This script deletes ALL data from both databases to allow for a clean start.
It includes a safety confirmation prompt unless the --force flag is used.

Usage:
    python scripts/reset_databases.py [--force]

Environment variables are loaded from the project configuration.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from rich.prompt import Confirm

from src.storage.neo4j_manager import Neo4jManager
from src.storage.qdrant_manager import QdrantManager
from src.utils.config import load_config


def reset_neo4j(config) -> bool:
    """Clear all data from Neo4j database.

    Args:
        config: Application configuration

    Returns:
        True if reset successful, False otherwise
    """
    logger.info("Resetting Neo4j database...")

    try:
        # Initialize Neo4j manager
        neo4j_manager = Neo4jManager(config.database)
        neo4j_manager.connect()

        # Clear database
        neo4j_manager.clear_database()

        # Re-create schema to ensure constraints exist for next run
        logger.info("Re-applying Neo4j schema constraints...")
        neo4j_manager.create_schema()

        logger.success("Neo4j database cleared and schema reset successfully")
        return True

    except Exception as e:
        logger.error(f"Neo4j reset failed: {e}")
        return False
    finally:
        if "neo4j_manager" in locals():
            neo4j_manager.close()


def reset_qdrant(config) -> bool:
    """Clear all data from Qdrant database.

    Args:
        config: Application configuration

    Returns:
        True if reset successful, False otherwise
    """
    logger.info("Resetting Qdrant database...")

    try:
        # Initialize Qdrant manager
        qdrant_manager = QdrantManager(config.database)

        # Recreate collections (effectively clears them)
        qdrant_manager.create_collections(recreate=True)

        logger.success("Qdrant database cleared and collections recreated successfully")
        return True

    except Exception as e:
        logger.error(f"Qdrant reset failed: {e}")
        return False
    finally:
        if "qdrant_manager" in locals():
            qdrant_manager.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete ALL data from Neo4j and Qdrant databases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt (DANGEROUS).",
    )
    return parser.parse_args()


def main():
    """Main reset function."""
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    args = parse_args()

    if not args.force:
        print("\n⚠️  WARNING: This will DELETE ALL DATA from both Neo4j and Qdrant databases.")
        print("   This action cannot be undone.\n")
        if not Confirm.ask("Are you sure you want to continue?"):
            logger.info("Reset aborted by user.")
            return 0

    logger.info("Starting database reset...")

    try:
        # Load configuration
        config = load_config()

        # Reset databases
        neo4j_success = reset_neo4j(config)
        qdrant_success = reset_qdrant(config)

        # Report results
        if neo4j_success and qdrant_success:
            logger.success("All databases have been successfully reset.")
            logger.info("You can now run 'make ingest' to start fresh.")
            return 0
        else:
            logger.error("Database reset failed. Check logs above for details.")
            return 1

    except Exception as e:
        logger.error(f"Reset failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

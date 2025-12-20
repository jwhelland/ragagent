#!/usr/bin/env python3
"""Document ingestion CLI script.

This script processes PDF documents through the complete ingestion pipeline,
storing chunks and embeddings in Neo4j and Qdrant databases.

Usage:
    python scripts/ingest_documents.py document.pdf
    python scripts/ingest_documents.py --directory data/raw/
    python scripts/ingest_documents.py --config config/custom.yaml file1.pdf file2.pdf

Options:
    --directory, -d: Process all PDFs in directory
    --config, -c: Path to config file (default: config/config.yaml)
    --batch-size: Number of documents to process in parallel (default: 1)
    --verbose, -v: Enable verbose logging
    --dry-run: Show what would be processed without actually doing it
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.utils.config import load_config


def find_pdf_files(paths: list[Path]) -> list[Path]:
    """Find all PDF files from the given paths.

    Args:
        paths: List of file or directory paths

    Returns:
        List of PDF file paths
    """
    pdf_files = []

    for path in paths:
        if path.is_file():
            if path.suffix.lower() == ".pdf":
                pdf_files.append(path)
            else:
                logger.warning(f"Skipping non-PDF file: {path}")
        elif path.is_dir():
            # Find all PDFs in directory recursively
            for pdf_path in path.rglob("*.pdf"):
                pdf_files.append(pdf_path)
        else:
            logger.warning(f"Path does not exist: {path}")

    return sorted(pdf_files)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the Graph RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("paths", nargs="*", type=Path, help="PDF files or directories to process")

    parser.add_argument(
        "--directory", "-d", type=Path, help="Directory containing PDF files to process"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file (default: config/config.yaml)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=1, help="Number of documents to process (default: 1)"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually doing it",
    )

    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Reprocess even if checksum/status indicate the document was already completed",
    )

    args = parser.parse_args()

    # Collect all paths to process
    paths_to_process = args.paths or []
    if args.directory:
        paths_to_process.append(args.directory)

    if not paths_to_process:
        parser.error("No PDF files or directories specified. Use --help for usage.")

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )

    # Add file logging
    log_file = Path("logs/ingest_documents.log")
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, rotation="10 MB", retention="1 week", level="DEBUG")

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Find PDF files
        pdf_files = find_pdf_files(paths_to_process)

        if not pdf_files:
            logger.error("No PDF files found to process")
            return 1

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        if args.dry_run:
            logger.info("Dry run mode - would process:")
            for pdf in pdf_files:
                logger.info(f"  {pdf}")
            return 0

        # Initialize pipeline and components before health check so readiness is accurate
        logger.info("Initializing ingestion pipeline...")
        pipeline = IngestionPipeline(config)
        pipeline.initialize_components()

        # Check component health
        health = pipeline.health_check()
        unhealthy = [comp for comp, healthy in health.items() if not healthy]
        if unhealthy:
            logger.warning(f"Unhealthy components: {', '.join(unhealthy)}")
            logger.warning("Pipeline may not work correctly")

        # Process documents
        start_time = time.time()

        if args.batch_size == 1:
            # Process one by one with detailed logging
            results = []
            for i, pdf_path in enumerate(pdf_files, 1):
                logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_path.name}")
                result = pipeline.process_document(pdf_path, force_reingest=args.force_reingest)
                results.append(result)

                if result.success:
                    logger.success(
                        f"✓ {pdf_path.name}: {result.chunks_created} chunks, "
                        f"{result.processing_time:.2f}s"
                    )
                else:
                    logger.error(f"✗ {pdf_path.name}: {result.error}")
        else:
            # Process in batches
            logger.info(f"Processing in batches of {args.batch_size}")
            results = pipeline.process_batch(pdf_files, force_reingest=args.force_reingest)

        # Calculate statistics
        total_time = time.time() - start_time
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_chunks = sum(r.chunks_created for r in successful)
        avg_time = sum(r.processing_time for r in successful) / len(successful) if successful else 0

        # Print summary
        logger.info("=" * 50)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total documents: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total chunks created: {total_chunks}")
        logger.info(f"Average processing time: {avg_time:.2f}s per document")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Processing rate: {len(successful) / total_time:.2f} docs/sec")

        if failed:
            logger.warning("Failed documents:")
            for result in failed:
                logger.warning(f"  - {result.document_id}: {result.error}")

        # Pipeline statistics
        pipeline_stats = pipeline.get_statistics()
        logger.info("Pipeline statistics:")
        for key, value in pipeline_stats.items():
            logger.info(f"  {key}: {value}")

        # Success/failure exit code
        return 0 if len(successful) > 0 else 1

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1
    finally:
        # Cleanup
        if "pipeline" in locals():
            pipeline.close()


if __name__ == "__main__":
    sys.exit(main())

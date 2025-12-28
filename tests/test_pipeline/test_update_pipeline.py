"""Unit tests for the update pipeline (Task 5.1)."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.update_pipeline import UpdatePipeline
from src.storage.neo4j_manager import Neo4jManager
from src.utils.config import Config


@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.database = MagicMock()
    # Ensure database mock has expected values to avoid type errors
    config.database.qdrant_location = ""
    config.database.qdrant_api_key = "test"
    return config


@pytest.fixture
def mock_neo4j():
    return MagicMock(spec=Neo4jManager)


@pytest.fixture
def update_pipeline(mock_config, mock_neo4j):
    with (
        patch("src.pipeline.update_pipeline.QdrantManager"),
        patch("src.pipeline.update_pipeline.IngestionPipeline"),
    ):
        return UpdatePipeline(mock_config, neo4j_manager=mock_neo4j)


def create_dummy_file(path: Path, content: str = "content") -> str:
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def test_detect_changes_new_file(update_pipeline, mock_neo4j, tmp_path):
    # Setup local file
    file_path = tmp_path / "new_doc.pdf"
    checksum = create_dummy_file(file_path)

    # Mock DB response (empty)
    mock_neo4j.list_entities.return_value = []

    report = update_pipeline.detect_changes(tmp_path, extensions=[".pdf"])

    assert len(report.new_files) == 1
    assert report.new_files[0].path == str(file_path.resolve())
    assert report.new_files[0].checksum == checksum
    assert report.new_files[0].status == "new"
    assert not report.modified_files
    assert not report.deleted_files
    assert not report.unchanged_files


def test_detect_changes_unchanged_file(update_pipeline, mock_neo4j, tmp_path):
    # Setup local file
    file_path = tmp_path / "doc.pdf"
    checksum = create_dummy_file(file_path)

    # Mock DB response (match)
    mock_neo4j.list_entities.return_value = [
        {
            "id": "doc1",
            "file_path": str(file_path.resolve()),
            "checksum": checksum,
            "filename": "doc.pdf",
        }
    ]

    report = update_pipeline.detect_changes(tmp_path, extensions=[".pdf"])

    assert len(report.unchanged_files) == 1
    assert report.unchanged_files[0].document_id == "doc1"
    assert not report.new_files
    assert not report.modified_files


def test_detect_changes_modified_file(update_pipeline, mock_neo4j, tmp_path):
    # Setup local file
    file_path = tmp_path / "doc.pdf"
    new_checksum = create_dummy_file(file_path, "new content")

    # Mock DB response (old checksum)
    mock_neo4j.list_entities.return_value = [
        {
            "id": "doc1",
            "file_path": str(file_path.resolve()),
            "checksum": "old_checksum",
            "filename": "doc.pdf",
        }
    ]

    report = update_pipeline.detect_changes(tmp_path, extensions=[".pdf"])

    assert len(report.modified_files) == 1
    assert report.modified_files[0].checksum == new_checksum
    assert report.modified_files[0].status == "modified"
    assert report.modified_files[0].document_id == "doc1"
    assert not report.new_files


def test_detect_changes_deleted_file(update_pipeline, mock_neo4j, tmp_path):
    # Setup local file (none)

    # Mock DB response (file exists in DB but not on disk)
    missing_path = tmp_path / "missing.pdf"
    mock_neo4j.list_entities.return_value = [
        {
            "id": "doc1",
            "file_path": str(missing_path.resolve()),
            "checksum": "checksum",
            "filename": "missing.pdf",
        }
    ]

    report = update_pipeline.detect_changes(tmp_path, extensions=[".pdf"])

    assert len(report.deleted_files) == 1
    assert report.deleted_files[0].path == str(missing_path.resolve())
    assert report.deleted_files[0].status == "deleted"


def test_detect_changes_moved_file(update_pipeline, mock_neo4j, tmp_path):
    # Setup local file at new location
    new_path = tmp_path / "new_folder" / "doc.pdf"
    new_path.parent.mkdir()
    checksum = create_dummy_file(new_path)

    # Mock DB response (old location, same checksum)
    old_path = tmp_path / "old_folder" / "doc.pdf"
    mock_neo4j.list_entities.return_value = [
        {
            "id": "doc1",
            "file_path": str(old_path.resolve()),
            "checksum": checksum,
            "filename": "doc.pdf",
        }
    ]

    report = update_pipeline.detect_changes(tmp_path, extensions=[".pdf"])

    # Should detect as RENAMED (content-wise same, path changed)
    assert len(report.renamed_files) == 1
    assert report.renamed_files[0].path == str(new_path.resolve())
    assert report.renamed_files[0].document_id == "doc1"
    assert report.renamed_files[0].old_path == str(old_path.resolve())

    # Should NOT detect as deleted (because doc1 was claimed by new path)
    assert len(report.deleted_files) == 0


def test_detect_changes_renamed_file_same_folder(update_pipeline, mock_neo4j, tmp_path):
    # Setup local file renamed
    new_path = tmp_path / "doc_renamed.pdf"
    checksum = create_dummy_file(new_path)

    # Mock DB response (old name, same checksum)
    old_path = tmp_path / "doc.pdf"
    mock_neo4j.list_entities.return_value = [
        {
            "id": "doc1",
            "file_path": str(old_path.resolve()),
            "checksum": checksum,
            "filename": "doc.pdf",
        }
    ]

    report = update_pipeline.detect_changes(tmp_path, extensions=[".pdf"])

    # Now we support content-based rename detection.

    assert len(report.renamed_files) == 1
    assert report.renamed_files[0].path == str(new_path.resolve())
    assert report.renamed_files[0].document_id == "doc1"
    assert report.renamed_files[0].old_path == str(old_path.resolve())

    assert len(report.new_files) == 0
    assert len(report.deleted_files) == 0

"""Unit tests for Neo4j entity node labeling."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.storage.neo4j_manager import Neo4jManager
from src.storage.schemas import Entity, EntityType
from src.utils.config import DatabaseConfig


def test_upsert_entity_adds_entity_label() -> None:
    manager = Neo4jManager(DatabaseConfig())
    manager._connected = True

    session = MagicMock()
    run_result = MagicMock()
    run_result.single.return_value = {"id": "ent-1"}
    session.run.return_value = run_result

    driver = MagicMock()
    driver.session.return_value = session
    manager.driver = driver

    entity = Entity(canonical_name="Power System", entity_type=EntityType.SYSTEM)
    manager.upsert_entity(entity)

    query = session.run.call_args[0][0]
    assert "MERGE (n:Entity:SYSTEM" in query


def test_create_entity_adds_entity_label() -> None:
    manager = Neo4jManager(DatabaseConfig())
    manager._connected = True

    session = MagicMock()
    run_result = MagicMock()
    run_result.single.return_value = {"id": "ent-1"}
    session.run.return_value = run_result

    driver = MagicMock()
    driver.session.return_value = session
    manager.driver = driver

    entity = Entity(canonical_name="Power System", entity_type=EntityType.SYSTEM)
    manager.create_entity(entity)

    query = session.run.call_args[0][0]
    assert "CREATE (n:Entity:SYSTEM" in query

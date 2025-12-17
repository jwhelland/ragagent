"""Tests for configuration loading and override behavior.

Phase 1 requirement: environment variables override YAML values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import load_config, reset_config


@pytest.fixture(autouse=True)
def _reset_global_config() -> None:
    """Ensure config singleton doesn't leak between tests."""
    reset_config()
    yield
    reset_config()


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_yaml_loads_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "database": {
                "neo4j_password": "yaml_pw",
            }
        },
    )

    cfg = load_config(cfg_path)

    assert cfg.database.neo4j_password == "yaml_pw"


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "database": {
                "neo4j_password": "yaml_pw",
            }
        },
    )

    monkeypatch.setenv("NEO4J_PASSWORD", "env_pw")

    cfg = load_config(cfg_path)

    assert cfg.database.neo4j_password == "env_pw"


def test_invalid_yaml_root_type_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(ValueError, match="YAML config root must be a mapping"):
        load_config(cfg_path)

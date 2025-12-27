import os
from unittest.mock import patch

import pytest

from src.utils.llm_client import create_openai_client


@pytest.fixture
def mock_openai():
    with patch("src.utils.llm_client.OpenAI") as mock:
        yield mock


def test_create_client_defaults(mock_openai):
    # Ensure no env vars interfere
    with patch.dict(os.environ, {}, clear=True):
        create_openai_client()
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs.get("api_key") is None
        assert call_kwargs.get("base_url") is None
        assert call_kwargs.get("max_retries") == 2


def test_create_client_explicit_args(mock_openai):
    create_openai_client(
        api_key="sk-explicit", base_url="https://explicit.com", timeout=30.0, max_retries=5
    )

    mock_openai.assert_called_once()
    call_kwargs = mock_openai.call_args.kwargs
    assert call_kwargs["api_key"] == "sk-explicit"
    assert call_kwargs["base_url"] == "https://explicit.com"
    assert call_kwargs["timeout"] == 30.0
    assert call_kwargs["max_retries"] == 5


def test_create_client_env_vars(mock_openai):
    env = {"OPENAI_API_KEY": "sk-env", "OPENAI_BASE_URL": "https://env.com"}
    with patch.dict(os.environ, env):
        create_openai_client()

        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-env"
        assert call_kwargs["base_url"] == "https://env.com"


def test_create_client_args_override_env(mock_openai):
    env = {"OPENAI_API_KEY": "sk-env", "OPENAI_BASE_URL": "https://env.com"}
    with patch.dict(os.environ, env):
        create_openai_client(api_key="sk-override", base_url="https://override.com")

        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-override"
        assert call_kwargs["base_url"] == "https://override.com"

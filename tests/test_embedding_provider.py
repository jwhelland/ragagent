from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils.config import DatabaseConfig
from src.utils.embeddings import EmbeddingGenerator


@pytest.fixture
def mock_openai_client():
    with patch("src.utils.embeddings.create_openai_client") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance

        # Mock embeddings response
        mock_response = MagicMock()
        mock_data_1 = MagicMock()
        mock_data_1.embedding = [0.1, 0.2, 0.3]
        mock_data_2 = MagicMock()
        mock_data_2.embedding = [0.4, 0.5, 0.6]

        mock_response.data = [mock_data_1, mock_data_2]
        client_instance.embeddings.create.return_value = mock_response

        yield mock, client_instance


@pytest.fixture
def mock_fastembed():
    with patch("src.utils.embeddings.TextEmbedding") as mock:
        yield mock


def test_openai_provider_initialization(mock_openai_client):
    mock_factory, mock_instance = mock_openai_client

    config = DatabaseConfig(
        embedding_provider="openai",
        embedding_api_key="sk-test",
        embedding_base_url="https://api.test.com/v1",
        embedding_model="text-embedding-3-small",
    )

    generator = EmbeddingGenerator(config=config)

    assert generator.provider == "openai"
    mock_factory.assert_called_once_with(api_key="sk-test", base_url="https://api.test.com/v1")
    assert generator.client == mock_instance
    assert generator.model is None


def test_openai_provider_generate(mock_openai_client):
    _, mock_instance = mock_openai_client

    config = DatabaseConfig(
        embedding_provider="openai",
        embedding_api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )

    generator = EmbeddingGenerator(config=config)
    texts = ["hello", "world"]

    embeddings = generator.generate(texts)

    assert len(embeddings) == 2
    assert isinstance(embeddings[0], np.ndarray)
    assert np.allclose(embeddings[0], [0.1, 0.2, 0.3])

    # Verify call arguments
    mock_instance.embeddings.create.assert_called_once()
    call_args = mock_instance.embeddings.create.call_args
    assert call_args.kwargs["model"] == "text-embedding-3-small"
    assert call_args.kwargs["input"] == texts


def test_local_provider_initialization(mock_fastembed):
    config = DatabaseConfig(embedding_provider="local", embedding_model="BAAI/bge-small-en-v1.5")

    generator = EmbeddingGenerator(config=config)

    assert generator.provider == "local"
    assert generator.client is None
    mock_fastembed.assert_called_once()


def test_generate_truncation(mock_openai_client):
    _, mock_instance = mock_openai_client

    config = DatabaseConfig(
        embedding_provider="openai",
        embedding_api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )

    generator = EmbeddingGenerator(config=config)

    # Create a very long string
    long_text = "word " * 10000

    generator.generate([long_text])

    # Verify truncation happened before call
    call_args = mock_instance.embeddings.create.call_args
    sent_text = call_args.kwargs["input"][0]

    # Check if it was truncated (original is ~50k chars, truncated should be ~32k chars for 8191 tokens * 4 chars/token)
    assert len(sent_text) < len(long_text)
    assert len(sent_text) <= 8191 * 4 + 100  # Approx limit + ellipsis/margin

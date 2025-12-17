"""Embedding generation module using FastEmbed or Sentence Transformers.

This module provides efficient embedding generation for document chunks and
entities, with support for batching, caching, and text truncation.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastembed import TextEmbedding
from loguru import logger

from src.utils.config import DatabaseConfig


class EmbeddingGenerator:
    """Generate embeddings for text using FastEmbed or Sentence Transformers.

    This class handles embedding generation with batching for efficiency,
    text truncation for long inputs, and optional caching for repeated texts.

    Example:
        >>> generator = EmbeddingGenerator(config)
        >>> embeddings = generator.generate(["text1", "text2"])
        >>> print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    """

    def __init__(
        self,
        config: Optional[DatabaseConfig] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True
    ) -> None:
        """Initialize the embedding generator.

        Args:
            config: Database configuration with embedding settings
            cache_dir: Directory for caching embeddings (optional)
            use_cache: Whether to cache embeddings
        """
        self.config = config or DatabaseConfig()
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        # Initialize cache
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Load embedding model
        try:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.model = TextEmbedding(
                model_name=self.config.embedding_model,
                cache_dir=str(cache_dir) if cache_dir else None
            )
            logger.success(
                f"Loaded {self.config.embedding_model} "
                f"({self.config.embedding_dimension}d embeddings)"
            )
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

        # Model-specific settings
        self.max_seq_length = self._get_max_seq_length()

    def _get_max_seq_length(self) -> int:
        """Get maximum sequence length for the model.

        Returns:
            Maximum sequence length in tokens
        """
        # Default max lengths for common models
        max_lengths = {
            "BAAI/bge-small-en-v1.5": 512,
            "BAAI/bge-base-en-v1.5": 512,
            "BAAI/bge-large-en-v1.5": 512,
            "sentence-transformers/all-MiniLM-L6-v2": 256,
            "sentence-transformers/all-mpnet-base-v2": 384,
        }

        return max_lengths.get(self.config.embedding_model, 512)

    def generate(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (uses config default if None)
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors (numpy arrays)
        """
        if not texts:
            return []

        batch_size = batch_size or self.config.embedding_batch_size

        # Check cache for existing embeddings
        embeddings: List[Optional[np.ndarray]] = []
        texts_to_embed: List[Tuple[int, str]] = []

        for i, text in enumerate(texts):
            if self.use_cache:
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    embeddings.append(self._cache[cache_key])
                    self._cache_hits += 1
                    continue

            embeddings.append(None)
            texts_to_embed.append((i, text))
            self._cache_misses += 1

        # Generate embeddings for uncached texts
        if texts_to_embed:
            # Truncate long texts
            truncated_texts = [
                self._truncate_text(text) for _, text in texts_to_embed
            ]

            # Generate embeddings in batches
            try:
                generated = list(self.model.embed(
                    truncated_texts,
                    batch_size=batch_size
                ))

                # Store in cache and result list
                for (i, original_text), embedding in zip(texts_to_embed, generated):
                    embedding_array = np.array(embedding, dtype=np.float32)

                    if self.use_cache:
                        cache_key = self._get_cache_key(original_text)
                        self._cache[cache_key] = embedding_array

                    embeddings[i] = embedding_array

            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise

        # Convert to list of arrays (remove None values)
        result = [e for e in embeddings if e is not None]

        logger.debug(
            f"Generated {len(result)} embeddings "
            f"(cache hits: {self._cache_hits}, misses: {self._cache_misses})"
        )

        return result

    def generate_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (numpy array)
        """
        embeddings = self.generate([text])
        return embeddings[0] if embeddings else np.zeros(self.config.embedding_dimension)

    def _truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncate text to fit model's maximum sequence length.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens (uses model default if None)

        Returns:
            Truncated text
        """
        max_tokens = max_tokens or self.max_seq_length

        # Simple approximation: ~4 characters per token
        max_chars = max_tokens * 4

        if len(text) <= max_chars:
            return text

        # Truncate and add ellipsis
        truncated = text[:max_chars].rsplit(' ', 1)[0]  # Break at word boundary
        logger.debug(f"Truncated text from {len(text)} to {len(truncated)} chars")

        return truncated + "..."

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text.

        Args:
            text: Text to generate key for

        Returns:
            MD5 hash of text
        """
        return hashlib.md5(text.encode()).hexdigest()

    def batch_generate(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """Generate embeddings in batches (alias for generate).

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        return self.generate(texts, batch_size=batch_size)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Cleared embedding cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            Embedding dimension
        """
        return self.config.embedding_dimension

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize an embedding to unit length.

        Args:
            embedding: Embedding vector

        Returns:
            Normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def normalize_embeddings(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize multiple embeddings to unit length.

        Args:
            embeddings: List of embedding vectors

        Returns:
            List of normalized embedding vectors
        """
        return [self.normalize_embedding(e) for e in embeddings]

    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (between -1 and 1)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(embedding1, embedding2) / (norm1 * norm2)

    def batch_cosine_similarity(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray]
    ) -> List[float]:
        """Calculate cosine similarity between query and multiple embeddings.

        Args:
            query_embedding: Query embedding vector
            embeddings: List of embedding vectors to compare

        Returns:
            List of cosine similarity scores
        """
        return [
            self.cosine_similarity(query_embedding, e)
            for e in embeddings
        ]

    def save_cache(self, cache_path: Path) -> None:
        """Save embedding cache to disk.

        Args:
            cache_path: Path to save cache file
        """
        if not self.use_cache:
            logger.warning("Caching is disabled, nothing to save")
            return

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                keys=list(self._cache.keys()),
                embeddings=[self._cache[k] for k in self._cache.keys()]
            )
            logger.info(f"Saved {len(self._cache)} embeddings to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def load_cache(self, cache_path: Path) -> None:
        """Load embedding cache from disk.

        Args:
            cache_path: Path to cache file
        """
        if not cache_path.exists():
            logger.warning(f"Cache file not found: {cache_path}")
            return

        try:
            data = np.load(cache_path, allow_pickle=True)
            keys = data["keys"]
            embeddings = data["embeddings"]

            self._cache = {
                key: embedding
                for key, embedding in zip(keys, embeddings)
            }

            logger.info(f"Loaded {len(self._cache)} embeddings from cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")

    def __repr__(self) -> str:
        """String representation of the generator."""
        return (
            f"EmbeddingGenerator(model={self.config.embedding_model}, "
            f"dimension={self.config.embedding_dimension}, "
            f"cache_size={len(self._cache)})"
        )


class SentenceTransformerEmbedding:
    """Alternative embedding generator using Sentence Transformers.

    This is a fallback option if FastEmbed has issues or for using
    specific Sentence Transformers models.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ) -> None:
        """Initialize Sentence Transformers embedding generator.

        Args:
            model_name: Name of the Sentence Transformers model
            device: Device to use ('cpu' or 'cuda')
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device=device)
            self.model_name = model_name
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()

            logger.info(
                f"Loaded Sentence Transformers model: {model_name} "
                f"({self.embedding_dimension}d)"
            )
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise

    def generate(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return [np.array(e, dtype=np.float32) for e in embeddings]

    def generate_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.generate([text])[0]

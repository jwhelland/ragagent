from __future__ import annotations

import json
from typing import List, Optional

import urllib.request


class EmbeddingsClient:
    def __init__(self, endpoint: Optional[str]):
        self.endpoint = endpoint or "http://localhost:8080/embed"

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps({"inputs": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        # TEI returns { "embeddings": [ [..], .. ] }
        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]
        # Fallback: some versions return a list of vectors
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data
        raise RuntimeError("Unexpected embeddings response format")

    def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        if not texts:
            return all_embeddings
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeds = self._embed_batch(batch)
            all_embeddings.extend(batch_embeds)
        return all_embeddings

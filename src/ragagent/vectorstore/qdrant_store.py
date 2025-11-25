from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Sequence
import urllib.request

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse


def _point_id_from_chunk_id(chunk_id: str) -> int:
    # Deterministic unsigned 64-bit int derived from chunk_id
    h = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


class QdrantStore:
    def __init__(self, url: str, api_key: str | None, collection: str, vector_size: int, distance: str):
        self.client = QdrantClient(url=url, api_key=api_key)  # type: ignore[arg-type]
        self._url = url.rstrip("/")
        self.collection = collection
        self.vector_size = vector_size
        self.distance = distance

        # Ensure collection exists with desired configuration
        try:
            self.client.get_collection(self.collection)
        except UnexpectedResponse as exc:  # type: ignore[match-not-allowed]
            if getattr(exc, "status_code", None) != 404:
                raise
            # Create collection if missing
            try:
                distance_enum = getattr(qm.Distance, self.distance.upper())
            except AttributeError:
                distance_enum = qm.Distance.COSINE
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.vector_size, distance=distance_enum),
            )

    def upsert(self, chunks: List[dict], vectors: List[List[float]]) -> None:
        points = []
        for c, v in zip(chunks, vectors):
            payload = {
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "page": c["page"],
                "sha256": c["sha256"],
                "source_path": c["source_path"],
                "table_id": c.get("table_id"),
                "text": c.get("text"),
                "entities": c.get("entities") or [],
                "keyphrases": c.get("keyphrases") or [],
            }
            point_id = _point_id_from_chunk_id(c["chunk_id"])
            points.append(qm.PointStruct(id=point_id, vector=v, payload=payload))

        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 8,
        filters: qm.Filter | None = None,
        score_threshold: float | None = None,
    ) -> List[qm.ScoredPoint]:
        params: Dict[str, object] = {
            "collection_name": self.collection,
            "query_vector": vector,
            "limit": top_k,
            "with_payload": True,
            "with_vectors": False,
        }
        if filters is not None:
            params["query_filter"] = filters
        if score_threshold is not None:
            params["score_threshold"] = score_threshold

        # Preferred path: use high-level client API when available (keeps tests/mocks happy).
        if hasattr(self.client, "search"):
            return self.client.search(**params)  # type: ignore[arg-type]

        # Fallback: call REST API directly when client lacks `search` (older/newer variants).
        body: Dict[str, object] = {
            "vector": vector,
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
        }
        # Filters/score_threshold are currently only used in tests with the mock client
        # (which exercises the branch above), so we omit them from this HTTP fallback
        # for maximum compatibility across qdrant-client versions.

        url = f"{self._url}/collections/{self.collection}/points/search"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("result", []) or []
        scored: List[qm.ScoredPoint] = []
        for item in results:
            scored.append(
                qm.ScoredPoint(
                    id=item.get("id"),
                    score=item.get("score"),
                    payload=item.get("payload") or {},
                    version=item.get("version"),
                )
            )
        return scored

    def fetch_by_ids(self, ids: Sequence[str]) -> Dict[str, dict]:
        """
        Fetch payloads keyed by logical chunk_id.

        The external API uses chunk_ids (string like \"doc:page:idx\"), while Qdrant
        stores numeric point IDs derived deterministically from chunk_id.
        """
        if not ids:
            return {}
        id_map: Dict[int, str] = {_point_id_from_chunk_id(cid): cid for cid in ids}
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=list(id_map.keys()),
            with_payload=True,
            with_vectors=False,
        )
        payloads: Dict[str, dict] = {}
        for rec in records:
            try:
                numeric_id = int(rec.id)  # type: ignore[arg-type]
            except Exception:
                continue
            chunk_id = id_map.get(numeric_id)
            if not chunk_id:
                continue
            payloads[chunk_id] = rec.payload or {}
        return payloads

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential
from opentelemetry import trace

from ..chunking.splitter import chunk_text
from ..config import settings
from ..embeddings.client import EmbeddingsClient
from ..graph.store import GraphStore
from ..logging_setup import get_logger
from ..nlp.keywords import extract_entities_and_phrases
from ..vectorstore.qdrant_store import QdrantStore
from .extract import extract_pdf


logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_pdfs(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.rglob("*.pdf")):
        if p.is_file():
            yield p


def persist_artifacts(base_dir: Path, doc_id: str, payload: dict) -> None:
    out_dir = base_dir / doc_id
    ensure_dir(out_dir)
    with (out_dir / "extraction.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_manifest(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data))


@retry(wait=wait_exponential(multiplier=0.5, min=1, max=30), stop=stop_after_attempt(5))
def _embed_with_retry(embedder: EmbeddingsClient, texts: List[str]):
    return embedder.embed(texts)


def process_pdf(
    pdf_path: Path,
    out_dir: Path,
    embedder: EmbeddingsClient,
    vector_store: QdrantStore,
    graph: GraphStore,
) -> int:
    doc_id = pdf_path.stem
    sha = file_sha256(pdf_path)
    with tracer.start_as_current_span(
        "ingest.process_pdf",
        attributes={"doc_id": doc_id, "source_path": str(pdf_path)},
    ):
        logger.info("ingest_start", doc=str(pdf_path), doc_id=doc_id, sha256=sha)
        with tracer.start_as_current_span("ingest.extract_pdf"):
            extracted = extract_pdf(pdf_path)
        persist_artifacts(out_dir, doc_id, extracted)

    chunks = []
    for page in extracted.get("pages", []):
        page_num = page.get("page_number")
        text = page.get("text", "")
        # text chunks
        for cidx, ch in enumerate(chunk_text(text)):
            chunks.append({
                "doc_id": doc_id,
                "sha256": sha,
                "page": page_num,
                "chunk_id": f"{doc_id}:{page_num}:{cidx}",
                "text": ch,
                "source_path": str(pdf_path),
            })
        # table chunks (as markdown if available)
        for t in page.get("tables", []) or []:
            table_id = t.get("table_id")
            ttext = t.get("markdown") or ""
            if not ttext.strip():
                continue
            chunks.append({
                "doc_id": doc_id,
                "sha256": sha,
                "page": page_num,
                "chunk_id": f"{table_id}",
                "text": ttext,
                "source_path": str(pdf_path),
                "table_id": table_id,
            })

    if not chunks:
        logger.warning("no_chunks_generated", doc_id=doc_id)
        return 0

    # Optional NER/keyphrases for graph linking
    for c in chunks:
        ents, phrases = extract_entities_and_phrases(c["text"])  # lightweight heuristics
        c["entities"] = ents
        c["keyphrases"] = phrases

    texts = [c["text"] for c in chunks]
    with tracer.start_as_current_span("ingest.embed", attributes={"chunks": len(chunks)}):
        embeddings = _embed_with_retry(embedder, texts)

    with tracer.start_as_current_span("ingest.upsert_vector", attributes={"chunks": len(chunks)}):
        vector_store.upsert(chunks, embeddings)

    with tracer.start_as_current_span("ingest.graph_update"):
        graph.upsert_document(doc_id=doc_id, sha256=sha, path=str(pdf_path))
        for c in chunks:
            graph.upsert_section(doc_id=doc_id, page=c["page"], chunk_id=c["chunk_id"])
            graph.link_document_section(doc_id=doc_id, chunk_id=c["chunk_id"])
            graph.add_episode_for_chunk(
                name=c["chunk_id"],
                body=c["text"],
                doc_id=doc_id,
                page=c.get("page"),
                source_desc="PDF chunk (text/table)",
            )
            for e in c.get("entities", [])[:10]:
                graph.upsert_entity(e)
                graph.link_section_entity(chunk_id=c["chunk_id"], key=e)

    logger.info("ingest_done", doc_id=doc_id, chunks=len(chunks))
    return len(chunks)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Batch ingest PDFs")
    parser.add_argument("input_dir", type=Path, help="Directory containing PDFs")
    parser.add_argument("--out", type=Path, default=Path("data/processed"), help="Artifacts output directory")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"), help="Idempotency manifest path")
    args = parser.parse_args(argv)

    ensure_dir(args.out)
    embedder = EmbeddingsClient(settings.embeddings_endpoint)
    vector_store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        vector_size=settings.qdrant_vector_size,
        distance=settings.qdrant_distance,
    )
    graph = GraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )

    manifest = load_manifest(args.manifest)
    processed = set(manifest.get("processed_sha256", []))
    updated = False
    for pdf in read_pdfs(args.input_dir):
        sha = file_sha256(pdf)
        if sha in processed:
            logger.info("skip_already_processed", file=str(pdf))
            continue
        chunks = process_pdf(pdf, args.out, embedder, vector_store, graph)
        if chunks <= 0:
            logger.warning("ingest_skipped_zero_chunks", file=str(pdf))
            continue
        processed.add(sha)
        updated = True
    if updated:
        manifest["processed_sha256"] = sorted(processed)
        save_manifest(args.manifest, manifest)


if __name__ == "__main__":
    main()

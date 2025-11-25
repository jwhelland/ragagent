from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

from ..agent import AnswerVerifier, RetrievalAgent
from ..agent.verification import VerificationReport
from ..config import settings
from ..embeddings.client import EmbeddingsClient
from ..graph.store import GraphStore
from ..logging_setup import get_logger
from ..retrieval import ContextAssembler, GraphRetriever, VectorRetriever
from ..vectorstore.qdrant_store import QdrantStore
from .dataset import EvalExample, load_dataset
from .metrics import ExampleMetrics, compute_accuracy, compute_citation_coverage, compute_hallucination_flag, summarize_metrics

try:
    from tabulate import tabulate
except Exception:  # noqa: BLE001
    tabulate = None

log = get_logger(__name__)


def build_agent(enable_llm_verifier: bool = False) -> RetrievalAgent:
    embedder = EmbeddingsClient(settings.embeddings_endpoint)
    qdrant_store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        vector_size=settings.qdrant_vector_size,
        distance=settings.qdrant_distance,
    )
    graph_store = GraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    vector_retriever = VectorRetriever(qdrant_store, embedder, top_k=8)
    graph_retriever = GraphRetriever(graph_store, qdrant_store)
    context_builder = ContextAssembler()

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    verifier = AnswerVerifier(openai_client=client, enable_llm=enable_llm_verifier)
    agent = RetrievalAgent(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        context_assembler=context_builder,
        openai_client=client,
        verifier=verifier,
    )
    setattr(agent, "_graph_store", graph_store)
    return agent


def run_example(agent: RetrievalAgent, example: EvalExample) -> tuple[ExampleMetrics, VerificationReport]:
    start = time.perf_counter()
    response = agent.run(example.question, guidance=example.guidance)
    latency_ms = (time.perf_counter() - start) * 1000

    accuracy = compute_accuracy(response.answer, example.answers)
    coverage = compute_citation_coverage(response.verification)
    hallucination = compute_hallucination_flag(response.verification)

    metrics = ExampleMetrics(
        question=example.question,
        expected_answers=example.answers,
        model_answer=response.answer,
        accuracy=accuracy,
        citation_coverage=coverage,
        hallucination=hallucination,
        latency_ms=latency_ms,
        verification_status=response.verification.overall_status,
    )
    return metrics, response.verification


def format_summary(entries: List[ExampleMetrics]) -> str:
    summary = summarize_metrics(entries)
    rows = [
        ("examples", summary.total_examples),
        ("accuracy", f"{summary.mean_accuracy:.2f}"),
        ("citation coverage", f"{summary.mean_citation_coverage:.2f}"),
        ("hallucination rate", f"{summary.hallucination_rate:.2f}"),
        ("avg latency (ms)", f"{summary.avg_latency_ms:.1f}"),
        ("p95 latency (ms)", f"{summary.p95_latency_ms:.1f}"),
    ]
    if tabulate:
        return tabulate(rows, headers=["metric", "value"])
    return "\n".join(f"{name}: {value}" for name, value in rows)


def write_results(path: Optional[str], metrics: List[ExampleMetrics]) -> None:
    if not path:
        return
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for entry in metrics:
            f.write(json.dumps(entry.__dict__) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG agent on a dataset of questions")
    parser.add_argument("dataset", type=Path, help="Path to JSONL file containing evaluation questions")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of questions to run")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write per-question metrics JSONL")
    parser.add_argument("--enable-llm-verifier", action="store_true", help="Use LLM-backed verification (slower, costlier)")
    parser.add_argument("--verbose", action="store_true", help="Log per-question metrics")
    args = parser.parse_args(argv)

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for evaluation.")

    examples = load_dataset(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    agent = build_agent(enable_llm_verifier=args.enable_llm_verifier)

    metrics: List[ExampleMetrics] = []
    for idx, example in enumerate(examples, start=1):
        log.info("eval_start", idx=idx, question=example.question)
        result, verification = run_example(agent, example)
        metrics.append(result)
        if args.verbose:
            log.info(
                "eval_result",
                idx=idx,
                accuracy=result.accuracy,
                citation=result.citation_coverage,
                hallucination=result.hallucination,
                latency_ms=result.latency_ms,
                verification=verification.overall_status,
            )

    write_results(args.output, metrics)
    graph_store = getattr(agent, "_graph_store", None)
    if graph_store:
        graph_store.close()
    print(format_summary(metrics))


if __name__ == "__main__":
    main()

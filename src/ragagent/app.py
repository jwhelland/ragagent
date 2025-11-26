from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from openai import OpenAI
from pydantic import BaseModel, Field

from .agent.service import RetrievalAgent
from .config import settings
from .embeddings.client import EmbeddingsClient
from .errors import AppError, install_exception_handlers
from .graph.store import GraphStore
from .logging_setup import configure_logging, get_logger
from .retrieval import ContextAssembler, GraphRetriever, VectorRetriever
from .tracing import setup_tracing
from .vectorstore.qdrant_store import QdrantStore


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)
    guidance: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)
    top_k: int = 3


class SourceEntry(BaseModel):
    tag: str
    doc_id: str
    chunk_id: str
    page: Optional[int] = None
    table_id: Optional[str] = None
    origin: str
    preview: str


class SectionVerification(BaseModel):
    tag: str
    status: Literal["supported", "missing", "weak", "llm_unsupported"]
    overlap_ratio: Optional[float] = None
    overlap_tokens: Optional[int] = None
    issues: List[str] = Field(default_factory=list)
    llm_verdict: Optional[str] = None


class VerificationSummary(BaseModel):
    overall_status: Literal["supported", "attention", "failed", "no_citations"]
    cited_tags: List[str] = Field(default_factory=list)
    missing_tags: List[str] = Field(default_factory=list)
    uncited_tags: List[str] = Field(default_factory=list)
    sections: List[SectionVerification] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: Dict[str, Dict[str, Any]]
    sources: List[SourceEntry]
    verification: VerificationSummary


def create_app() -> FastAPI:
    configure_logging(settings.log_level)
    setup_tracing(service_name="rag-agent")

    logger = get_logger(__name__)
    app = FastAPI(title="RAG Agent", version="0.1.0")
    install_exception_handlers(app)

    @app.on_event("startup")
    async def startup() -> None:
        if not settings.openai_api_key:
            logger.warning("openai_api_key_missing")
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
        graph_retriever = GraphRetriever(
            graph_store,
            qdrant_store,
            use_entity_relations=True,
        )
        context_builder = ContextAssembler()
        client = OpenAI(api_key=settings.openai_api_key)
        app.state.graph_store = graph_store
        app.state.agent = RetrievalAgent(
            vector_retriever=vector_retriever,
            graph_retriever=graph_retriever,
            context_assembler=context_builder,
            openai_client=client,
        )
        logger.info("startup_complete")

    @app.on_event("shutdown")
    async def shutdown() -> None:
        graph_store: GraphStore | None = getattr(app.state, "graph_store", None)
        if graph_store:
            graph_store.close()
        logger.info("shutdown_complete")

    @app.get("/health")
    async def health():
        logger.info("health_check", status="ok")
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
        agent: RetrievalAgent | None = getattr(app.state, "agent", None)
        if agent is None:
            raise AppError("agent_not_initialized", status_code=503)
        try:
            result = await run_in_threadpool(
                agent.run,
                payload.question,
                payload.top_k,
                history=[msg.model_dump() for msg in payload.history],
                guidance=payload.guidance,
            )
        except ValueError as exc:
            raise AppError("context_not_found", status_code=404) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat_failure", error=str(exc))
            raise AppError("agent_failure", status_code=500) from exc

        sources = [
            SourceEntry(
                tag=context_chunk.tag,
                doc_id=context_chunk.chunk.doc_id,
                chunk_id=context_chunk.chunk.chunk_id,
                page=context_chunk.chunk.page,
                table_id=context_chunk.chunk.table_id,
                origin=context_chunk.chunk.origin,
                preview=context_chunk.snippet,
            )
            for context_chunk in result.context.chunks
        ]

        verification = VerificationSummary(**result.verification.to_dict())
        return ChatResponse(
            answer=result.answer,
            citations=result.citations,
            sources=sources,
            verification=verification,
        )

    return app


app = create_app()

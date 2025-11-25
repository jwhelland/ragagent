from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, MutableSequence, Optional, Sequence

from openai import OpenAI
from opentelemetry import trace

from ..logging_setup import get_logger
from ..retrieval import ContextAssembler, GraphRetriever, VectorRetriever
from ..retrieval.models import ContextBundle
from .prompts import CHAT_SYSTEM_PROMPT, build_user_prompt
from .verification import AnswerVerifier, VerificationReport


logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


@dataclass
class AgentResponse:
    answer: str
    citations: Dict[str, Dict[str, Any]]
    context: ContextBundle
    verification: VerificationReport
    raw_model_response: Any | None = None


class RetrievalAgent:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        context_assembler: ContextAssembler,
        openai_client: OpenAI,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_graph_results: int = 4,
        verifier: AnswerVerifier | None = None,
    ) -> None:
        self._vector = vector_retriever
        self._graph = graph_retriever
        self._context = context_assembler
        self._client = openai_client
        self._model = model
        self._temperature = temperature
        self._max_graph_results = max_graph_results
        self._verifier = verifier or AnswerVerifier(
            openai_client=openai_client, enable_llm=True
        )

    def run(
        self,
        question: str,
        top_k: int,
        *,
        history: Optional[Sequence[Dict[str, str]]] = None,
        guidance: Optional[str] = None,
    ) -> AgentResponse:
        with tracer.start_as_current_span(
            "agent.run", attributes={"question_length": len(question)}
        ):
            with tracer.start_as_current_span("agent.vector_retrieval"):
                vector_result = self._vector.retrieve(query=question, top_k=top_k)
            if not vector_result.chunks:
                raise ValueError("no_vector_results")

            with tracer.start_as_current_span("agent.graph_expansion"):
                graph_chunks = self._graph.expand(
                    question,
                    vector_result,
                    max_results=self._max_graph_results,
                )

            with tracer.start_as_current_span("agent.context_assembly"):
                context = self._context.build(vector_result.chunks, graph_chunks)

            user_prompt = build_user_prompt(question, context, guidance)

            with tracer.start_as_current_span("agent.llm_completion"):
                completion = self._call_model(history or [], user_prompt)
            answer = (completion.choices[0].message.content or "").strip()

            with tracer.start_as_current_span("agent.verification"):
                verification = self._verifier.verify(answer, context)

            return AgentResponse(
                answer=answer,
                citations=context.citation_map,
                context=context,
                verification=verification,
                raw_model_response=completion,
            )

    def _call_model(self, history: Sequence[Dict[str, str]], user_prompt: str):
        chat_history = self._normalize_history(history)
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            *chat_history,
            {"role": "user", "content": user_prompt},
        ]
        logger.info("openai_request", history_turns=len(chat_history))
        return self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=messages,
        )

    def _normalize_history(
        self, history: Sequence[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        allowed_roles = {"user", "assistant"}
        normalized: List[Dict[str, str]] = []
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if not role or role not in allowed_roles or not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

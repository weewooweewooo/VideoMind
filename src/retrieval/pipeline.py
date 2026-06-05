"""End-to-end VideoMind retrieval augmented generation pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import string
from pathlib import Path
from typing import Any, Iterator

import redis

try:
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from src.retrieval.embedder import CLIPEmbedder
from src.retrieval.store import VideoMindStore


SYSTEM_PROMPT = Path("prompts/system.txt").read_text()
logger = logging.getLogger(__name__)


class VideoMindPipeline:
    """Retrieve lecture context from Redis Stack and answer with local Ollama."""

    def __init__(
        self,
        store: VideoMindStore | None = None,
        checkpoint: str | Path | None = None,
        device: str = "auto",
        ollama_model: str = "llama3.2:3b",
        top_k: int = int(os.environ.get("RETRIEVAL_TOP_K", "5")),
    ) -> None:
        """Initialize retrieval store and local Ollama LLM."""
        self.store = store or VideoMindStore()
        self.embedder = CLIPEmbedder(checkpoint=checkpoint, device=device)
        self.llm = OllamaLLM(model=ollama_model)
        self.streaming_llm = OllamaLLM(model=ollama_model, streaming=True)
        self.top_k = top_k
        self.conversation_history: list[dict[str, str]] = []
        self.cache = redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379")
        )
        self.cache_ttl = 3600

    def _format_context(self, sources: list[dict[str, Any]]) -> str:
        """Format retrieved sources for the LLM prompt."""
        return "\n\n".join(
            f"[{source['start']:.2f}s - {source['end']:.2f}s] "
            f"video={source['video']} score={source['score']:.4f}\n"
            f"{source['text']}"
            for source in sources
        )

    def _format_history(self) -> str:
        """Format the last six conversation messages for the prompt."""
        recent_messages = self.conversation_history[-6:]
        if not recent_messages:
            return "No previous conversation."
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in recent_messages
        )

    def _build_retrieval_query(self, question: str) -> str:
        """Build a retrieval query that includes recent conversational context."""
        recent_messages = self.conversation_history[-6:]
        if not recent_messages:
            return question
        history_text = "\n".join(message["content"] for message in recent_messages)
        return f"{history_text}\n{question}"

    def _build_prompt(self, question: str, sources: list[dict[str, Any]]) -> str:
        """Build a context-aware prompt for the local LLM."""
        human_prompt = f"""Conversation history:
{self._format_history()}

Retrieved video context:
{self._format_context(sources)}

Question: {question}
Answer:"""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=human_prompt),
            ]
        )
        return prompt.format_prompt().to_string()

    def _get_cache_key(self, question: str, video_name: str | None = None) -> str:
        """Build a stable Redis cache key for a normalized question."""
        normalized = question.lower().strip()
        normalized = normalized.translate(str.maketrans("", "", string.punctuation))
        video_scope = (video_name or "all").lower().strip()
        cache_input = f"{video_scope}:{normalized}"
        return f"videomind:cache:{hashlib.md5(cache_input.encode()).hexdigest()}"

    def _get_cached(
        self, question: str, video_name: str | None = None
    ) -> dict[str, Any] | None:
        """Return a cached query response if Redis has one."""
        try:
            key = self._get_cache_key(question, video_name)
            cached = self.cache.get(key)
            if cached:
                logger.debug("Query cache hit")
                return json.loads(cached)
            return None
        except Exception:
            return None

    def _set_cache(
        self,
        question: str,
        response: dict[str, Any],
        video_name: str | None = None,
    ) -> None:
        """Store a query response in Redis without interrupting normal queries."""
        try:
            key = self._get_cache_key(question, video_name)
            self.cache.setex(key, self.cache_ttl, json.dumps(response))
        except Exception:
            pass

    def clear_cache(self) -> None:
        """Delete cached query responses from Redis."""
        try:
            keys = self.cache.keys("videomind:cache:*")
            if keys:
                self.cache.delete(*keys)
        except Exception:
            pass

    def clear_history(self) -> None:
        """Clear the stored conversation history."""
        self.conversation_history = []

    def get_history(self) -> list[dict[str, str]]:
        """Return a shallow copy of the stored conversation history."""
        return self.conversation_history.copy()

    def _format_sources_response(
        self, sources: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Format retrieved sources for API responses."""
        return [
            {
                "video": source["video"],
                "start": source["start"],
                "end": source["end"],
                "text": source["text"],
                "score": source["score"],
            }
            for source in sources
        ]

    def _retrieve_context(
        self, question: str, video_name: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve source context for a question using recent conversation history."""
        retrieval_query = self._build_retrieval_query(question)
        text_embedding = self.embedder.query_embedding(retrieval_query)
        return self.store.query(
            text_embedding,
            top_k=self.top_k,
            video_name=video_name,
        )

    def _record_answer(self, question: str, answer: str) -> None:
        """Append a completed user/assistant turn to conversation history."""
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

    def _insufficient_context_response(self) -> dict[str, Any]:
        """Return the standard insufficient-context response."""
        return {
            "answer": "This video does not cover that topic",
            "sources": [],
        }

    def query(
        self, question: str, video_name: str | None = None
    ) -> dict[str, Any]:
        """Answer a question using retrieved lecture context from all indexed videos."""
        if not question.strip():
            raise ValueError("question must not be empty")

        cached = self._get_cached(question, video_name)
        if cached:
            return cached

        sources = self._retrieve_context(question, video_name)
        if not sources:
            response = self._insufficient_context_response()
            self._record_answer(question, response["answer"])
            self._set_cache(question, response, video_name)
            return response

        prompt = self._build_prompt(question, sources)
        answer = self.llm.invoke(prompt)
        answer_text = str(answer).strip()
        self._record_answer(question, answer_text)
        response = {
            "answer": answer_text,
            "sources": self._format_sources_response(sources),
        }
        self._set_cache(question, response, video_name)
        return response

    def query_stream(
        self,
        question: str,
        session_id: str | None = None,
        video_name: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream an answer token by token, then yield retrieved sources."""
        _ = session_id
        if not question.strip():
            raise ValueError("question must not be empty")

        cached = self._get_cached(question, video_name)
        if cached:
            answer = str(cached.get("answer", ""))
            if answer:
                yield {"token": answer}
            yield {"sources": cached.get("sources", [])}
            return

        sources = self._retrieve_context(question, video_name)
        if not sources:
            response = self._insufficient_context_response()
            self._record_answer(question, response["answer"])
            self._set_cache(question, response, video_name)
            yield {"token": response["answer"]}
            yield {"sources": []}
            return

        prompt = self._build_prompt(question, sources)
        answer_parts: list[str] = []
        for chunk in self.streaming_llm.stream(prompt):
            token = str(getattr(chunk, "content", chunk))
            if not token:
                continue
            answer_parts.append(token)
            yield {"token": token}

        answer_text = "".join(answer_parts).strip()
        self._record_answer(question, answer_text)
        response = {
            "answer": answer_text,
            "sources": self._format_sources_response(sources),
        }
        self._set_cache(question, response, video_name)
        yield {"sources": response["sources"]}

"""End-to-end VideoMind retrieval augmented generation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from src.retrieval.embedder import CLIPEmbedder
from src.retrieval.store import VideoMindStore


SYSTEM_PROMPT = """You are VideoMind, an AI assistant that answers questions about lecture videos.
Answer based only on the retrieved video context and conversation history provided.
Always cite timestamps when referencing specific moments.
If the context is insufficient, say so clearly.
"""


class VideoMindPipeline:
    """Retrieve lecture context from Redis Stack and answer with local Ollama."""

    def __init__(
        self,
        store: VideoMindStore | None = None,
        checkpoint: str | Path | None = None,
        device: str = "auto",
        ollama_model: str = "llama3.2:3b",
        top_k: int = 5,
    ) -> None:
        """Initialize retrieval store and local Ollama LLM."""
        self.store = store or VideoMindStore()
        self.embedder = CLIPEmbedder(checkpoint=checkpoint, device=device)
        self.llm = OllamaLLM(model=ollama_model)
        self.top_k = top_k
        self.conversation_history: list[dict[str, str]] = []

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

    def clear_history(self) -> None:
        """Clear the stored conversation history."""
        self.conversation_history = []

    def get_history(self) -> list[dict[str, str]]:
        """Return a shallow copy of the stored conversation history."""
        return self.conversation_history.copy()

    def query(self, question: str) -> dict[str, Any]:
        """Answer a question using retrieved lecture context from all indexed videos."""
        if not question.strip():
            raise ValueError("question must not be empty")

        retrieval_query = self._build_retrieval_query(question)
        text_embedding = self.embedder.query_embedding(retrieval_query)
        sources = self.store.query(text_embedding, top_k=self.top_k)
        if not sources:
            answer = "The retrieved video context is insufficient to answer the question."
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})
            return {
                "answer": answer,
                "sources": [],
            }

        prompt = self._build_prompt(question, sources)
        answer = self.llm.invoke(prompt)
        answer_text = str(answer).strip()
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append(
            {"role": "assistant", "content": answer_text}
        )
        return {
            "answer": answer_text,
            "sources": [
                {
                    "video": source["video"],
                    "start": source["start"],
                    "end": source["end"],
                    "text": source["text"],
                    "score": source["score"],
                }
                for source in sources
            ],
        }

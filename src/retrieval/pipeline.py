"""End-to-end VideoMind retrieval augmented generation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_ollama import OllamaLLM

from src.retrieval.store import VideoMindStore


SYSTEM_PROMPT = """You are VideoMind, a local lecture video assistant.
Answer only from the retrieved lecture context below.
If the context does not contain the answer, say that the retrieved video context is insufficient.
Cite timestamps in seconds whenever you use a source.
"""


class VideoMindPipeline:
    """Retrieve lecture context from ChromaDB and answer with local Ollama."""

    def __init__(
        self,
        store: VideoMindStore | None = None,
        checkpoint: str | Path | None = None,
        device: str = "auto",
        ollama_model: str = "llama3.2:3b",
        top_k: int = 5,
    ) -> None:
        """Initialize retrieval store and local Ollama LLM."""
        self.store = store or VideoMindStore(checkpoint=checkpoint, device=device)
        self.llm = OllamaLLM(model=ollama_model)
        self.top_k = top_k

    def _format_context(self, sources: list[dict[str, Any]]) -> str:
        """Format retrieved sources for the LLM prompt."""
        return "\n\n".join(
            f"[{source['timestamp']:.2f}s] "
            f"video={source['video']} score={source['score']:.4f}\n"
            f"{source['text']}"
            for source in sources
        )

    def query(self, question: str, video_name: str | None = None) -> dict[str, Any]:
        """Answer a question using retrieved lecture context."""
        if not question.strip():
            raise ValueError("question must not be empty")

        sources = self.store.query(question, top_k=self.top_k, video_name=video_name)
        if not sources:
            return {
                "answer": "The retrieved video context is insufficient to answer the question.",
                "sources": [],
            }

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Retrieved context:\n{self._format_context(sources)}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        answer = self.llm.invoke(prompt)
        return {
            "answer": str(answer).strip(),
            "sources": [
                {
                    "timestamp": source["timestamp"],
                    "text": source["text"],
                    "frame_path": source["frame_path"],
                    "score": source["score"],
                    "video": source["video"],
                }
                for source in sources
            ],
        }

"""Regression tests for local LLM availability validation."""

from __future__ import annotations

import json
import unittest
from typing import Any

from src.answering import VideoAnswerer


class _FakeResponse:
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "object": "list",
                "data": [
                    {
                        "id": "gemma4:e4b",
                        "object": "model",
                        "created": 1787811971,
                        "owned_by": "library",
                    }
                ],
            }
        ).encode("utf-8")


class AvailabilityTests(unittest.TestCase):
    def test_accepts_ollama_openai_compatible_model_list(self) -> None:
        def opener(request: Any, *, timeout: float) -> _FakeResponse:
            self.assertEqual(request.full_url, "http://localhost:11434/v1/models")
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(timeout, 120.0)
            return _FakeResponse()

        VideoAnswerer(opener=opener).check_availability()


if __name__ == "__main__":
    unittest.main()

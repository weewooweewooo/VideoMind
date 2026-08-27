"""Fixed configuration for the current VideoMind application."""

import os

WHISPER_MODEL = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_BEAM_SIZE = 5

VIDEOMIND_LLM_BASE_URL = os.environ.get(
    "VIDEOMIND_LLM_BASE_URL",
    "http://localhost:11434/v1",
).strip().rstrip("/")
VIDEOMIND_LLM_MODEL = os.environ.get(
    "VIDEOMIND_LLM_MODEL",
    "qwen3:4b-instruct-2507-q4_K_M",
).strip()

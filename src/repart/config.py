"""Environment and API clients."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

BASE_URL = os.environ.get("MODEL_API_BASE_URL", "https://api.meta.ai/v1")
SPARK_MODEL = os.environ.get("REPART_SPARK_MODEL", "muse-spark-1.3")
WONDER_MODEL = os.environ.get("REPART_WONDER_MODEL", "muse-wonder-1.0")  # used when a fit stalls
SAM_MODEL = "sam-3.1"
VOICE_MODEL = "muse-voice-transcribe-1.0"
VOICE_FILE_URL = "https://api.meta.ai/v1/asr/transcribe"
GLIMMER_URLS = [
    u
    for u in (
        os.environ.get("REPART_GLIMMER_URL"),
        "http://localhost:11434/v1",  # Ollama
        "http://localhost:1234/v1",  # LM Studio
        "http://localhost:8080/v1",  # llama.cpp server
    )
    if u
]

RUNS_DIR = ROOT / "runs"
CACHE_DIR = RUNS_DIR / "cache"


def api_key() -> str:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise SystemExit(
            "MODEL_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def client() -> OpenAI:
    # Agent rounds can run minutes (Wonder, long reasoning); retry dropped connections.
    return OpenAI(base_url=BASE_URL, api_key=api_key(), timeout=600, max_retries=4)


def async_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE_URL, api_key=api_key())

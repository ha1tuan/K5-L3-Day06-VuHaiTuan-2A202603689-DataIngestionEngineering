from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_MODEL = "gemini-embedding-2"
DEFAULT_DIMENSIONS = 768
MAX_BATCH_SIZE = 100
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class GeminiEmbedder:
    """Gemini embedding adapter. Raises instead of silently returning mock vectors."""

    def __init__(self, model_name: str | None = None, dimensions: int | None = None,
                 api_key: str | None = None, batch_size: int = MAX_BATCH_SIZE,
                 max_retries: int = 5) -> None:
        load_dotenv(ENV_PATH)
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(f"GEMINI_API_KEY is missing; set it in {ENV_PATH}")

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
        self.dimensions = int(dimensions or os.getenv("EMBEDDING_DIMENSIONS") or DEFAULT_DIMENSIONS)
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}]")
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._client = genai.Client(api_key=api_key)

    # ---- public contract -------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._validate_texts(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed(batch, task_type="RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        self._validate_texts([question])
        return self._embed([question], task_type="RETRIEVAL_QUERY")[0]

    # ---- internals -------------------------------------------------------
    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dimensions,
        )
        response = self._call_with_retry(texts, config)
        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            raise RuntimeError(f"requested {len(texts)} embeddings, got {len(embeddings)}")
        return [self._normalize(e.values or []) for e in embeddings]

    def _call_with_retry(self, texts: list[str], config: types.EmbedContentConfig):
        for attempt in range(self.max_retries + 1):
            try:
                contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
                return self._client.models.embed_content(
                    model=self.model_name, contents=contents, config=config,
                )
            except errors.APIError as exc:
                if exc.code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise AssertionError("unreachable")

    def _normalize(self, values: list[float]) -> list[float]:
        if len(values) != self.dimensions:
            raise RuntimeError(f"expected {self.dimensions} dimensions, got {len(values)}")
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("embedding contains NaN or infinity")
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            raise RuntimeError("embedding is a zero vector")
        return [v / norm for v in values]

    @staticmethod
    def _validate_texts(texts: list[str]) -> None:
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"text at index {i} is empty; filter empty chunks before embedding")

MyEmbedder = GeminiEmbedder


def make_embedder() -> GeminiEmbedder:
    return GeminiEmbedder()

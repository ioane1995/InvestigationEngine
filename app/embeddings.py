"""Embedding providers behind one interface.

* HashingEmbedder   - OUR OWN offline embedder: signed feature hashing of word + bigram + char-trigram
                      features into a dense, L2-normalised vector. No model download, deterministic;
                      used for tests, CI, offline evaluation and as a last-resort fallback.
                      It captures lexical/sub-word similarity, NOT true semantics.
* FastEmbedEmbedder - real neural embeddings via ONNX (default BAAI/bge-small-en-v1.5, 384-d), CPU only.
* OpenAICompatEmbedder - any /v1/embeddings compatible HTTP API.
* HFInferenceEmbedder  - Hugging Face Inference Providers feature-extraction route (raw list-of-vectors response).
"""
from __future__ import annotations

import logging
import re
import time
import zlib
from collections import OrderedDict
from typing import Protocol

import httpx
import numpy as np

from app.config import Settings
from app.text import tokenize

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class EmbeddingError(RuntimeError):
    pass


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        feats: dict[str, float] = {}
        for t in toks:
            feats[t] = feats.get(t, 0.0) + 1.0
        for a, b in zip(toks, toks[1:]):
            k = f"{a}_{b}"
            feats[k] = feats.get(k, 0.0) + 0.5
        for t in set(toks):
            if len(t) >= 5:
                p = f"^{t}$"
                for i in range(len(p) - 2):  # char trigrams give sub-word robustness
                    k = "#" + p[i : i + 3]
                    feats[k] = feats.get(k, 0.0) + 0.15
        for f, w in feats.items():
            h = zlib.crc32(f.encode("utf-8"))
            sign = 1.0 if (h >> 31) & 1 else -1.0
            v[h % self.dim] += sign * (1.0 + np.log(w) if w >= 1 else w)
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vec(text)[None, :])[0]


class FastEmbedEmbedder:
    def __init__(self, model: str, dim: int):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover
            raise EmbeddingError("fastembed is not installed: pip install fastembed") from e
        self._m = TextEmbedding(model_name=model)
        self.dim, self.name = dim, f"fastembed:{model}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2(np.asarray(list(self._m.embed(texts, batch_size=64)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(np.asarray(list(self._m.query_embed([text])), dtype=np.float32))[0]


class OpenAICompatEmbedder:
    """Any ``POST {base_url}/embeddings`` API with the OpenAI response shape ({"data": [{"index", "embedding"}]})."""

    def __init__(self, base_url: str, api_key: str | None, model: str, dim: int, timeout: float = 30.0,
                 client: httpx.Client | None = None, sleep=time.sleep, max_retries: int = 2):
        base_url = (base_url or "").strip()
        # a malformed base URL (empty scheme, Markdown link pasted from a chat, stray quotes/brackets) must surface as a clear
        # embedding error on every call -> keyword-only fallback + readable log line, never an unhandled exception
        self._config_error = None if re.fullmatch(r"https?://[^\s\[\]()<>\"']+", base_url) else (
            f"EMBEDDING_BASE_URL is not a plain http(s) URL: {base_url[:60]!r} (no brackets, quotes or Markdown link syntax)")
        if self._config_error:
            log.error(self._config_error)
        self.url = self._url(base_url.rstrip("/"), model)
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model, self.dim, self.name = model, dim, self._name(model)
        self.client = client or httpx.Client(timeout=timeout)
        self._sleep, self.max_retries = sleep, max_retries

    # -- the three things that differ between providers
    @staticmethod
    def _url(base: str, model: str) -> str:
        return base + "/embeddings"

    @staticmethod
    def _name(model: str) -> str:
        return f"openai:{model}"

    def _payload(self, texts: list[str]) -> dict:
        return {"model": self.model, "input": texts}

    @staticmethod
    def _parse(body) -> list:
        return [d["embedding"] for d in sorted(body["data"], key=lambda d: d["index"])]

    def _call(self, texts: list[str]) -> np.ndarray:
        if self._config_error:
            raise EmbeddingError(self._config_error)
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client.post(self.url, headers=self.headers, json=self._payload(texts))
                if r.status_code == 429 or r.status_code >= 500:  # transient: back off and retry
                    last = EmbeddingError(f"embedding API returned {r.status_code}")
                    if attempt < self.max_retries:
                        self._sleep(0.5 * 3 ** attempt)
                        continue
                    raise last
                r.raise_for_status()
                arr = np.asarray(self._parse(r.json()), dtype=np.float32)
                if arr.ndim == 1 and len(texts) == 1:
                    arr = arr[None, :]
                if arr.shape != (len(texts), self.dim):  # wrong model / dimension / token-level output: never feed it to the vector DB
                    raise EmbeddingError(f"embedding API returned shape {tuple(arr.shape)}, expected ({len(texts)}, {self.dim})")
                return _l2(arr)
            except EmbeddingError:
                raise
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last = e
                if attempt < self.max_retries:
                    self._sleep(0.5 * 3 ** attempt)
                    continue
            except (httpx.HTTPError, httpx.InvalidURL, KeyError, ValueError, TypeError, IndexError, AttributeError) as e:
                # any malformed / unexpected response is an *embedding* failure, so callers can degrade to keyword search
                raise EmbeddingError(f"embedding API failed: {type(e).__name__}: {e}") from e
        raise EmbeddingError(f"embedding API unreachable after retries: {last}")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        parts = [self._call(texts[i : i + 64]) for i in range(0, len(texts), 64)]
        return np.concatenate(parts) if parts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text])[0]


class HFInferenceEmbedder(OpenAICompatEmbedder):
    """Hugging Face Inference Providers, feature-extraction route (NOT the OpenAI shape):
    ``POST {base_url}/models/{model}`` with ``{"inputs": [...]}`` returns ``[[... dim floats ...], ...]``.
    Use EMBEDDING_PROVIDER=hf and EMBEDDING_BASE_URL=https://router.huggingface.co/hf-inference (no /v1)."""

    @staticmethod
    def _url(base: str, model: str) -> str:
        return f"{base}/models/{model}"

    @staticmethod
    def _name(model: str) -> str:
        return f"hf:{model}"

    def _payload(self, texts: list[str]) -> dict:
        return {"inputs": texts}

    @staticmethod
    def _parse(body) -> list:
        if isinstance(body, dict):  # e.g. {"error": "Model ... is currently loading"}
            raise EmbeddingError(f"embedding API error: {str(body.get('error', body))[:200]}")
        return body


class CachedQueryEmbedder:
    """LRU cache around embed_query (queries repeat a lot: planner sub-questions, retries, UI refreshes)."""

    def __init__(self, inner: Embedder, size: int = 512):
        self.inner, self.size = inner, size
        self.name, self.dim = inner.name, inner.dim
        self._c: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.hits = self.misses = 0

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if text in self._c:
            self._c.move_to_end(text)
            self.hits += 1
            return self._c[text]
        self.misses += 1
        v = self.inner.embed_query(text)
        self._c[text] = v
        if len(self._c) > self.size:
            self._c.popitem(last=False)
        return v


def build_embedder(s: Settings) -> Embedder:
    p = s.embedding_provider.lower()
    if p == "hashing":
        e: Embedder = HashingEmbedder(s.embedding_dim)
    elif p == "fastembed":
        e = FastEmbedEmbedder(s.embedding_model, s.embedding_dim)
    elif p in ("openai", "hf"):
        if not s.embedding_base_url:
            raise EmbeddingError(f"EMBEDDING_BASE_URL is required for the {p} provider")
        cls = HFInferenceEmbedder if p == "hf" else OpenAICompatEmbedder
        e = cls(s.embedding_base_url, s.embedding_api_key, s.embedding_model, s.embedding_dim)
    else:
        raise EmbeddingError(f"unknown EMBEDDING_PROVIDER '{s.embedding_provider}'")
    cached = CachedQueryEmbedder(e)
    if s.embedding_store_id:
        log.warning("EMBEDDING_STORE_ID override: vectors are served by %s but pinned as %s", e.name, s.embedding_store_id)
        cached.name = s.embedding_store_id
    log.info("embedder ready: %s (dim=%d)", cached.name, e.dim)
    return cached

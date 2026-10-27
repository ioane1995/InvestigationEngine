"""Embedding providers behind one interface.

* HashingEmbedder   - OUR OWN offline embedder: signed feature hashing of word + bigram + char-trigram
                      features into a dense, L2-normalised vector. No model download, deterministic;
                      used for tests, CI, offline evaluation and as a last-resort fallback.
                      It captures lexical/sub-word similarity, NOT true semantics.
* FastEmbedEmbedder - real neural embeddings via ONNX (default BAAI/bge-small-en-v1.5, 384-d), CPU only.
* OpenAICompatEmbedder - any /v1/embeddings compatible HTTP API.
"""
from __future__ import annotations

import logging
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
    def __init__(self, base_url: str, api_key: str | None, model: str, dim: int, timeout: float = 30.0):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model, self.dim, self.name = model, dim, f"openai:{model}"
        self.client = httpx.Client(timeout=timeout)

    def _call(self, texts: list[str]) -> np.ndarray:
        try:
            r = self.client.post(self.url, headers=self.headers, json={"model": self.model, "input": texts})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise EmbeddingError(f"embedding API failed: {e}") from e
        return _l2(np.asarray([d["embedding"] for d in data], dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        parts = [self._call(texts[i : i + 64]) for i in range(0, len(texts), 64)]
        return np.concatenate(parts) if parts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text])[0]


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
    elif p == "openai":
        if not s.embedding_base_url:
            raise EmbeddingError("EMBEDDING_BASE_URL is required for the openai provider")
        e = OpenAICompatEmbedder(s.embedding_base_url, s.embedding_api_key, s.embedding_model, s.embedding_dim)
    else:
        raise EmbeddingError(f"unknown EMBEDDING_PROVIDER '{s.embedding_provider}'")
    log.info("embedder ready: %s (dim=%d)", e.name, e.dim)
    return CachedQueryEmbedder(e)

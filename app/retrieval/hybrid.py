"""Hybrid retrieval: dense (Qdrant) + lexical (own BM25) -> fusion (own RRF / weighted) ->
near-duplicate collapse (own SimHash) -> own feature re-ranker + MMR."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

import logging

from app.embeddings import Embedder, EmbeddingError
from app.models import Chunk, Hit
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from app.retrieval.rerank import FeatureReranker
from app.stores.qdrant_store import QdrantStore
from app.text import hamming


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid"  # hybrid | dense | bm25
    fusion: str = "rrf"  # rrf | weighted
    candidate_k: int = 40
    top_k: int = 8
    rrf_k: int = 60
    w_dense: float = 1.0
    w_bm25: float = 1.0
    rerank: bool = True
    mmr_lambda: float = 0.8
    collapse_near_dups: bool = True
    near_dup_bits: int = 3


class CorpusIndex:
    """In-memory mirror of the chunk payloads + the BM25 index. Rebuilt from Qdrant at start-up and
    updated incrementally on ingest/delete, so lexical search needs no extra infrastructure."""

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self._by_doc: dict[str, list[str]] = {}
        self.bm25 = BM25Index()
        self._lock = threading.RLock()

    def add(self, chunks: Iterable[Chunk]) -> None:
        with self._lock:
            for c in chunks:
                self.chunks[c.chunk_id] = c
                self._by_doc.setdefault(c.doc_id, []).append(c.chunk_id)
                self.bm25.add(c.chunk_id, c.doc_id, f"{c.title} {c.heading} {c.text}", {"format": c.format, "source": c.source})

    def remove_doc(self, doc_id: str) -> None:
        with self._lock:
            self.bm25.remove_doc(doc_id)
            for cid in self._by_doc.pop(doc_id, []):
                self.chunks.pop(cid, None)

    def doc_chunks(self, doc_id: str) -> list[Chunk]:
        with self._lock:
            return sorted((self.chunks[c] for c in self._by_doc.get(doc_id, []) if c in self.chunks), key=lambda c: c.index)


class HybridRetriever:
    def __init__(self, store: QdrantStore, embedder: Embedder, corpus: CorpusIndex):
        self.store, self.embedder, self.corpus = store, embedder, corpus

    def search(self, query: str, cfg: RetrievalConfig = RetrievalConfig(), filters: Optional[dict[str, str]] = None) -> list[Hit]:
        ck = cfg.candidate_k
        dense: list[tuple[str, float]] = []
        lex: list[tuple[str, float]] = []
        dense_failed = False
        if cfg.mode in ("hybrid", "dense"):
            try:
                dense = [(c, s) for c, s in self.store.dense_search(self.embedder.embed_query(query), ck, filters)]
            except Exception as e:  # embedding API down / rate limited / vector DB error: keep serving with keyword search only
                logging.getLogger(__name__).warning("dense retrieval unavailable, falling back to BM25 only: %s: %s", type(e).__name__, e,
                                                    exc_info=not isinstance(e, EmbeddingError))
                dense_failed = True
                if cfg.mode == "dense":
                    cfg = replace(cfg, mode="bm25")
        if cfg.mode in ("hybrid", "bm25"):
            lex = self.corpus.bm25.search(query, ck, filters)

        d_rank = {c: i for i, (c, _) in enumerate(dense, 1)}
        l_rank = {c: i for i, (c, _) in enumerate(lex, 1)}
        if cfg.mode == "dense":
            fused = dict(dense)
        elif cfg.mode == "bm25":
            fused = dict(lex)
        elif cfg.fusion == "weighted":
            fused = weighted_score_fusion([dict(dense), dict(lex)], [cfg.w_dense, cfg.w_bm25])
        else:
            fused = reciprocal_rank_fusion([[c for c, _ in dense], [c for c, _ in lex]], cfg.rrf_k, [cfg.w_dense, cfg.w_bm25])

        hits: list[Hit] = []
        for cid, sc in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            chunk = self.corpus.chunks.get(cid)
            if chunk is None:  # index briefly out of sync (concurrent delete) -> skip safely
                continue
            comp = {}
            if cid in d_rank:
                comp.update(dense_rank=d_rank[cid], dense_score=round(dict(dense)[cid], 4))
            if cid in l_rank:
                comp.update(bm25_rank=l_rank[cid], bm25_score=round(dict(lex)[cid], 4))
            if dense_failed:
                comp["dense_unavailable"] = 1.0
            hits.append(Hit(chunk=chunk, score=sc, components=comp))

        if cfg.collapse_near_dups:
            hits = self._collapse(hits, cfg.near_dup_bits)
        hits = hits[:ck]
        if cfg.rerank:
            return FeatureReranker(self.corpus.bm25, mmr_lambda=cfg.mmr_lambda).rerank(query, hits, cfg.top_k)
        return hits[: cfg.top_k]

    @staticmethod
    def _collapse(hits: list[Hit], bits: int) -> list[Hit]:
        kept: list[tuple[int, Hit]] = []
        for h in hits:
            sh = int(h.chunk.simhash, 16) if h.chunk.simhash else None
            dup = None
            if sh is not None:
                for ksh, kh in kept:
                    if hamming(sh, ksh) <= bits:
                        dup = kh
                        break
            if dup is None:
                kept.append((sh if sh is not None else -1, h))
            elif h.chunk.doc_id not in dup.doc_ids:
                dup.also_doc_ids.append(h.chunk.doc_id)
        return [h for _, h in kept]

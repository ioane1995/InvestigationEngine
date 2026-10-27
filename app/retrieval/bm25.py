"""Our own BM25 (Okapi) inverted index with incremental add / remove-by-document.
No search library is used: postings, IDF, length normalisation and top-k selection are implemented here."""
from __future__ import annotations

import heapq
import math
import threading
from collections import Counter, defaultdict
from typing import Optional

from app.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._lock = threading.RLock()
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)  # term -> {internal_id: tf}
        self._len: dict[int, int] = {}
        self._ext: dict[int, str] = {}  # internal id -> chunk_id
        self._int: dict[str, int] = {}  # chunk_id -> internal id
        self._meta: dict[int, dict[str, str]] = {}
        self._terms: dict[int, list[str]] = {}
        self._by_doc: dict[str, set[int]] = defaultdict(set)
        self._total_len = 0
        self._next = 0

    # ---------------------------------------------------------------- stats
    def __len__(self) -> int:
        return len(self._len)

    @property
    def avgdl(self) -> float:
        return self._total_len / len(self._len) if self._len else 0.0

    def df(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    def idf(self, term: str) -> float:
        """Robertson-Sparck-Jones IDF (always >= 0). Unseen terms get the maximum IDF, which makes a
        query term that appears nowhere in the corpus (e.g. a non-existent incident id) very 'expensive'
        to leave unmatched - this is what the abstention gate relies on."""
        n = len(self._len)
        d = self.df(term)
        return math.log(1.0 + (n - d + 0.5) / (d + 0.5))

    # ---------------------------------------------------------------- updates
    def add(self, chunk_id: str, doc_id: str, text: str, meta: Optional[dict[str, str]] = None) -> None:
        with self._lock:
            if chunk_id in self._int:
                self._remove_internal(self._int[chunk_id])
            iid = self._next
            self._next += 1
            tf = Counter(tokenize(text))
            for t, c in tf.items():
                self._postings[t][iid] = c
            dl = sum(tf.values())
            self._len[iid], self._ext[iid], self._int[chunk_id] = dl, chunk_id, iid
            self._meta[iid] = {"doc_id": doc_id, **(meta or {})}
            self._terms[iid] = list(tf)
            self._by_doc[doc_id].add(iid)
            self._total_len += dl

    def _remove_internal(self, iid: int) -> None:
        for t in self._terms.pop(iid, []):
            p = self._postings.get(t)
            if p is not None:
                p.pop(iid, None)
                if not p:
                    del self._postings[t]
        self._total_len -= self._len.pop(iid, 0)
        cid = self._ext.pop(iid, None)
        if cid is not None:
            self._int.pop(cid, None)
        m = self._meta.pop(iid, None)
        if m:
            s = self._by_doc.get(m["doc_id"])
            if s is not None:
                s.discard(iid)

    def remove_doc(self, doc_id: str) -> int:
        with self._lock:
            ids = list(self._by_doc.pop(doc_id, ()))
            for iid in ids:
                self._remove_internal(iid)
            return len(ids)

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int = 20, filters: Optional[dict[str, str]] = None) -> list[tuple[str, float]]:
        q_terms = list(dict.fromkeys(tokenize(query)))
        if not q_terms or not self._len:
            return []
        scores: dict[int, float] = defaultdict(float)
        avgdl = self.avgdl or 1.0
        with self._lock:
            for t in q_terms:
                post = self._postings.get(t)
                if not post:
                    continue
                idf = self.idf(t)
                for iid, tf in post.items():
                    if filters and any(self._meta[iid].get(fk) != fv for fk, fv in filters.items()):
                        continue
                    dl = self._len[iid]
                    denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                    scores[iid] += idf * tf * (self.k1 + 1) / denom
            top = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
            return [(self._ext[i], s) for i, s in top]

"""Ingestion service: create / update / delete / bulk-ingest with exact-duplicate detection.

Dedup policy (documented in docs/ARCHITECTURE.md):
 1. Exact duplicate *document* (same normalised-content SHA-256 under another doc_id) -> rejected, the
    existing doc_id is returned. Same doc_id + same hash -> 'unchanged' (idempotent re-upload).
 2. Same doc_id + different content -> 'updated': old chunks deleted from Qdrant + BM25, new ones inserted.
 3. Near-duplicate *chunks* across different documents are kept (provenance matters) but are collapsed at
    query time via SimHash, so the LLM context never contains the same paragraph twice and the collapsed
    documents remain visible as `also_doc_ids`."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterable, Optional

from app.config import Settings
from app.embeddings import Embedder
from app.ingestion.chunking import get_chunker
from app.ingestion.loaders import content_hash
from app.models import Chunk, DocRecord, IngestResult, RawDocument
from app.retrieval.hybrid import CorpusIndex
from app.stores.qdrant_store import QdrantStore

log = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, store: QdrantStore, embedder: Embedder, corpus: CorpusIndex, settings: Settings):
        self.store, self.embedder, self.corpus, self.s = store, embedder, corpus, settings
        self.docs: dict[str, DocRecord] = {}
        self.hashes: dict[str, str] = {}
        self._lock = threading.RLock()
        self.version = 0  # bumped on every change; used to invalidate the answer cache

    # ------------------------------------------------------------------ start-up
    def load_from_store(self) -> None:
        """Rebuild in-memory mirrors (registry + BM25) from the persistent store after a restart."""
        t0 = time.time()
        cursor = None
        while True:
            recs, cursor = self.store.list_docs(limit=1000, cursor=cursor)
            for r in recs:
                self.docs[r.doc_id], self.hashes[r.content_hash] = r, r.doc_id
            if cursor is None:
                break
        batch: list[Chunk] = []
        for c in self.store.scroll_chunks():
            batch.append(c)
            if len(batch) >= 2000:
                self.corpus.add(batch)
                batch = []
        self.corpus.add(batch)
        log.info("loaded %d docs / %d chunks from store in %.1fs", len(self.docs), len(self.corpus.chunks), time.time() - t0)

    # ------------------------------------------------------------------ helpers
    def _plan(self, raw: RawDocument) -> tuple[str, Optional[IngestResult]]:
        h = content_hash(raw.text)
        existing = self.docs.get(raw.doc_id)
        if existing and existing.content_hash == h:
            return h, IngestResult(raw.doc_id, "unchanged", existing.n_chunks, "identical content already indexed")
        other = self.hashes.get(h)
        if other and other != raw.doc_id:
            return h, IngestResult(raw.doc_id, "duplicate", 0, f"identical content already indexed as '{other}'", duplicate_of=other)
        return h, None

    def _chunk(self, raw: RawDocument, strategy: Optional[str]) -> tuple[list[Chunk], str]:
        name = strategy or self.s.chunk_strategy
        chunker = get_chunker(name, self.s.chunk_max_tokens, self.s.chunk_overlap_tokens)
        return chunker.split(raw), name

    # ------------------------------------------------------------------ public API
    def ingest(self, raw: RawDocument, strategy: Optional[str] = None) -> IngestResult:
        return self.ingest_many([raw], strategy)[0]

    def ingest_many(self, raws: Iterable[RawDocument], strategy: Optional[str] = None, batch_docs: int = 64,
                    progress: Optional[Callable[[int], None]] = None) -> list[IngestResult]:
        results: list[IngestResult] = []
        raws = list(raws)
        for i in range(0, len(raws), batch_docs):
            results.extend(self._ingest_batch(raws[i : i + batch_docs], strategy))
            if progress:
                progress(min(i + batch_docs, len(raws)))
        return results

    def _ingest_batch(self, raws: list[RawDocument], strategy: Optional[str]) -> list[IngestResult]:
        with self._lock:
            results: dict[str, IngestResult] = {}
            work: list[tuple[RawDocument, str, list[Chunk], str, bool]] = []
            seen_in_batch: dict[str, str] = {}
            for raw in raws:
                try:
                    h, early = self._plan(raw)
                    if early is None and h in seen_in_batch and seen_in_batch[h] != raw.doc_id:
                        early = IngestResult(raw.doc_id, "duplicate", 0, f"identical to '{seen_in_batch[h]}' in the same batch",
                                             duplicate_of=seen_in_batch[h])
                    if early:
                        results[raw.doc_id] = early
                        continue
                    chunks, name = self._chunk(raw, strategy)
                    if not chunks:
                        results[raw.doc_id] = IngestResult(raw.doc_id, "failed", 0, "no chunks produced")
                        continue
                    seen_in_batch[h] = raw.doc_id
                    work.append((raw, h, chunks, name, raw.doc_id in self.docs))
                except Exception as e:  # one bad document must not kill the batch
                    log.exception("ingest failed for %s", raw.doc_id)
                    results[raw.doc_id] = IngestResult(raw.doc_id, "failed", 0, f"{type(e).__name__}: {e}")
            if work:
                all_chunks = [c for _, _, cs, _, _ in work for c in cs]
                try:
                    vecs = self.embedder.embed_documents([c.embed_text for c in all_chunks])
                    for raw, _, _, _, is_update in work:  # remove stale versions only after embeddings succeeded
                        if is_update:
                            self.store.delete_doc_chunks(raw.doc_id)
                            self.corpus.remove_doc(raw.doc_id)
                    self.store.upsert_chunks(all_chunks, vecs)
                    self.corpus.add(all_chunks)
                    now = time.time()
                    recs = []
                    for raw, h, cs, name, is_update in work:
                        old = self.docs.get(raw.doc_id)
                        if old:
                            self.hashes.pop(old.content_hash, None)
                        rec = DocRecord(raw.doc_id, raw.title, raw.source, raw.format, h, len(cs), name,
                                        old.created_at if old else now, now, len(raw.text),
                                        {k: v for k, v in raw.metadata.items() if isinstance(v, (str, int, float, bool))})
                        self.docs[raw.doc_id], self.hashes[h] = rec, raw.doc_id
                        recs.append(rec)
                        results[raw.doc_id] = IngestResult(raw.doc_id, "updated" if is_update else "created", len(cs))
                    self.store.put_docs(recs)
                    self.version += 1
                except Exception as e:
                    log.exception("batch embedding/upsert failed")
                    for raw, *_ in work:
                        results[raw.doc_id] = IngestResult(raw.doc_id, "failed", 0, f"{type(e).__name__}: {e}")
            return [results[r.doc_id] for r in raws if r.doc_id in results]

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            rec = self.docs.pop(doc_id, None)
            if rec is None:
                return False
            self.hashes.pop(rec.content_hash, None)
            self.store.delete_doc_chunks(doc_id)
            self.store.delete_doc_record(doc_id)
            self.corpus.remove_doc(doc_id)
            self.version += 1
            return True

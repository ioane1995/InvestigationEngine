"""Persistent storage on Qdrant: one collection for chunk vectors, one small collection for the
document registry (+ one metadata point that pins the embedding model, so a collection built with
model A can never be silently queried with model B)."""
from __future__ import annotations

import logging
import uuid
import warnings
from typing import Any, Iterator, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import Settings
from app.models import Chunk, DocRecord

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="Payload indexes have no effect in the local Qdrant")
_NS = uuid.UUID("6f1f4b0e-2f0a-4a53-9a39-0c1c7a2b7f10")
_META_ID = str(uuid.uuid5(_NS, "__meta__"))


def _pid(key: str) -> str:
    return str(uuid.uuid5(_NS, key))


class StoreMismatch(RuntimeError):
    pass


class QdrantStore:
    def __init__(self, settings: Settings, dim: int, embedder_name: str, client: Optional[QdrantClient] = None):
        if client is not None:
            self.client = client
        elif settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)
        elif settings.qdrant_path == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=settings.qdrant_path)
        self.chunks_col = f"{settings.collection_prefix}_chunks"
        self.docs_col = f"{settings.collection_prefix}_docs"
        self.dim, self.embedder_name = dim, embedder_name
        self._ensure()

    # ------------------------------------------------------------------ setup
    def _ensure(self) -> None:
        c = self.client
        if not c.collection_exists(self.chunks_col):
            c.create_collection(self.chunks_col, vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE))
            for f in ("doc_id", "format", "source"):
                c.create_payload_index(self.chunks_col, f, qm.PayloadSchemaType.KEYWORD)
        if not c.collection_exists(self.docs_col):
            c.create_collection(self.docs_col, vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT))
            for f in ("kind", "content_hash"):
                c.create_payload_index(self.docs_col, f, qm.PayloadSchemaType.KEYWORD)
        meta = c.retrieve(self.docs_col, ids=[_META_ID], with_payload=True)
        if meta:
            p = meta[0].payload or {}
            if p.get("embedder") != self.embedder_name or int(p.get("dim", -1)) != self.dim:
                raise StoreMismatch(
                    f"vector store was built with '{p.get('embedder')}' (dim={p.get('dim')}) but the app is configured "
                    f"with '{self.embedder_name}' (dim={self.dim}). Re-ingest or change EMBEDDING_* settings."
                )
        else:
            c.upsert(self.docs_col, points=[qm.PointStruct(id=_META_ID, vector=[1.0],
                     payload={"kind": "meta", "embedder": self.embedder_name, "dim": self.dim})])

    # ------------------------------------------------------------------ chunks
    def upsert_chunks(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 256) -> None:
        for i in range(0, len(chunks), batch):
            pts = [
                qm.PointStruct(id=_pid(c.chunk_id), vector=vectors[i + j].tolist(), payload=c.to_payload())
                for j, c in enumerate(chunks[i : i + batch])
            ]
            self.client.upsert(self.chunks_col, points=pts, wait=True)

    def delete_doc_chunks(self, doc_id: str) -> None:
        self.client.delete(self.chunks_col, points_selector=qm.FilterSelector(filter=self._doc_filter(doc_id)), wait=True)

    @staticmethod
    def _doc_filter(doc_id: str) -> qm.Filter:
        return qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])

    @staticmethod
    def _filter(filters: Optional[dict[str, str]]) -> Optional[qm.Filter]:
        if not filters:
            return None
        return qm.Filter(must=[qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k, v in filters.items()])

    def dense_search(self, vector: np.ndarray, k: int, filters: Optional[dict[str, str]] = None) -> list[tuple[str, float]]:
        res = self.client.query_points(self.chunks_col, query=vector.tolist(), limit=k, with_payload=["chunk_id"],
                                       query_filter=self._filter(filters))
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def scroll_chunks(self, batch: int = 1000) -> Iterator[Chunk]:
        offset = None
        while True:
            pts, offset = self.client.scroll(self.chunks_col, limit=batch, offset=offset, with_payload=True, with_vectors=False)
            for p in pts:
                yield Chunk.from_payload(p.payload or {})
            if offset is None:
                break

    def get_doc_chunks(self, doc_id: str) -> list[Chunk]:
        out: list[Chunk] = []
        offset = None
        while True:
            pts, offset = self.client.scroll(self.chunks_col, scroll_filter=self._doc_filter(doc_id), limit=200,
                                             offset=offset, with_payload=True, with_vectors=False)
            out.extend(Chunk.from_payload(p.payload or {}) for p in pts)
            if offset is None:
                break
        return sorted(out, key=lambda c: c.index)

    def count_chunks(self) -> int:
        return self.client.count(self.chunks_col, exact=True).count

    # ------------------------------------------------------------------ document registry
    def put_doc(self, rec: DocRecord) -> None:
        self.client.upsert(self.docs_col, points=[qm.PointStruct(id=_pid(rec.doc_id), vector=[1.0], payload=rec.to_payload())], wait=True)

    def put_docs(self, recs: list[DocRecord], batch: int = 500) -> None:
        for i in range(0, len(recs), batch):
            pts = [qm.PointStruct(id=_pid(r.doc_id), vector=[1.0], payload=r.to_payload()) for r in recs[i : i + batch]]
            self.client.upsert(self.docs_col, points=pts, wait=True)

    def get_doc(self, doc_id: str) -> Optional[DocRecord]:
        r = self.client.retrieve(self.docs_col, ids=[_pid(doc_id)], with_payload=True)
        return DocRecord.from_payload(r[0].payload) if r and (r[0].payload or {}).get("kind") == "doc" else None

    def delete_doc_record(self, doc_id: str) -> None:
        self.client.delete(self.docs_col, points_selector=qm.PointIdsList(points=[_pid(doc_id)]), wait=True)

    def find_by_hash(self, content_hash: str) -> Optional[DocRecord]:
        flt = qm.Filter(must=[qm.FieldCondition(key="kind", match=qm.MatchValue(value="doc")),
                              qm.FieldCondition(key="content_hash", match=qm.MatchValue(value=content_hash))])
        pts, _ = self.client.scroll(self.docs_col, scroll_filter=flt, limit=1, with_payload=True, with_vectors=False)
        return DocRecord.from_payload(pts[0].payload) if pts else None

    def list_docs(self, limit: int = 50, cursor: Optional[str] = None) -> tuple[list[DocRecord], Optional[str]]:
        flt = qm.Filter(must=[qm.FieldCondition(key="kind", match=qm.MatchValue(value="doc"))])
        pts, nxt = self.client.scroll(self.docs_col, scroll_filter=flt, limit=limit, offset=cursor,
                                      with_payload=True, with_vectors=False)
        return [DocRecord.from_payload(p.payload or {}) for p in pts], (str(nxt) if nxt is not None else None)

    def count_docs(self) -> int:
        flt = qm.Filter(must=[qm.FieldCondition(key="kind", match=qm.MatchValue(value="doc"))])
        return self.client.count(self.docs_col, count_filter=flt, exact=True).count

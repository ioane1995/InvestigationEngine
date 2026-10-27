"""Plain dataclasses shared by every layer (no framework coupling)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RawDocument:
    doc_id: str
    title: str
    text: str
    source: str
    format: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str  # what the user / LLM sees
    title: str
    heading: str
    source: str
    format: str
    strategy: str
    n_tokens: int
    simhash: str = ""  # 64-bit hex, used for near-duplicate collapsing
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Contextual embedding text: title and heading are prepended so that a chunk
        such as '- Duration: 47 minutes' still knows which document it belongs to."""
        ctx = " | ".join(p for p in (self.title, self.heading) if p)
        return f"{ctx}\n{self.text}" if ctx else self.text

    def to_payload(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = "chunk"
        return d

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> "Chunk":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: p[k] for k in keys if k in p})


@dataclass
class DocRecord:
    doc_id: str
    title: str
    source: str
    format: str
    content_hash: str
    n_chunks: int
    strategy: str
    created_at: float
    updated_at: float
    n_chars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = "doc"
        return d

    @classmethod
    def from_payload(cls, p: dict[str, Any]) -> "DocRecord":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: p[k] for k in keys if k in p})


@dataclass
class Hit:
    """A retrieved chunk with a full score breakdown (kept for explainability / UI)."""
    chunk: Chunk
    score: float
    components: dict[str, float] = field(default_factory=dict)
    also_doc_ids: list[str] = field(default_factory=list)  # near-duplicate docs collapsed into this hit
    via: str = "query"  # query | hop | tool

    @property
    def doc_ids(self) -> list[str]:
        return [self.chunk.doc_id, *self.also_doc_ids]


@dataclass
class IngestResult:
    doc_id: str
    status: str  # created | updated | unchanged | duplicate | failed
    n_chunks: int = 0
    detail: str = ""
    duplicate_of: Optional[str] = None

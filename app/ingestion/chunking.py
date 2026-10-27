"""Two chunking strategies that are compared in the evaluation (see eval/run_eval.py --chunking).

* ``fixed``     - sliding window of N whitespace tokens with overlap. Simple, predictable size,
                  but cuts through sentences/sections and loses headings.
* ``structure`` - structure-aware: splits on headings and paragraphs, packs whole sentences up to a
                  token budget, carries the current heading with every chunk and overlaps by one
                  sentence. Small records (e.g. 'Duration: 47 minutes') stay attached to their
                  section instead of being split away from it.
"""
from __future__ import annotations

import re
from typing import Protocol

from app.models import Chunk, RawDocument
from app.text import count_tokens, simhash_hex, split_sentences

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


class Chunker(Protocol):
    name: str

    def split(self, doc: RawDocument) -> list[Chunk]: ...


def _mk(doc: RawDocument, idx: int, text: str, heading: str, strategy: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}::{idx}",
        doc_id=doc.doc_id,
        index=idx,
        text=text,
        title=doc.title,
        heading=heading,
        source=doc.source,
        format=doc.format,
        strategy=strategy,
        n_tokens=count_tokens(text),
        simhash=simhash_hex(text),
        metadata={k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))},
    )


class FixedWindowChunker:
    name = "fixed"

    def __init__(self, max_tokens: int = 200, overlap: int = 40):
        if overlap >= max_tokens:
            raise ValueError("overlap must be smaller than max_tokens")
        self.max_tokens, self.overlap = max_tokens, overlap

    def split(self, doc: RawDocument) -> list[Chunk]:
        words = doc.text.split()
        if not words:
            return []
        step = self.max_tokens - self.overlap
        chunks: list[Chunk] = []
        for start in range(0, len(words), step):
            window = words[start : start + self.max_tokens]
            chunks.append(_mk(doc, len(chunks), " ".join(window), "", self.name))
            if start + self.max_tokens >= len(words):
                break
        return chunks


class StructureAwareChunker:
    name = "structure"

    def __init__(self, max_tokens: int = 200, overlap: int = 40, min_tokens: int = 25):
        self.max_tokens, self.min_tokens = max_tokens, min_tokens
        self.overlap_sentences = 1 if overlap > 0 else 0

    def split(self, doc: RawDocument) -> list[Chunk]:
        sections: list[tuple[str, str]] = []
        stack: list[tuple[int, str]] = []
        body: list[str] = []
        head = ""

        def flush():
            nonlocal body
            txt = "\n".join(body).strip()
            if txt:
                sections.append((head, txt))
            body = []

        for line in doc.text.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                flush()
                level = len(m.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, m.group(2)))
                head = " > ".join(h for _, h in stack)
            else:
                body.append(line)
        flush()

        chunks: list[Chunk] = []
        for heading, text in sections:
            if heading == doc.title:  # don't repeat the title as a heading
                heading = ""
            sents = split_sentences(text)
            buf: list[str] = []
            size = 0
            for s in sents:
                n = count_tokens(s)
                if n > self.max_tokens:  # a single huge "sentence": hard-wrap it
                    if buf:
                        chunks.append(_mk(doc, len(chunks), "\n".join(buf), heading, self.name))
                        buf, size = [], 0
                    w = s.split()
                    for i in range(0, len(w), self.max_tokens):
                        chunks.append(_mk(doc, len(chunks), " ".join(w[i : i + self.max_tokens]), heading, self.name))
                    continue
                if size + n > self.max_tokens and buf:
                    chunks.append(_mk(doc, len(chunks), "\n".join(buf), heading, self.name))
                    buf = buf[-self.overlap_sentences :] if self.overlap_sentences else []
                    size = sum(count_tokens(x) for x in buf)
                buf.append(s)
                size += n
            if buf:
                chunks.append(_mk(doc, len(chunks), "\n".join(buf), heading, self.name))

        # merge tiny trailing chunks into their predecessor (same heading) to avoid noise chunks
        merged: list[Chunk] = []
        for c in chunks:
            if merged and c.n_tokens < self.min_tokens and merged[-1].heading == c.heading \
                    and merged[-1].n_tokens + c.n_tokens <= self.max_tokens * 1.25:
                prev = merged[-1]
                prev.text += "\n" + c.text
                prev.n_tokens = count_tokens(prev.text)
                prev.simhash = simhash_hex(prev.text)
            else:
                merged.append(c)
        for i, c in enumerate(merged):  # re-index after merging
            c.index, c.chunk_id = i, f"{doc.doc_id}::{i}"
        return merged


def get_chunker(name: str, max_tokens: int = 200, overlap: int = 40) -> Chunker:
    if name == "fixed":
        return FixedWindowChunker(max_tokens, overlap)
    if name == "structure":
        return StructureAwareChunker(max_tokens, overlap)
    raise ValueError(f"unknown chunking strategy '{name}' (use 'fixed' or 'structure')")

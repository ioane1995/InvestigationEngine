"""Bulk-ingest a corpus.jsonl (or a folder of files) into the configured store (Qdrant Cloud in production).
Idempotent / resumable: unchanged documents are skipped, so you can re-run after an interruption.

   python -m scripts.ingest_corpus --corpus data/corpus_50k/corpus.jsonl            # uses .env (QDRANT_URL, EMBEDDING_PROVIDER...)
   python -m scripts.ingest_corpus --dir data/sample_files                          # real files"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from app.config import get_settings
from app.container import build_container
from app.ingestion.loaders import LoaderError, UnsupportedFormat, load_bytes
from app.models import RawDocument


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", help="jsonl with doc_id,title,text,format,source")
    ap.add_argument("--dir", help="folder of real files")
    ap.add_argument("--chunking", default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level="INFO")
    s = get_settings()
    c = build_container(s)
    raws: list[RawDocument] = []
    if a.corpus:
        for line in Path(a.corpus).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                raws.append(RawDocument(d["doc_id"], d["title"], d["text"], d.get("source", d["doc_id"]), d.get("format", "txt"), d.get("metadata", {})))
    if a.dir:
        for p in sorted(Path(a.dir).iterdir()):
            try:
                raws.append(load_bytes(p.read_bytes(), p.name))
            except (UnsupportedFormat, LoaderError) as e:
                print(f"skip {p.name}: {e}")
    if a.limit:
        raws = raws[: a.limit]
    t = time.time()
    counts: dict[str, int] = {}

    def prog(n: int) -> None:
        print(f"\r{n}/{len(raws)} docs  ({n / max(time.time() - t, 1e-9):.0f} docs/s)", end="", flush=True)

    for r in c.ingestion.ingest_many(raws, a.chunking, a.batch, prog):
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"\nfinished in {time.time() - t:.0f}s: {counts}; store now has {len(c.ingestion.docs)} docs / {c.store.count_chunks()} chunks")


if __name__ == "__main__":
    main()

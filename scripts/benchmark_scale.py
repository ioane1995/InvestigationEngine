"""Scale benchmark: ingest a large corpus, then measure retrieval / pipeline latency and memory.
   python -m scripts.benchmark_scale --corpus data/corpus_50k --queries 200
Uses the offline hashing embedder + in-memory Qdrant by default (set --embedder fastembed for real numbers)."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from app.config import Settings
from app.retrieval.hybrid import RetrievalConfig
from eval.metrics import percentile
from eval.runner import build_system, load_jsonl
from app.embeddings import build_embedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_50k")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--embedder", default="hashing")
    ap.add_argument("--out", default="eval/results/scale.json")
    a = ap.parse_args()
    docs = load_jsonl(Path(a.corpus) / "corpus.jsonl")
    questions = load_jsonl(Path(a.corpus) / "questions.jsonl")
    s = Settings(qdrant_path=":memory:", embedding_provider=a.embedder, llm_provider="heuristic")
    t = time.perf_counter()
    c, info = build_system(docs, "structure", build_embedder(s), None, s)
    total = time.perf_counter() - t
    qs = [q["question"] for q in questions] * (a.queries // max(len(questions), 1) + 1)
    qs = qs[: a.queries]
    lat = {"dense": [], "bm25": [], "hybrid": [], "hybrid+rerank": [], "full_pipeline": []}
    for q in qs:
        for name, cfg in (("dense", RetrievalConfig(mode="dense", rerank=False)), ("bm25", RetrievalConfig(mode="bm25", rerank=False)),
                          ("hybrid", RetrievalConfig(rerank=False)), ("hybrid+rerank", RetrievalConfig())):
            t0 = time.perf_counter(); c.retriever.search(q, cfg); lat[name].append((time.perf_counter() - t0) * 1000)
    from app.rag.types import RagConfig
    for q in qs[: min(60, len(qs))]:
        t0 = time.perf_counter(); c.engine.ask(q, RagConfig(use_planner=False)); lat["full_pipeline"].append((time.perf_counter() - t0) * 1000)
    out = {"documents_in_file": len(docs), "indexed_documents": len(c.ingestion.docs), "chunks": len(c.corpus.chunks), "embedder": c.embedder.name,
           "ingest": info, "ingest_docs_per_s": round(len(c.ingestion.docs) / info["ingest_s"], 1),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
           "latency_ms": {k: {"p50": round(percentile(v, .5), 1), "p95": round(percentile(v, .95), 1), "n": len(v)} for k, v in lat.items()},
           "bm25_vocab": len(c.corpus.bm25._postings)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

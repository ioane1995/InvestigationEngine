"""Evaluation harness: ingest a corpus into a fresh in-memory Qdrant, run configurations, compute metrics."""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.config import Settings
from app.container import Container, build_container
from app.embeddings import Embedder
from app.llm.base import LLM
from app.models import RawDocument
from app.rag.types import RagConfig
from app.retrieval.hybrid import RetrievalConfig
from eval import metrics as M


@dataclass
class EvalConfig:
    name: str
    retrieval: RetrievalConfig
    rag: RagConfig
    chunking: str = "structure"
    description: str = ""


def standard_configs(llm_is_real: bool) -> list[EvalConfig]:
    plan = llm_is_real  # the offline heuristic planner is the identity function -> skip the useless call
    r = RetrievalConfig
    return [
        EvalConfig("dense_only", r(mode="dense", rerank=False), RagConfig(use_planner=False, hops=0, use_gate=False), description="dense vectors only"),
        EvalConfig("bm25_only", r(mode="bm25", rerank=False), RagConfig(use_planner=False, hops=0, use_gate=False), description="own BM25 only"),
        EvalConfig("hybrid_rrf", r(mode="hybrid", rerank=False), RagConfig(use_planner=False, hops=0, use_gate=False), description="dense + BM25 fused with RRF"),
        EvalConfig("hybrid_rrf_rerank", r(mode="hybrid", rerank=True), RagConfig(use_planner=False, hops=0, use_gate=False), description="+ own feature re-ranker / MMR"),
        EvalConfig("hybrid_weighted_rerank", r(mode="hybrid", fusion="weighted", rerank=True), RagConfig(use_planner=False, hops=0, use_gate=False),
                   description="min-max weighted fusion instead of RRF"),
        EvalConfig("full_no_gate", r(mode="hybrid", rerank=True), RagConfig(use_planner=plan, hops=1, use_gate=False), description="+ hop expansion"),
        EvalConfig("full", r(mode="hybrid", rerank=True), RagConfig(use_planner=plan, hops=1, use_gate=True), description="+ abstention gate (production default)"),
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def to_raw(d: dict[str, Any]) -> RawDocument:
    return RawDocument(d["doc_id"], d["title"], d["text"], d.get("source", d["doc_id"]), d.get("format", "txt"), d.get("metadata", {}))


def build_system(docs: list[dict[str, Any]], chunking: str, embedder: Optional[Embedder] = None, llm: Optional[LLM] = None,
                 settings: Optional[Settings] = None) -> tuple[Container, dict[str, Any]]:
    s = settings or Settings(qdrant_path=":memory:", embedding_provider="hashing", llm_provider="heuristic", chunk_strategy=chunking)
    s = s.model_copy(update={"qdrant_path": ":memory:", "qdrant_url": None, "chunk_strategy": chunking})
    c = build_container(s, embedder=embedder, llm=llm, load=False)
    t = time.perf_counter()
    res = c.ingestion.ingest_many([to_raw(d) for d in docs], chunking, batch_docs=128)
    st = Counter(r.status for r in res)
    return c, {"ingest_s": round(time.perf_counter() - t, 2), "statuses": dict(st), "chunks": len(c.corpus.chunks), "chunking": chunking}


def run_config(c: Container, questions: list[dict[str, Any]], cfg: EvalConfig, retrieval_k: int = 10) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for q in questions:
        gold = q["gold_doc_ids"]
        t = time.perf_counter()
        hits = c.retriever.search(q["question"], RetrievalConfigK(cfg.retrieval, retrieval_k))
        ranked = M.hit_docs(hits)
        ret_ms = (time.perf_counter() - t) * 1000
        ans = c.engine.ask(q["question"], cfg.rag, cfg.retrieval)
        ok = M.answer_correct(q, ans)
        cp, cr = M.citation_scores(q, ans)
        pipe_recall = M.recall_at_k([set(ans.evidence_doc_ids)], gold, 1)  # recall of everything the pipeline gathered
        rows.append({
            "id": q["id"], "type": q["type"], "question": q["question"], "expected": q["expected_status"], "got": ans.status,
            "correct": ok, "failure": M.classify_failure(q, ans, ok), "answer": ans.answer[:300],
            "recall@1": M.recall_at_k(ranked, gold, 1), "recall@3": M.recall_at_k(ranked, gold, 3), "recall@5": M.recall_at_k(ranked, gold, 5),
            "recall@10": M.recall_at_k(ranked, gold, 10), "full_recall@5": M.full_recall_at_k(ranked, gold, 5), "mrr": M.reciprocal_rank(ranked, gold),
            "ndcg@10": M.ndcg_at_k(ranked, gold, 10), "evidence_recall": pipe_recall, "cite_precision": cp, "cite_recall": cr,
            "faithfulness": M.faithfulness(ans), "retrieval_ms": ret_ms, "total_ms": ans.timings_ms.get("total_ms", 0.0),
            "llm_calls": ans.usage.get("llm_calls", 0), "tokens": ans.usage.get("prompt_tokens", 0) + ans.usage.get("completion_tokens", 0),
            "cost_usd": ans.usage.get("cost_usd", 0.0), "degraded": ans.degraded,
        })
    return {"config": cfg.name, "description": cfg.description, "chunking": cfg.chunking, "rows": rows, "summary": summarize(rows)}


def RetrievalConfigK(cfg: RetrievalConfig, k: int) -> RetrievalConfig:
    from dataclasses import replace
    return replace(cfg, top_k=k, candidate_k=max(cfg.candidate_k, k * 3))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ans_rows = [r for r in rows if r["type"] != "unanswerable"]
    un = [r for r in rows if r["type"] == "unanswerable"]
    cf = [r for r in rows if r["type"] == "conflict"]
    by_type: dict[str, dict[str, float]] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
    for t, rs in groups.items():
        by_type[t] = {"n": len(rs), "answer_correctness": round(M.mean(float(r["correct"]) for r in rs), 3),
                      "recall@5": round(M.mean(r["recall@5"] for r in rs), 3), "mrr": round(M.mean(r["mrr"] for r in rs), 3)}
    pred_abstain = [r for r in rows if r["got"] == "insufficient_evidence"]
    failures = Counter(r["failure"] for r in rows if r["failure"] != "ok")
    f = lambda x: round(x, 3) if isinstance(x, float) and not math.isnan(x) else None  # noqa: E731
    return {
        "n_questions": len(rows),
        "recall@1": f(M.mean(r["recall@1"] for r in rows)), "recall@3": f(M.mean(r["recall@3"] for r in rows)),
        "recall@5": f(M.mean(r["recall@5"] for r in rows)), "recall@10": f(M.mean(r["recall@10"] for r in rows)),
        "full_recall@5": f(M.mean(r["full_recall@5"] for r in rows)), "mrr": f(M.mean(r["mrr"] for r in rows)), "ndcg@10": f(M.mean(r["ndcg@10"] for r in rows)),
        "evidence_recall": f(M.mean(r["evidence_recall"] for r in rows)),
        "answer_correctness": f(M.mean(float(r["correct"]) for r in rows)),
        "answer_correctness_answerable": f(M.mean(float(r["correct"]) for r in ans_rows if r["expected"] == "answered")),
        "abstention_recall": f(M.mean(float(r["got"] == "insufficient_evidence") for r in un)),
        "abstention_precision": f((sum(1 for r in pred_abstain if r["type"] == "unanswerable") / len(pred_abstain)) if pred_abstain else math.nan),
        "conflict_detection_recall": f(M.mean(float(r["got"] == "conflicting_evidence") for r in cf)),
        "citation_precision": f(M.mean(r["cite_precision"] for r in rows)), "citation_recall": f(M.mean(r["cite_recall"] for r in rows)),
        "faithfulness": f(M.mean(r["faithfulness"] for r in rows)),
        "latency_p50_ms": f(M.percentile([r["total_ms"] for r in rows], 0.5)), "latency_p95_ms": f(M.percentile([r["total_ms"] for r in rows], 0.95)),
        "retrieval_p50_ms": f(M.percentile([r["retrieval_ms"] for r in rows], 0.5)),
        "avg_llm_calls": f(M.mean(float(r["llm_calls"]) for r in rows)), "avg_tokens": f(M.mean(float(r["tokens"]) for r in rows)),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 5), "by_type": by_type, "failures": dict(failures),
    }

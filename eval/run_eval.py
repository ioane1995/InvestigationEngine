"""Evaluation CLI. Examples:
  python -m eval.run_eval --corpus data/corpus_heldout --embedder fastembed --llm openai --configs hybrid_rrf_rerank,full --pause 10 --retry-degraded 2 --tag real_a
  python -m eval.run_eval --corpus data/corpus_50k --existing-store --retrieval-only --tag retrieval_50k   # no LLM, no re-ingest
Writes eval/results/<tag>.json, <tag>.md and a resumable <tag>.partial.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import Settings, get_settings
from app.container import build_container, build_llm
from app.embeddings import build_embedder
from eval.runner import EvalConfig, QuotaExhausted, build_system, load_jsonl, run_config, standard_configs, stratified_sample

COLS = ["degraded_rate", "recall@1", "recall@5", "recall@10", "mrr", "evidence_recall", "answer_correctness", "abstention_recall", "conflict_detection_recall",
        "citation_precision", "citation_recall", "faithfulness", "latency_p50_ms", "avg_llm_calls"]


def md_table(results: list[dict]) -> str:
    head = "| config | " + " | ".join(COLS) + " |\n|" + "---|" * (len(COLS) + 1)
    lines = [head]
    for r in results:
        s = r["summary"]
        lines.append(f"| {r['config']}{'/' + r['chunking'] if r.get('chunking') else ''} | " + " | ".join(str(s.get(c)) for c in COLS) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/corpus_small")
    ap.add_argument("--configs", default="all", help="comma list or 'all'")
    ap.add_argument("--chunking", default="structure", help="structure | fixed | both")
    ap.add_argument("--embedder", default=None, help="hashing (default) | fastembed | openai; with --existing-store: value from .env")
    ap.add_argument("--llm", default=None, help="heuristic (default) | openai; with --existing-store: value from .env")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to wait after each question (free-tier rate limits: try 8-15)")
    ap.add_argument("--retry-degraded", type=int, default=0, help="re-ask questions that fell back to the offline reader (LLM outage / 429)")
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM entirely: retrieval metrics only (fast, free)")
    ap.add_argument("--existing-store", action="store_true", help="evaluate the store configured in .env (e.g. Qdrant Cloud) instead of re-ingesting the corpus")
    ap.add_argument("--no-planner", action="store_true", help="skip the planning LLM call (about 1/3 fewer tokens per question)")
    ap.add_argument("--evidence-k", type=int, default=0, help="number of evidence chunks given to the LLM (default 10; 6 saves tokens)")
    ap.add_argument("--sample", type=int, default=0, help="evaluate only N questions (stratified by type, reproducible) - saves LLM quota")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run with the same --tag (answered questions are not repeated)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    partial = out / f"{a.tag}.partial.jsonl"
    done: dict[tuple[str, str], dict[str, dict]] = {}
    resume_lines = partial.read_text(encoding="utf-8").splitlines() if (a.resume and partial.exists()) else []
    if not a.resume and partial.exists():
        partial.unlink()

    def save_row(cfg: EvalConfig, row: dict) -> None:
        with open(partial, "a", encoding="utf-8") as f:
            f.write(json.dumps({"config": cfg.name, "chunking": cfg.chunking, "llm": llm.name, "embedder": embedder.name, "row": row}, default=str) + "\n")

    if a.existing_store:
        s = get_settings().model_copy(update={k: v for k, v in (("embedding_provider", a.embedder), ("llm_provider", a.llm)) if v})
    else:
        s = Settings(qdrant_path=":memory:", embedding_provider=a.embedder or "hashing", llm_provider=a.llm or "heuristic")
    embedder = build_embedder(s)
    try:
        llm = build_llm(s, strict=True)  # never fall back silently to the offline reader during an evaluation
    except ValueError as e:
        raise SystemExit(f"ERROR: {e}")
    print(f"==> LLM: {llm.name} | embeddings: {embedder.name}")
    real_llm = llm.name != "heuristic-extractive"
    reused = ignored = 0
    for line in resume_lines:  # rows from another model / embedder would silently mix two systems in one table
        d = json.loads(line)
        if d.get("llm") == llm.name and d.get("embedder") == embedder.name:
            done.setdefault((d["config"], d["chunking"]), {})[d["row"]["id"]] = d["row"]
            reused += 1
        else:
            ignored += 1
    if resume_lines:
        print(f"resuming: reused {reused} finished questions; IGNORED {ignored} rows produced by a different LLM/embedder (or an older file without that information)")
    questions = load_jsonl(Path(a.corpus) / "questions.jsonl")
    if a.sample:
        questions = stratified_sample(questions, a.sample)
        print(f"using a stratified sample of {len(questions)} questions")
    cfgs = standard_configs(real_llm)
    if a.configs != "all":
        want = set(a.configs.split(","))
        cfgs = [c for c in cfgs if c.name in want]
    from dataclasses import replace
    if a.no_planner or a.evidence_k:
        cfgs = [EvalConfig(c.name, c.retrieval, replace(c.rag, **({"use_planner": False} if a.no_planner else {}), **({"evidence_k": a.evidence_k} if a.evidence_k else {})), c.chunking, c.description) for c in cfgs]
        print(f"token saver: planner={'off' if a.no_planner else 'default'}, evidence_k={a.evidence_k or 'default'}")
    if a.retrieval_only:  # hop/gate/planner only matter when an answer is generated
        cfgs = [c for c in cfgs if not c.name.startswith("full")]
    results, ingest_info, n_docs = [], {}, 0
    if a.existing_store:
        container = build_container(s, embedder, llm, load=True)
        n_docs = len(container.ingestion.docs)
        runs = [("existing", container, {"existing_store": True, "documents": n_docs, "chunks": len(container.corpus.chunks)})]
    else:
        docs = load_jsonl(Path(a.corpus) / "corpus.jsonl")
        n_docs = len(docs)
        runs = []
        for strat in (["structure", "fixed"] if a.chunking == "both" else [a.chunking]):
            container, info = build_system(docs, strat, embedder, llm, s)
            runs.append((strat, container, info))
    for strat, container, info in runs:
        ingest_info[strat] = info
        print(f"[{strat}] {info}")
        for cfg in cfgs:
            cfg = EvalConfig(cfg.name, cfg.retrieval, cfg.rag, strat, cfg.description)
            try:
                r = run_config(container, questions, cfg, ask=not a.retrieval_only, pause=a.pause, retry_degraded=a.retry_degraded,
                               done=done.get((cfg.name, strat)), on_row=save_row)
            except QuotaExhausted as e:
                print(f"\nSTOPPED: the LLM quota is used up.\n  {e}\nNothing is lost: wait for the quota to reset (or switch LLM_MODEL) and run the same "
                      f"command again with --resume.")
                raise SystemExit(2)
            results.append(r)
            sm = r["summary"]
            print(f"  {cfg.name:24s} R@5={sm['recall@5']} MRR={sm['mrr']} correct={sm.get('answer_correctness')} degraded={sm.get('degraded_rate')}")
            if sm.get("degraded_rate"):
                print(f"  !! {sm['degraded_rate']:.0%} of answers fell back to the offline reader (LLM errors / rate limit): "
                      "re-run with --pause 10 --retry-degraded 2 --resume before using these numbers")
    payload = {"corpus": a.corpus, "embedder": embedder.name, "llm": llm.name, "n_docs": n_docs, "ingest": ingest_info, "results": results}
    (out / f"{a.tag}.json").write_text(json.dumps(payload, indent=1, default=str))
    (out / f"{a.tag}.md").write_text(md_table(results))
    print("\n" + md_table(results))


if __name__ == "__main__":
    main()

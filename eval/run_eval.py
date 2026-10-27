"""CLI: python -m eval.run_eval --corpus data/corpus_small [--llm heuristic|openai] [--embedder hashing|fastembed]
Writes eval/results/<tag>.json and a markdown table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import Settings
from app.container import build_llm
from app.embeddings import build_embedder
from eval.runner import EvalConfig, build_system, load_jsonl, run_config, standard_configs

COLS = ["recall@1", "recall@5", "recall@10", "mrr", "evidence_recall", "answer_correctness", "abstention_recall", "conflict_detection_recall",
        "citation_precision", "citation_recall", "faithfulness", "latency_p50_ms", "avg_llm_calls"]


def md_table(results: list[dict]) -> str:
    head = "| config | " + " | ".join(COLS) + " |\n|" + "---|" * (len(COLS) + 1)
    lines = [head]
    for r in results:
        s = r["summary"]
        lines.append(f"| {r['config']}{'/' + r['chunking'] if r.get('chunking') else ''} | " + " | ".join(str(s.get(c)) for c in COLS) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_small")
    ap.add_argument("--configs", default="all", help="comma list or 'all'")
    ap.add_argument("--chunking", default="structure", help="structure | fixed | both")
    ap.add_argument("--embedder", default="hashing")
    ap.add_argument("--llm", default="heuristic")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="eval/results")
    a = ap.parse_args()

    s = Settings(qdrant_path=":memory:", embedding_provider=a.embedder, llm_provider=a.llm)
    embedder = build_embedder(s)
    llm = build_llm(s)
    real_llm = llm.name != "heuristic-extractive"
    docs = load_jsonl(Path(a.corpus) / "corpus.jsonl")
    questions = load_jsonl(Path(a.corpus) / "questions.jsonl")
    cfgs = standard_configs(real_llm)
    if a.configs != "all":
        want = set(a.configs.split(","))
        cfgs = [c for c in cfgs if c.name in want]
    strategies = ["structure", "fixed"] if a.chunking == "both" else [a.chunking]
    results, ingest_info = [], {}
    for strat in strategies:
        container, info = build_system(docs, strat, embedder, llm, s)
        ingest_info[strat] = info
        print(f"[{strat}] ingested {info}")
        for cfg in cfgs:
            cfg = EvalConfig(cfg.name, cfg.retrieval, cfg.rag, strat, cfg.description)
            r = run_config(container, questions, cfg)
            results.append(r)
            print(f"  {cfg.name:24s} R@5={r['summary']['recall@5']} MRR={r['summary']['mrr']} correct={r['summary']['answer_correctness']}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"corpus": a.corpus, "embedder": embedder.name, "llm": llm.name, "n_docs": len(docs), "ingest": ingest_info, "results": results}
    (out / f"{a.tag}.json").write_text(json.dumps(payload, indent=1, default=str))
    (out / f"{a.tag}.md").write_text(md_table(results))
    print("\n" + md_table(results))


if __name__ == "__main__":
    main()

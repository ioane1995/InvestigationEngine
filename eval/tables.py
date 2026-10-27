"""Render result JSON files into markdown tables:  python -m eval.tables eval/results/heldout_seed11.json [structure|fixed]"""
from __future__ import annotations

import json
import sys

HEAD = [("recall@1", "R@1"), ("recall@5", "R@5"), ("mrr", "MRR"), ("evidence_recall", "Evid.recall"), ("answer_correctness", "Answer correct."),
        ("abstention_recall", "Abstain recall"), ("abstention_precision", "Abstain prec."), ("conflict_detection_recall", "Conflict recall"),
        ("citation_precision", "Cite prec."), ("citation_recall", "Cite recall"), ("faithfulness", "Faithful."), ("latency_p50_ms", "p50 ms")]


def headline(path: str, chunking: str = "structure") -> str:
    d = json.load(open(path))
    rows = [r for r in d["results"] if r["chunking"] == chunking]
    out = ["| config | " + " | ".join(h for _, h in HEAD) + " |", "|---|" + "---:|" * len(HEAD)]
    for r in rows:
        out.append(f"| `{r['config']}` | " + " | ".join(str(r["summary"][k]) for k, _ in HEAD) + " |")
    return "\n".join(out)


def by_type(path: str, config: str, chunking: str) -> str:
    d = json.load(open(path))
    r = next(x for x in d["results"] if x["config"] == config and x["chunking"] == chunking)
    out = ["| question type | n | answer correct. | R@5 (1st stage) | MRR |", "|---|---:|---:|---:|---:|"]
    for t, v in r["summary"]["by_type"].items():
        out.append(f"| {t} | {v['n']} | {v['answer_correctness']} | {v['recall@5']} | {v['mrr']} |")
    return "\n".join(out)


if __name__ == "__main__":
    p = sys.argv[1]
    print(headline(p, sys.argv[2] if len(sys.argv) > 2 else "structure"))

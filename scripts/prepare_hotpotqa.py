"""Build a reproducible corpus + eval set from HotpotQA (distractor setting).

   pip install datasets
   python -m scripts.prepare_hotpotqa --n-questions 6000 --eval-questions 60 --unanswerable 10 --seed 13 --out data/corpus_hotpot

* every Wikipedia paragraph in a question's context becomes one document (deduplicated by title) -> 50k+ docs at ~6k questions
* gold documents = the paragraphs named in `supporting_facts`
* `--unanswerable N`: the gold paragraphs of N sampled questions are REMOVED from the corpus, so the right behaviour is to abstain
Needs internet access to huggingface.co (not available in some sandboxes); `convert_rows` is unit-tested offline."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


def slug(title: str) -> str:
    return "wiki-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]


def convert_rows(rows: Iterable[dict[str, Any]], n_eval: int, n_unanswerable: int, seed: int) -> tuple[list[dict], list[dict]]:
    rows = list(rows)
    rng = random.Random(seed)
    docs: dict[str, dict] = {}
    for r in rows:
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            did = slug(title)
            docs.setdefault(did, {"doc_id": did, "title": title, "text": f"# {title}\n\n" + " ".join(s.strip() for s in sents),
                                  "format": "md", "source": f"hotpotqa/{did}.md", "metadata": {"kind": "wikipedia"}})
    eval_rows = rng.sample(rows, min(n_eval, len(rows)))
    questions = []
    gold_use: dict[str, int] = {}
    for r in rows:
        for t in set(r["supporting_facts"]["title"]):
            gold_use[slug(t)] = gold_use.get(slug(t), 0) + 1
    removable = [r for r in eval_rows if all(gold_use[slug(t)] == 1 for t in set(r["supporting_facts"]["title"]))]
    unans = {r["id"] for r in removable[:n_unanswerable]}
    for r in eval_rows:
        gold = sorted({slug(t) for t in r["supporting_facts"]["title"]})
        if r["id"] in unans:
            for g in gold:
                docs.pop(g, None)
            questions.append({"id": r["id"], "type": "unanswerable", "question": r["question"], "gold_doc_ids": [], "relevant_doc_ids": [],
                              "gold_answers": [], "expected_status": "insufficient_evidence", "notes": "supporting paragraphs removed from corpus"})
        else:
            questions.append({"id": r["id"], "type": "multi_hop", "question": r["question"], "gold_doc_ids": gold, "relevant_doc_ids": gold,
                              "gold_answers": [r["answer"]], "expected_status": "answered", "notes": f"hotpotqa {r.get('type', '')}/{r.get('level', '')}"})
    return list(docs.values()), questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-questions", type=int, default=6000)
    ap.add_argument("--eval-questions", type=int, default=60)
    ap.add_argument("--unanswerable", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/corpus_hotpot")
    a = ap.parse_args()
    from datasets import load_dataset  # lazy: only this script needs it

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=a.split)
    idx = random.Random(a.seed).sample(range(len(ds)), min(a.n_questions, len(ds)))
    docs, qs = convert_rows((ds[i] for i in idx), a.eval_questions, a.unanswerable, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
    (out / "questions.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs), encoding="utf-8")
    print(f"{len(docs)} documents, {len(qs)} eval questions -> {out}")


if __name__ == "__main__":
    main()

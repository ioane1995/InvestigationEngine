"""Generate a reproducible synthetic corpus + ground-truth question set.
   python -m scripts.generate_synthetic --incidents 120 --filler 600 --out data/corpus_small
   python -m scripts.generate_synthetic --incidents 3000 --filler 38000 --out data/corpus_50k"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.synthetic import generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", type=int, default=120)
    ap.add_argument("--filler", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/corpus_small")
    a = ap.parse_args()
    c = generate(a.incidents, a.filler, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in c.docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(out / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in c.questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps(c.stats, indent=2))
    print(json.dumps(c.stats))


if __name__ == "__main__":
    main()

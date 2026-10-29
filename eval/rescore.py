"""Re-score an existing result file with the current (v2) correctness metric WITHOUT calling the LLM again.
   python -m eval.rescore eval/results/real_20b.json data/corpus_heldout/questions.jsonl
Only rows with expected_status == 'answered' whose status matched can change, and only from wrong to right.
Uses the stored answer text (older files kept just 300 characters: such rows are a lower bound)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app.rag.types import Answer, Claim
from eval import metrics as M


def main(result_path: str, questions_path: str) -> None:
    d = json.loads(Path(result_path).read_text(encoding="utf-8"))
    qs = {q["id"]: q for q in (json.loads(l) for l in Path(questions_path).read_text(encoding="utf-8").splitlines() if l.strip())}
    for res in d["results"]:
        rows, flipped = res["rows"], []
        for r in rows:
            q = qs[r["id"]]
            ans = Answer(r["question"], r["got"], r.get("answer", ""), [Claim(t, []) for t in r.get("claims_text", "").split(" | ") if t])
            strict = r.get("correct_strict", r["correct"])
            v2 = M.answer_correct(q, ans)
            if v2 and not strict:
                flipped.append(r["id"])
        n = len(rows)
        strict_acc = sum(r.get("correct_strict", r["correct"]) for r in rows) / n
        print(f"{res['config']}/{res['chunking']}: strict v1 = {strict_acc:.3f}  ->  normalised v2 = {strict_acc + len(flipped) / n:.3f}   "
              f"(rows re-scored as correct: {flipped})")
        for r in rows:
            if r["id"] in flipped:
                print(f"   {r['id']} {r['type']}: gold {qs[r['id']]['gold_answers']}  | answer: {r['answer'][:140]!r}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

"""Evaluation metrics - all implemented here (no eval frameworks)."""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from app.models import Hit
import re

from app.rag.types import Answer
from app.text import normalize_typography


def hit_docs(hits: Sequence[Hit]) -> list[set[str]]:
    return [set(h.doc_ids) for h in hits]


def recall_at_k(ranked: list[set[str]], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    if not gold:
        return math.nan
    found = set().union(*ranked[:k]) if ranked[:k] else set()
    return len(gold & found) / len(gold)


def full_recall_at_k(ranked: list[set[str]], gold: Iterable[str], k: int) -> float:
    r = recall_at_k(ranked, gold, k)
    return math.nan if math.isnan(r) else float(r == 1.0)


def reciprocal_rank(ranked: list[set[str]], gold: Iterable[str]) -> float:
    gold = set(gold)
    if not gold:
        return math.nan
    for i, docs in enumerate(ranked, 1):
        if docs & gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[set[str]], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    if not gold:
        return math.nan
    seen: set[str] = set()
    dcg = 0.0
    for i, docs in enumerate(ranked[:k], 1):
        new = (docs & gold) - seen
        seen |= new
        dcg += len(new) / math.log2(i + 1)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def _text(ans: Answer) -> str:
    parts = [ans.answer, *(c.text for c in ans.claims)]
    for k in ans.conflicts:
        parts.append(k.topic)
        parts += [p.claim for p in k.positions]
    return " ".join(parts).lower()


_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def gold_variants(g: str) -> list[str]:
    """Equivalent surface forms of a gold string. An ISO date may legitimately be written '28 March 2024'."""
    out = [g]
    m = _ISO.match(g)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12:
            name = _MONTHS[mo - 1]
            out += [f"{d} {name} {y}", f"{name} {d}, {y}", f"{name} {d} {y}", f"{d} {name[:3]} {y}", f"{d:02d} {name} {y}"]
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", normalize_typography(s).lower())


def answer_correct(q: dict[str, Any], ans: Answer, strict: bool = False) -> bool:
    """Rule-based correctness. answered -> every gold string appears; insufficient -> status is insufficient;
    conflict -> status is conflicting AND every conflicting value is reported.
    strict=True is the original v1 metric (exact substring only). The default (v2) additionally normalises typographic
    dashes / spaces and accepts equivalent date formats; v2 can only turn a miss into a hit, never the reverse."""
    exp = q["expected_status"]
    if ans.status != exp:
        return False
    if exp == "insufficient_evidence":
        return True
    t = _text(ans) if strict else _norm(_text(ans))
    if strict:
        return all(a.lower() in t for a in q["gold_answers"])
    return all(any(_norm(v) in t for v in gold_variants(a)) for a in q["gold_answers"])


def citation_scores(q: dict[str, Any], ans: Answer) -> tuple[float, float]:
    """(precision, recall). precision = share of cited docs that are relevant; recall = share of gold docs cited."""
    cited = ans.cited_doc_ids()
    rel, gold = set(q["relevant_doc_ids"]), set(q["gold_doc_ids"])
    prec = (sum(1 for d in cited if d in rel) / len(cited)) if cited else math.nan
    rec = (len(gold & set(cited)) / len(gold)) if gold else math.nan
    return prec, rec


def faithfulness(ans: Answer) -> float:
    return (sum(1 for c in ans.claims if c.supported) / len(ans.claims)) if ans.claims else math.nan


def mean(xs: Iterable[float]) -> float:
    v = [x for x in xs if not math.isnan(x)]
    return sum(v) / len(v) if v else math.nan


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return math.nan
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def classify_failure(q: dict[str, Any], ans: Answer, correct: bool) -> str:
    if correct:
        return "ok"
    exp, got = q["expected_status"], ans.status
    gold = set(q["gold_doc_ids"])
    if exp == "insufficient_evidence":
        return "answered_unanswerable"
    if gold and not gold <= set(ans.evidence_doc_ids):
        return "retrieval_miss"
    if exp == "conflicting_evidence":
        return "missed_conflict" if got != exp else "conflict_values_missing"
    if got == "insufficient_evidence":
        return "false_abstention"
    if got == "conflicting_evidence":
        return "false_conflict"
    return "generation_error"

"""Claim verification: every claim must be supported by the evidence it cites.
Numbers and identifiers must appear verbatim; other content words need >= `min_overlap` coverage."""
from __future__ import annotations

import re

from app.models import Hit
from app.rag.types import Claim
from app.text import STOPWORDS, tokenize

_NUM_ID = re.compile(r"\b(?:[A-Za-z]{2,6}-\d{2,6}|\d+(?:[.,]\d+)?)\b")


def support_score(claim: str, evidence_text: str) -> float:
    ev_tokens = set(tokenize(evidence_text))
    ev_lower = evidence_text.lower()
    for tok in _NUM_ID.findall(claim):
        if tok.lower() not in ev_lower:
            return 0.0  # a number / identifier the evidence never mentions is a hallucination signal
    words = [t for t in tokenize(claim) if t not in STOPWORDS and len(t) > 2]
    if not words:
        return 1.0
    return sum(1 for w in words if w in ev_tokens) / len(words)


def verify_claims(claims: list[Claim], evidence: dict[str, Hit], min_overlap: float = 0.6) -> list[Claim]:
    for c in claims:
        c.evidence_ids = [e for e in c.evidence_ids if e in evidence]  # drop invented citation ids
        if not c.evidence_ids:
            c.supported, c.support_score = False, 0.0
            continue
        text = "\n".join(f"{evidence[e].chunk.title}\n{evidence[e].chunk.text}" for e in c.evidence_ids)
        c.support_score = round(support_score(c.text, text), 3)
        c.supported = c.support_score >= min_overlap
    return claims

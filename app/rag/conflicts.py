"""Candidate-conflict detection (algorithmic pre-filter, adjudicated by the LLM).

Idea: structured-looking lines ('Duration: 47 minutes', 'Root cause change: CHG-0123') are extracted from the
evidence. If two *different documents* state different normalised values for the same (entity, field), and the
field is relevant to the question, we report it as a candidate conflict. The LLM receives these candidates as hints
and decides; the offline heuristic LLM trusts them directly."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models import Hit
from app.retrieval.rerank import query_terms
from app.text import identifiers, tokenize

_KV = re.compile(r"^\s*(?:[-*•]\s*)?\**([A-Za-z][A-Za-z0-9 _/()]{1,40}?)\**\s*:\s*(.{1,200}?)\s*$")
_DUR = re.compile(r"^(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|h|days?)\b")
# question words that point at a field without naming it ("how long" -> duration, "when" -> date)
_FOCUS = {"long": {"duration", "length"}, "last": {"duration"}, "when": {"date", "time"}, "start": {"date", "time"}, "who": {"author", "owner"}}
_GENERIC_KEYS = {"note", "notes", "status", "date", "title", "summary", "id", "source", "owner", "team", "author"}


def normalize_value(v: str) -> str:
    s = v.strip().strip("*").lower().rstrip(".")
    m = _DUR.match(s)
    if m:
        n, unit = float(m.group(1)), m.group(2)
        mult = 1 / 60 if unit.startswith("sec") else 60 if unit.startswith(("h", "hr")) else 1440 if unit.startswith("day") else 1
        return f"{round(n * mult, 2)} min"
    return re.sub(r"[^a-z0-9.\- ]+", "", re.sub(r"\s+", " ", s))


@dataclass
class Position:
    value: str
    evidence_ids: list[str]
    doc_ids: list[str]


@dataclass
class CandidateConflict:
    entity: str
    field: str
    positions: list[Position] = field(default_factory=list)

    def describe(self) -> str:
        return f"{self.field} of {self.entity}: " + " vs ".join(f"'{p.value}' ({','.join(p.evidence_ids)})" for p in self.positions)


def find_candidate_conflicts(evidence: dict[str, Hit], question: str, focus_top: int = 3) -> list[CandidateConflict]:
    q_ids = set(identifiers(question))
    q_terms = set(query_terms(question))
    for w in set(tokenize(question)):
        q_terms |= _FOCUS.get(w, set())
    # without identifiers in the question, only entities of the best-ranked evidence count as "the thing asked about"
    focus = {(identifiers(f"{h.chunk.title}\n{h.chunk.text}") or [""])[0] for h in list(evidence.values())[:focus_top]}
    groups: dict[tuple[str, str], dict[str, Position]] = {}
    for eid, hit in evidence.items():
        c = hit.chunk
        ids = identifiers(f"{c.title}\n{c.text}")
        if not ids:
            continue
        entity = ids[0]  # primary entity = first identifier mentioned (title/heading comes first)
        for line in c.text.splitlines():
            m = _KV.match(line)
            if not m:
                continue
            key = re.sub(r"\s+", " ", m.group(1).strip().lower())
            if key in _GENERIC_KEYS:
                continue
            val = normalize_value(m.group(2))
            if not val:
                continue
            pos = groups.setdefault((entity, key), {}).setdefault(val, Position(m.group(2).strip().strip("*"), [], []))
            pos.evidence_ids.append(eid)
            for d in hit.doc_ids:
                if d not in pos.doc_ids:
                    pos.doc_ids.append(d)
    out: list[CandidateConflict] = []
    for (entity, key), by_val in groups.items():
        if len(by_val) < 2:
            continue
        all_docs = {d for p in by_val.values() for d in p.doc_ids}
        # conflicting values must come from different documents, and no document may hold two values
        if len(all_docs) < 2 or any(len(set(p.doc_ids)) == 0 for p in by_val.values()):
            continue
        key_terms = set(tokenize(key))
        overlap = len(key_terms & q_terms) / len(key_terms) if key_terms else 0.0
        # the question must talk about (most of) the field, not just share a generic word such as 'caused'
        relevant = (entity in q_ids) if q_ids else (entity in focus)
        if relevant and overlap >= 0.5:
            out.append(CandidateConflict(entity, key, list(by_val.values())))
    return out

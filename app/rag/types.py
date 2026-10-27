from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

STATUSES = ("answered", "insufficient_evidence", "conflicting_evidence")


@dataclass
class Citation:
    evidence_id: str
    chunk_id: str
    doc_id: str
    title: str
    source: str
    heading: str
    text: str
    also_doc_ids: list[str] = field(default_factory=list)


@dataclass
class Claim:
    text: str
    evidence_ids: list[str]
    supported: bool = True
    support_score: float = 1.0


@dataclass
class ConflictPosition:
    claim: str
    evidence_ids: list[str]


@dataclass
class Conflict:
    topic: str
    positions: list[ConflictPosition]


@dataclass
class Answer:
    question: str
    status: str
    answer: str
    claims: list[Claim] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    missing_info: str = ""
    citations: dict[str, Citation] = field(default_factory=dict)
    sub_queries: list[str] = field(default_factory=list)
    evidence_doc_ids: list[str] = field(default_factory=list)  # every document the pipeline looked at
    trace: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    degraded: bool = False
    cached: bool = False
    usage: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def cited_doc_ids(self) -> list[str]:
        """Documents the answer actually relies on (claims + conflicts), in citation order."""
        ids: list[str] = []
        used = [e for c in self.claims for e in c.evidence_ids] + \
               [e for k in self.conflicts for p in k.positions for e in p.evidence_ids]
        for e in used:
            cit = self.citations.get(e)
            if cit:
                for d in [cit.doc_id, *cit.also_doc_ids]:
                    if d not in ids:
                        ids.append(d)
        return ids


@dataclass
class RagConfig:
    use_planner: bool = True
    hops: int = 1
    hop_phrases: bool = False  # also follow capitalised multi-word entities (useful for Wikipedia-style data)
    max_hop_entities: int = 4
    max_tool_rounds: int = 2
    evidence_k: int = 10
    min_coverage: float = 0.30
    use_gate: bool = True
    max_sub_queries: int = 4
    name: str = "default"
    extra: Optional[dict[str, Any]] = None

"""Own re-ranking + evidence assessment.

FeatureReranker scores each fused candidate with interpretable, IDF-weighted features, then
MMR (Maximal Marginal Relevance) selects a diverse top-k so the LLM is not fed five paraphrases
of the same paragraph.

assess_evidence() measures how well the retrieved text *covers the query*; it drives the
abstention gate in the RAG pipeline (answer / ask LLM / refuse)."""
from __future__ import annotations

from dataclasses import dataclass

from app.models import Hit
from app.retrieval.bm25 import BM25Index
from app.text import STOPWORDS, identifiers, tokenize

DEFAULT_WEIGHTS = {"fused": 0.35, "coverage": 0.25, "id": 0.20, "bigram": 0.08, "proximity": 0.07, "title": 0.05}


# Words that describe the *format of the question* ("which documents confirm this?", "how long did it last?")
# rather than its content. Generic English question phrasing, not corpus-specific.
QUESTION_META = frozenset({"confirm", "evidence", "source", "document", "prove", "support", "cite", "verify", "mention", "list", "tell",
                           "explain", "long", "last", "start", "begin", "happen", "occur", "take", "took", "many", "much", "often"})


def query_terms(query: str) -> list[str]:
    return [t for t in dict.fromkeys(tokenize(query)) if t not in STOPWORDS and t not in QUESTION_META]


def _min_window(tokens: list[str], needed: set[str]) -> int:
    """Length of the smallest token window that contains every needed term (sliding window). 0 if impossible."""
    if not needed:
        return 0
    need = len(needed)
    counts: dict[str, int] = {}
    have = 0
    best = 0
    left = 0
    for right, tok in enumerate(tokens):
        if tok in needed:
            counts[tok] = counts.get(tok, 0) + 1
            if counts[tok] == 1:
                have += 1
        while have == need:
            w = right - left + 1
            best = w if best == 0 or w < best else best
            lt = tokens[left]
            if lt in needed:
                counts[lt] -= 1
                if counts[lt] == 0:
                    have -= 1
            left += 1
    return best


class FeatureReranker:
    def __init__(self, index: BM25Index, weights: dict[str, float] | None = None, mmr_lambda: float = 0.8):
        self.index = index
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.mmr_lambda = mmr_lambda

    def features(self, query: str, hit: Hit, fused_norm: float) -> dict[str, float]:
        q_terms = query_terms(query)
        q_ids = set(identifiers(query))
        body = tokenize(hit.chunk.text)
        title_toks = set(tokenize(f"{hit.chunk.title} {hit.chunk.heading}"))
        tok_set = set(body) | title_toks
        idf = {t: self.index.idf(t) for t in q_terms}
        tot = sum(idf.values()) or 1.0
        matched = [t for t in q_terms if t in tok_set]
        coverage = sum(idf[t] for t in matched) / tot
        id_match = (sum(1 for i in q_ids if i in tok_set) / len(q_ids)) if q_ids else 0.0
        q_seq = [t for t in tokenize(query)]
        bigrams = set(zip(q_seq, q_seq[1:]))
        body_bi = set(zip(body, body[1:]))
        bigram = (len(bigrams & body_bi) / len(bigrams)) if bigrams else 0.0
        need = {t for t in matched if t in set(body)}
        win = _min_window(body, need)
        proximity = (len(need) / win) if win else 0.0
        title = (sum(idf[t] for t in q_terms if t in title_toks) / tot) if q_terms else 0.0
        return {"fused": fused_norm, "coverage": coverage, "id": id_match, "bigram": bigram,
                "proximity": proximity, "title": title}

    def rerank(self, query: str, hits: list[Hit], top_k: int, fused: dict[str, float] | None = None) -> list[Hit]:
        if not hits:
            return []
        mx = max(h.score for h in hits) or 1.0
        for h in hits:
            f = self.features(query, h, h.score / mx)
            score = sum(self.w[k] * v for k, v in f.items())
            h.components.update({f"rr_{k}": round(v, 4) for k, v in f.items()})
            h.components["fused"] = round(h.score, 6)
            h.score = score
        return self._mmr(hits, top_k)

    def _mmr(self, hits: list[Hit], top_k: int) -> list[Hit]:
        pool = sorted(hits, key=lambda h: h.score, reverse=True)
        sets = {h.chunk.chunk_id: set(tokenize(h.chunk.text)) for h in pool}
        chosen: list[Hit] = []
        while pool and len(chosen) < top_k:
            best, best_val = None, -1e9
            for h in pool:
                s = sets[h.chunk.chunk_id]
                sim = 0.0
                for c in chosen:
                    o = sets[c.chunk.chunk_id]
                    u = len(s | o)
                    sim = max(sim, (len(s & o) / u) if u else 0.0)
                val = self.mmr_lambda * h.score - (1 - self.mmr_lambda) * sim
                if val > best_val:
                    best, best_val = h, val
            chosen.append(best)  # type: ignore[arg-type]
            pool.remove(best)  # type: ignore[arg-type]
        return chosen


@dataclass
class EvidenceAssessment:
    coverage: float  # IDF-weighted share of query terms found in the top evidence
    missing_ids: list[str]  # identifiers in the question that appear in NO retrieved text
    missing_terms: list[str]
    ok: bool


def assess_evidence(query: str, hits: list[Hit], index: BM25Index, min_coverage: float, top_n: int = 5) -> EvidenceAssessment:
    q_terms = query_terms(query)
    if not q_terms or not hits:
        return EvidenceAssessment(0.0, identifiers(query), q_terms, False)
    union: set[str] = set()
    for h in hits[:top_n]:
        union |= set(tokenize(f"{h.chunk.title} {h.chunk.heading} {h.chunk.text}"))
    idf = {t: index.idf(t) for t in q_terms}
    tot = sum(idf.values()) or 1.0
    cov = sum(idf[t] for t in q_terms if t in union) / tot
    miss_ids = [i for i in identifiers(query) if i not in union]
    miss_terms = sorted((t for t in q_terms if t not in union), key=lambda t: -idf[t])[:5]
    return EvidenceAssessment(cov, miss_ids, miss_terms, ok=(cov >= min_coverage and not miss_ids))

"""Rank-fusion algorithms (own implementations)."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60,
                           weights: Sequence[float] | None = None) -> dict[str, float]:
    """RRF (Cormack et al. 2009): score(d) = sum_i w_i / (k + rank_i(d)). Rank-based, so it needs no
    score calibration between cosine similarities and BM25 values."""
    weights = weights or [1.0] * len(rankings)
    out: dict[str, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for r, item in enumerate(ranking, start=1):
            out[item] += w / (k + r)
    return dict(out)


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def weighted_score_fusion(score_lists: Sequence[dict[str, float]], weights: Sequence[float] | None = None) -> dict[str, float]:
    """Alternative to RRF: min-max normalise every score list, then take a weighted sum.
    Compared against RRF in the evaluation."""
    weights = weights or [1.0] * len(score_lists)
    out: dict[str, float] = defaultdict(float)
    for w, s in zip(weights, score_lists):
        for k, v in minmax(s).items():
            out[k] += w * v
    return dict(out)

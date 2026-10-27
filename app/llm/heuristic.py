"""Offline, deterministic 'LLM' used for tests, CI and baseline evaluation (no network, no cost).
It performs *extractive* question answering over the evidence blocks it receives and obeys the same
tool-calling protocol as a real model. It cannot reason or paraphrase - it is a lower-bound baseline and
the graceful-degradation fallback when the real LLM is unavailable."""
from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Optional

from app.llm.base import LLMResponse, ToolCall, Usage
from app.rag.conflicts import _FOCUS
from app.retrieval.rerank import query_terms
from app.text import identifiers, split_sentences, tokenize

_EV = re.compile(r'<evidence id="(E\d+)"([^>]*)>\n(.*?)\n</evidence>', re.S)
_Q = re.compile(r"<question>(.*?)</question>", re.S)
_CONF = re.compile(r"<candidate_conflicts>(.*?)</candidate_conflicts>", re.S)
_MISS = re.compile(r"<missing_terms>(.*?)</missing_terms>", re.S)


class HeuristicLLM:
    name = "heuristic-extractive"

    def __init__(self, idf: Optional[Callable[[str], float]] = None):
        self.idf = idf  # optional corpus IDF so rare words weigh more than common ones

    def chat(self, messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None,
             tool_choice: Any = None, temperature: float = 0.0, max_tokens: int = 1500) -> LLMResponse:
        names = {t["function"]["name"] for t in (tools or [])}
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "submit_plan" in names and "submit_answer" not in names:
            q = (_Q.search(user).group(1) if _Q.search(user) else user).strip()
            return LLMResponse(None, [ToolCall("h1", "submit_plan", {"sub_queries": [q]})], Usage(len(user) // 4, 20))
        full = "\n".join(m["content"] for m in messages if m["role"] in ("user", "tool") and isinstance(m.get("content"), str))
        args = self._answer(full)
        return LLMResponse(None, [ToolCall("h2", "submit_answer", args)], Usage(len(full) // 4, len(json.dumps(args)) // 4))

    def _answer(self, text: str) -> dict[str, Any]:
        qm = _Q.search(text)
        question = qm.group(1).strip() if qm else ""
        evidence = {eid: body for eid, _attrs, body in _EV.findall(text)}
        titles = {eid: (re.search(r'title="([^"]*)"', attrs).group(1) if re.search(r'title="([^"]*)"', attrs) else "") for eid, attrs, _ in _EV.findall(text)}
        via = {eid: ("hop" if 'via="hop"' in attrs or 'via="tool"' in attrs else "query") for eid, attrs, _ in _EV.findall(text)}
        q_terms = set(query_terms(question))
        for w in set(tokenize(question)):
            q_terms |= _FOCUS.get(w, set())
        q_ids = set(identifiers(question))
        id_parts = {p for i in q_ids for p in tokenize(i)}  # 'chg-0414' -> chg-0414, chg, 0414: none of these is content
        conf = []
        if (cm := _CONF.search(text)):
            try:
                conf = json.loads(cm.group(1))
            except json.JSONDecodeError:
                conf = []
        if conf:
            conflicts = [{"topic": c["topic"], "positions": [{"claim": p["claim"], "evidence_ids": p["evidence_ids"]} for p in c["positions"]]}
                         for c in conf]
            summary = "; ".join(f"{c['topic']}: " + " vs ".join(f"{p['claim']} [{','.join(p['evidence_ids'])}]" for p in c["positions"])
                                for c in conf)
            claims = [{"text": p["claim"], "evidence_ids": p["evidence_ids"]} for c in conf for p in c["positions"]]
            return {"status": "conflicting_evidence", "answer": f"Sources disagree - {summary}", "claims": claims, "conflicts": conflicts}
        if (mm := _MISS.search(text)) and len([t for t in mm.group(1).split(",") if t.strip()]) >= 2:
            # baseline rule: two or more content words of the question occur nowhere in the corpus
            return {"status": "insufficient_evidence", "answer": "The documents do not contain this information.",
                    "claims": [], "missing_info": f"No document mentions: {mm.group(1).strip()}"}
        scored: list[tuple[float, str, str]] = []
        for pos, (eid, body) in enumerate(evidence.items()):
            btoks = set(tokenize(body)) | set(tokenize(titles[eid]))  # the title carries context a chunk may lack
            # when the question names entities, only read blocks about them (or blocks reached by following a link)
            if q_ids and not (q_ids & btoks) and via[eid] == "query":
                continue
            block_w = (2.0 if q_ids & btoks else 1.0) / (1.0 + 0.1 * pos)
            for sent in split_sentences(body):
                toks = set(tokenize(sent))
                content = {t for t in q_terms if t not in id_parts and "-" not in t}
                matched = content & toks
                if not matched:  # a sentence that only repeats the identifier tells us nothing
                    continue
                w = sum((self.idf(t) if self.idf else 1.0) for t in matched)
                scored.append((w * block_w / math.sqrt(len(toks)), eid, sent))
        scored.sort(key=lambda x: -x[0])
        picked: list[tuple[str, str]] = []
        used_ev: dict[str, int] = {}
        for _, eid, sent in scored:  # at most 2 sentences per evidence block, 5 overall
            if used_ev.get(eid, 0) < 2 and len(picked) < 5:
                picked.append((eid, sent))
                used_ev[eid] = used_ev.get(eid, 0) + 1
        if not picked:
            return {"status": "insufficient_evidence", "answer": "No relevant evidence found.", "claims": [],
                    "missing_info": "retrieval returned nothing relevant"}
        return {"status": "answered", "answer": " ".join(s for _, s in picked),
                "claims": [{"text": s, "evidence_ids": [e]} for e, s in picked]}

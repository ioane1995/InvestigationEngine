"""The investigation pipeline (agentic RAG).

  question
    1. PLAN        LLM tool call `submit_plan` -> sub-queries (falls back to the raw question on any LLM error)
    2. RETRIEVE    hybrid search per sub-query -> own RRF merge across sub-queries
    3. HOP         follow identifiers found in the best hits (e.g. INC-0012 -> CHG-0345) with extra searches
    4. GATE        evidence assessment: refuse *before* calling the LLM if the question's key terms/ids are absent
    5. REASON      LLM with tools (search_documents / get_document / submit_answer), structured final answer
    6. VERIFY      every claim is checked against the evidence it cites; unsupported claims are flagged,
                   fully unsupported answers are downgraded to `insufficient_evidence`
    7. CONFLICTS   algorithmic candidate-conflict detection is attached even if the LLM missed it
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import replace
from typing import Any, Callable, Optional

from app.llm.base import LLM, LLMError, Usage
from app.llm.heuristic import HeuristicLLM
from app.models import Hit
from app.rag import prompts
from app.rag.conflicts import CandidateConflict, find_candidate_conflicts
from app.rag.safety import guard_output, neutralize, quarantine
from app.rag.types import STATUSES, Answer, Citation, Claim, Conflict, ConflictPosition, RagConfig
from app.rag.verify import verify_claims
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever, RetrievalConfig
from app.retrieval.rerank import assess_evidence, query_terms
from app.text import identifiers, normalize_typography

log = logging.getLogger(__name__)
_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b")


class EvidenceBook:
    """Per-request registry: stable short ids (E1, E2...) -> hits. Short ids keep prompts small and make
    hallucinated citations trivially detectable (an id that is not in the book)."""

    def __init__(self) -> None:
        self.by_id: dict[str, Hit] = {}
        self._chunk_to_id: dict[str, str] = {}

    def add(self, hit: Hit) -> tuple[str, bool]:
        cid = hit.chunk.chunk_id
        if cid in self._chunk_to_id:
            return self._chunk_to_id[cid], False
        eid = f"E{len(self.by_id) + 1}"
        self.by_id[eid] = hit
        self._chunk_to_id[cid] = eid
        return eid, True

    def render(self, eids: list[str]) -> tuple[str, list[str]]:
        blocks, flagged = [], []
        for e in eids:
            c = self.by_id[e].chunk
            body, removed = quarantine(c.text)  # injected sentences never reach the model; the UI still shows the original
            if removed:
                flagged.append(e)
            sec = f' section="{neutralize(c.heading)[:120]}"' if c.heading else ""
            sec += f' via="{self.by_id[e].via}"' if self.by_id[e].via != "query" else ""
            blocks.append(f'<evidence id="{e}" doc_id="{neutralize(c.doc_id)}" title="{neutralize(c.title)[:120]}"{sec} '
                          f'source="{neutralize(c.source)}">\n{neutralize(body)}\n</evidence>')
        return "\n".join(blocks), flagged


class InvestigationEngine:
    def __init__(self, retriever: HybridRetriever, llm: LLM, price_in_per_m: float = 0.0, price_out_per_m: float = 0.0):
        self.retriever, self.llm = retriever, llm
        self.fallback = HeuristicLLM(idf=retriever.corpus.bm25.idf)
        self.price_in, self.price_out = price_in_per_m, price_out_per_m

    # ------------------------------------------------------------------ public
    def ask(self, question: str, cfg: Optional[RagConfig] = None, rcfg: Optional[RetrievalConfig] = None,
            on_event: Optional[Callable[[dict[str, Any]], None]] = None) -> Answer:
        cfg, rcfg = cfg or RagConfig(), rcfg or RetrievalConfig()
        t0 = time.perf_counter()
        usage = Usage()
        calls = 0
        trace: list[dict[str, Any]] = []
        warnings: list[str] = []
        timings: dict[str, float] = {}

        def emit(step: str, **kw: Any) -> None:
            ev = {"step": step, **kw}
            trace.append(ev)
            if on_event:
                try:
                    on_event(ev)
                except Exception:  # a broken listener must never break the pipeline
                    log.exception("event listener failed")

        def llm_call(llm: LLM, **kw: Any):
            nonlocal calls
            calls += 1
            r = llm.chat(**kw)
            usage.add(r.usage)
            return r

        # 1 ---- plan
        sub_queries = [question]
        t = time.perf_counter()
        if cfg.use_planner:
            try:
                r = llm_call(self.llm, messages=[
                    {"role": "system", "content": prompts.PLANNER_SYSTEM.format(max_q=cfg.max_sub_queries)},
                    {"role": "user", "content": f"<question>{neutralize(question)}</question>"}],
                    tools=[prompts.SUBMIT_PLAN], tool_choice={"type": "function", "function": {"name": "submit_plan"}}, max_tokens=300)
                plan = next((c for c in r.tool_calls if c.name == "submit_plan" and not c.error), None)
                if plan:
                    qs = [q.strip() for q in plan.arguments.get("sub_queries", []) if isinstance(q, str) and q.strip()]
                    sub_queries = list(dict.fromkeys([question, *qs]))[: cfg.max_sub_queries + 1]
                else:
                    warnings.append("planner returned no valid plan; using the original question")
            except LLMError as e:
                warnings.append(f"planner unavailable ({type(e).__name__}: {str(e)[:160]}); using the original question")
        timings["plan_ms"] = round((time.perf_counter() - t) * 1000, 1)
        emit("plan", sub_queries=sub_queries)

        # 2 ---- retrieve (+ merge across sub-queries)
        t = time.perf_counter()
        per_query: list[list[Hit]] = [self.retriever.search(q, rcfg) for q in sub_queries]
        merged = self._merge(per_query)
        main = merged[: cfg.evidence_k]
        if any("dense_unavailable" in h.components for hs in per_query for h in hs):
            warnings.append("semantic search unavailable (embedding service error); results use keyword search only")
        emit("retrieve", per_query=[{"query": q, "hits": [h.chunk.chunk_id for h in hs]} for q, hs in zip(sub_queries, per_query)])

        # 3 ---- hop expansion
        hop_hits: list[Hit] = []
        known = {h.chunk.chunk_id for h in main}
        for _ in range(cfg.hops):
            ents = self._bridge_entities(main[:3] + hop_hits[:2], question, cfg)
            new = []
            for ent in ents[: cfg.max_hop_entities]:
                for h in self.retriever.search(ent, replace(rcfg, top_k=2)):
                    if h.chunk.chunk_id not in known:
                        h.via = "hop"
                        known.add(h.chunk.chunk_id)
                        new.append(h)
            emit("hop", entities=ents[: cfg.max_hop_entities], added=[h.chunk.chunk_id for h in new])
            if not new:
                break
            hop_hits.extend(new)
        book = EvidenceBook()
        for h in main + hop_hits:
            book.add(h)
        timings["retrieve_ms"] = round((time.perf_counter() - t) * 1000, 1)

        # 4 ---- gate
        ordered = [book.by_id[e] for e in book.by_id]
        index = self.retriever.corpus.bm25
        assess = assess_evidence(question, ordered, index, cfg.min_coverage)
        emit("assess", coverage=round(assess.coverage, 3), missing_ids=assess.missing_ids, missing_terms=assess.missing_terms, ok=assess.ok)
        ans: Optional[Answer] = None
        if cfg.use_gate and not assess.ok:
            why = (f"identifiers not found in any retrieved document: {', '.join(assess.missing_ids)}" if assess.missing_ids
                   else f"key terms of the question are not covered by the documents (coverage {assess.coverage:.2f}): {', '.join(assess.missing_terms)}")
            ans = Answer(question, "insufficient_evidence", "The available documents do not contain enough information to answer this question.",
                         missing_info=why)
        else:
            conflicts = find_candidate_conflicts(book.by_id, question)
            hard_missing = [t_ for t_ in assess.missing_terms if index.df(t_) == 0 and t_ in set(query_terms(question))]
            # 5 ---- reason
            t = time.perf_counter()
            ans = self._reason(question, book, conflicts, hard_missing, cfg, rcfg, llm_call, emit, warnings)
            timings["reason_ms"] = round((time.perf_counter() - t) * 1000, 1)

        # 6/7 ---- finalize
        ans.sub_queries = sub_queries
        ans.evidence_doc_ids = list(dict.fromkeys(d for h in book.by_id.values() for d in h.doc_ids))
        ans.trace, ans.warnings = trace, warnings + ans.warnings
        ans.citations = self._citations(ans, book)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        ans.timings_ms = timings
        ans.usage = {"llm_calls": calls, "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                     "cost_usd": round(usage.prompt_tokens * self.price_in / 1e6 + usage.completion_tokens * self.price_out / 1e6, 6)}
        emit("done", status=ans.status)
        return ans

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _merge(per_query: list[list[Hit]]) -> list[Hit]:
        if len(per_query) == 1:
            return list(per_query[0])
        best: dict[str, Hit] = {}
        for hs in per_query:
            for h in hs:
                b = best.get(h.chunk.chunk_id)
                if b is None or h.score > b.score:
                    best[h.chunk.chunk_id] = h
        scores = reciprocal_rank_fusion([[h.chunk.chunk_id for h in hs] for hs in per_query], k=10,
                                        weights=[1.5] + [1.0] * (len(per_query) - 1))
        return [best[c] for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

    @staticmethod
    def _bridge_entities(hits: list[Hit], question: str, cfg: RagConfig) -> list[str]:
        """Entities mentioned by the best hits but absent from the question = the 'bridge' to the next hop."""
        q_low = question.lower()
        ents: list[str] = []
        for h in hits:
            text = f"{h.chunk.title}\n{h.chunk.text}"
            for i in identifiers(text):
                if i not in q_low and i not in ents:
                    ents.append(i)
            if cfg.hop_phrases:
                for m in _PHRASE_RE.findall(text):
                    if m.lower() not in q_low and m not in ents:
                        ents.append(m)
        return ents

    def _reason(self, question: str, book: EvidenceBook, conflicts: list[CandidateConflict], missing: list[str],
                cfg: RagConfig, rcfg: RetrievalConfig, llm_call, emit, warnings: list[str]) -> Answer:
        eids = list(book.by_id)
        rendered, flagged = book.render(eids)
        if flagged:
            warnings.append(f"possible prompt-injection text in evidence {flagged}; instruction-like sentences were removed from the model's view")
            emit("injection_flag", evidence_ids=flagged)
        user = f"<question>{neutralize(question)}</question>\n{rendered}"
        if conflicts:
            hint = [{"topic": f"{c.field} of {c.entity.upper()}", "positions": [
                {"claim": f"{c.field} = {p.value}", "evidence_ids": p.evidence_ids} for p in c.positions]} for c in conflicts]
            user += f"\n<candidate_conflicts>{json.dumps(hint)}</candidate_conflicts>"
        if missing:
            user += f"\n<missing_terms>{', '.join(missing)}</missing_terms>"
        messages: list[dict[str, Any]] = [{"role": "system", "content": prompts.ANSWER_SYSTEM}, {"role": "user", "content": user}]
        sent: set[str] = set(eids)
        sub: Optional[dict[str, Any]] = None
        try:
            for rnd in range(cfg.max_tool_rounds + 1):
                last = rnd == cfg.max_tool_rounds
                choice: Any = {"type": "function", "function": {"name": "submit_answer"}} if last else "auto"
                r = llm_call(self.llm, messages=messages, tools=prompts.TOOLS, tool_choice=choice)
                if not r.tool_calls:
                    messages += [{"role": "assistant", "content": r.content or ""},
                                 {"role": "user", "content": "Call the submit_answer tool now with your final structured answer."}]
                    continue
                messages.append({"role": "assistant", "content": r.content, "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in r.tool_calls]})
                results: list[dict[str, Any]] = []
                for c in r.tool_calls:
                    if c.error:
                        out = f"ERROR: {c.error}. Re-send the call with valid JSON arguments."
                    elif c.name == "submit_answer":
                        try:
                            self._parse_submission(c.arguments)
                            sub = c.arguments
                            break
                        except ValueError as e:
                            out = f"ERROR: invalid submit_answer arguments: {e}"
                    elif c.name == "search_documents":
                        out = self._tool_search(c.arguments, book, sent, rcfg)
                        emit("tool_call", tool="search_documents", args=c.arguments)
                    elif c.name == "get_document":
                        out = self._tool_get_document(c.arguments, book, sent)
                        emit("tool_call", tool="get_document", args=c.arguments)
                    else:
                        out = f"ERROR: unknown tool '{c.name}'"
                    results.append({"role": "tool", "tool_call_id": c.id, "content": out})
                if sub is not None:
                    break
                messages += results
        except LLMError as e:
            warnings.append(f"LLM unavailable ({type(e).__name__}: {e}); returned a degraded extractive answer")
            return self._degraded(question, book, conflicts, missing, warnings)
        if sub is None:
            warnings.append("model did not produce a valid structured answer; returned a degraded extractive answer")
            return self._degraded(question, book, conflicts, missing, warnings)
        return self._finalize(question, sub, book, conflicts, warnings, degraded=False)

    def _tool_search(self, args: dict[str, Any], book: EvidenceBook, sent: set[str], rcfg: RetrievalConfig) -> str:
        q = str(args.get("query", "")).strip()
        if not q:
            return "ERROR: empty query"
        k = max(1, min(int(args.get("top_k", 4) or 4), 8))
        new: list[str] = []
        for h in self.retriever.search(q, replace(rcfg, top_k=k)):
            h.via = "tool"
            eid, added = book.add(h)
            if added or eid not in sent:
                new.append(eid)
        if not new:
            return "No new evidence found for this query."
        text, _ = book.render(new)
        sent.update(new)
        return text

    def _tool_get_document(self, args: dict[str, Any], book: EvidenceBook, sent: set[str]) -> str:
        chunks = self.retriever.corpus.doc_chunks(str(args.get("doc_id", "")))[:4]
        if not chunks:
            return "ERROR: unknown doc_id"
        new = []
        for c in chunks:
            eid, _ = book.add(Hit(chunk=c, score=0.0, via="tool"))
            if eid not in sent:
                new.append(eid)
        sent.update(new)
        return book.render(new)[0] if new else "Those chunks were already provided."

    @staticmethod
    def _parse_submission(a: dict[str, Any]) -> None:
        if a.get("status") not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if not isinstance(a.get("answer"), str) or not a["answer"].strip():
            raise ValueError("answer must be a non-empty string")
        if not isinstance(a.get("claims", []), list):
            raise ValueError("claims must be a list")

    def _finalize(self, question: str, a: dict[str, Any], book: EvidenceBook, cand: list[CandidateConflict],
                  warnings: list[str], degraded: bool) -> Answer:
        nt = normalize_typography
        a = {**a, "answer": nt(str(a.get("answer", ""))), "missing_info": nt(str(a.get("missing_info", "")))}
        claims = [Claim(nt(str(c.get("text", ""))), [str(e) for e in c.get("evidence_ids", []) if isinstance(e, str)])
                  for c in a.get("claims", []) if isinstance(c, dict) and c.get("text")]
        verify_claims(claims, book.by_id)
        conflicts = [Conflict(nt(str(k.get("topic", ""))), [ConflictPosition(nt(str(p.get("claim", ""))), [e for e in p.get("evidence_ids", []) if e in book.by_id])
                                                        for p in k.get("positions", []) if isinstance(p, dict)])
                     for k in a.get("conflicts", []) if isinstance(k, dict)]
        status = a["status"]
        if status == "answered":
            if claims and not any(c.supported for c in claims):
                warnings.append("no claim could be verified against the cited evidence; answer withheld")
                status = "insufficient_evidence"
            elif not claims:
                warnings.append("answer carries no citations")
            if cand and not conflicts:
                warnings.append("sources may disagree: " + "; ".join(c.describe() for c in cand))
        if status == "conflicting_evidence" and not conflicts and cand:
            conflicts = [Conflict(f"{c.field} of {c.entity.upper()}", [ConflictPosition(f"{c.field} = {p.value}", p.evidence_ids) for p in c.positions])
                         for c in cand]
        answer_text = a["answer"].strip()
        evidence_blob = "\n".join(f"{h.chunk.title}\n{h.chunk.text}" for h in book.by_id.values())
        answer_text, guard_warnings = guard_output(answer_text, evidence_blob)
        warnings.extend(guard_warnings)
        for c in claims:
            c.text, _ = guard_output(c.text, evidence_blob)
        if status == "insufficient_evidence" and a["status"] == "answered":
            answer_text = "The documents do not contain verifiable support for an answer."
        unsup = [c.text for c in claims if not c.supported]
        if unsup and status == "answered":
            warnings.append(f"{len(unsup)} claim(s) could not be verified against their citations")
        return Answer(question, status, answer_text, claims, conflicts, str(a.get("missing_info", "")), degraded=degraded, warnings=[])

    def _degraded(self, question: str, book: EvidenceBook, cand: list[CandidateConflict], missing: list[str], warnings: list[str]) -> Answer:
        rendered, _ = book.render(list(book.by_id))
        user = f"<question>{neutralize(question)}</question>\n{rendered}"
        if cand:
            user += "\n<candidate_conflicts>" + json.dumps([{"topic": f"{c.field} of {c.entity.upper()}", "positions": [
                {"claim": f"{c.field} = {p.value}", "evidence_ids": p.evidence_ids} for p in c.positions]} for c in cand]) + "</candidate_conflicts>"
        if missing:
            user += f"\n<missing_terms>{', '.join(missing)}</missing_terms>"
        r = self.fallback.chat([{"role": "user", "content": user}], tools=prompts.TOOLS)
        return self._finalize(question, r.tool_calls[0].arguments, book, cand, warnings, degraded=True)

    @staticmethod
    def _citations(ans: Answer, book: EvidenceBook) -> dict[str, Citation]:
        used: list[str] = []
        for c in ans.claims:
            used += c.evidence_ids
        for k in ans.conflicts:
            for p in k.positions:
                used += p.evidence_ids
        out: dict[str, Citation] = {}
        for e in dict.fromkeys(used):
            h = book.by_id.get(e)
            if h:
                c = h.chunk
                out[e] = Citation(e, c.chunk_id, c.doc_id, c.title, c.source, c.heading, c.text, h.also_doc_ids)
        return out

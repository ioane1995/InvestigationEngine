import json
import math

from eval import metrics as M
from eval.runner import build_system, run_config, standard_configs
from eval.synthetic import generate
from scripts.prepare_hotpotqa import convert_rows


def test_ranking_metrics():
    ranked = [{"x"}, {"g1", "dup"}, {"y"}, {"g2"}]
    assert M.recall_at_k(ranked, ["g1", "g2"], 2) == 0.5 and M.recall_at_k(ranked, ["g1", "g2"], 4) == 1.0
    assert M.reciprocal_rank(ranked, ["g2", "g1"]) == 0.5 and M.reciprocal_rank(ranked, ["zzz"]) == 0.0
    assert math.isnan(M.recall_at_k(ranked, [], 3))
    assert 0 < M.ndcg_at_k(ranked, ["g1", "g2"], 4) < 1 and M.ndcg_at_k([{"g1"}, {"g2"}], ["g1", "g2"], 2) == 1.0
    assert M.full_recall_at_k(ranked, ["g1", "g2"], 3) == 0.0


def test_generator_is_deterministic_and_covers_all_question_types():
    a, b = generate(40, 100, seed=5), generate(40, 100, seed=5)
    assert json.dumps(a.docs) == json.dumps(b.docs) and a.questions == b.questions
    assert {q["type"] for q in a.questions} == {"simple", "multi_hop", "multi_hop_nl", "unanswerable", "conflict"}
    assert a.stats["exact_duplicates"] > 0 and generate(40, 100, seed=6).docs != a.docs
    ids = {d["doc_id"] for d in a.docs}
    assert all(set(q["gold_doc_ids"]) <= ids for q in a.questions)  # gold labels always point at existing documents


def test_end_to_end_eval_runs_and_produces_sane_numbers():
    corp = generate(40, 120, seed=5)
    c, info = build_system(corp.docs, "structure")
    assert info["statuses"]["duplicate"] == corp.stats["exact_duplicates"]
    cfgs = {x.name: x for x in standard_configs(False)}
    res = run_config(c, corp.questions, cfgs["full"])
    s = res["summary"]
    assert s["n_questions"] == len(corp.questions) and 0 <= s["recall@5"] <= 1 and s["faithfulness"] == 1.0
    assert s["abstention_recall"] > 0 and s["conflict_detection_recall"] > 0.5 and s["answer_correctness"] > 0.4
    base = run_config(c, corp.questions, cfgs["dense_only"])["summary"]
    assert s["answer_correctness"] >= base["answer_correctness"]


def test_hotpot_conversion_and_unanswerable_removal():
    def row(i, titles, gold):
        return {"id": str(i), "question": f"q{i}?", "answer": f"a{i}", "type": "bridge", "level": "hard",
                "supporting_facts": {"title": gold, "sent_id": [0] * len(gold)}, "context": {"title": titles, "sentences": [[f"{t} text."] for t in titles]}}
    rows = [row(i, [f"T{i}a", f"T{i}b", "Shared"], [f"T{i}a", f"T{i}b"]) for i in range(6)]
    docs, qs = convert_rows(rows, n_eval=6, n_unanswerable=2, seed=1)
    ids = {d["doc_id"] for d in docs}
    un = [q for q in qs if q["type"] == "unanswerable"]
    assert len(un) == 2 and all(q["expected_status"] == "insufficient_evidence" for q in un)
    assert all(q["gold_doc_ids"] and set(q["gold_doc_ids"]) <= ids for q in qs if q["type"] != "unanswerable")
    assert "wiki-shared" in ids  # shared paragraphs are de-duplicated into one document


def test_retrieval_only_mode_never_calls_the_llm():
    from app.rag.types import RagConfig
    corp = generate(30, 60, seed=5)
    c, _ = build_system(corp.docs, "structure")
    c.engine.ask = lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM path must not run"))  # type: ignore[assignment]
    res = run_config(c, corp.questions, standard_configs(False)[3], ask=False)
    assert res["summary"]["retrieval_only"] and res["summary"]["recall@5"] > 0.3 and "correct" not in res["rows"][0]


def test_degraded_answers_are_retried_not_silently_scored(monkeypatch):
    import eval.runner as R
    from app.rag.types import Answer
    corp = generate(30, 60, seed=5)
    c, _ = build_system(corp.docs, "structure")
    real = c.engine.ask
    state = {"n": 0}
    def flaky(q, rag, ret, **k):
        a = real(q, rag, ret)
        state["n"] += 1
        a.degraded = state["n"] % 2 == 1  # every first attempt "falls back"
        return a
    c.engine.ask = flaky  # type: ignore[assignment]
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    res = run_config(c, corp.questions[:6], standard_configs(False)[-1], retry_degraded=1)
    assert res["summary"]["degraded_rate"] == 0.0 and state["n"] == 12
    state["n"] = 0
    res = run_config(c, corp.questions[:6], standard_configs(False)[-1], retry_degraded=0)
    assert res["summary"]["degraded_rate"] == 0.5


def test_resume_skips_finished_questions():
    corp = generate(30, 60, seed=5)
    c, _ = build_system(corp.docs, "structure")
    cfg = standard_configs(False)[-1]
    first = run_config(c, corp.questions[:5], cfg)
    done = {r["id"]: r for r in first["rows"]}
    calls = {"n": 0}
    real = c.engine.ask
    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    c.engine.ask = counting  # type: ignore[assignment]
    res = run_config(c, corp.questions[:8], cfg, done=done)
    assert calls["n"] == 3 and len(res["rows"]) == 8


def test_quota_exhaustion_stops_the_evaluation_and_sample_is_stratified():
    import pytest
    from eval.runner import QuotaExhausted, stratified_sample
    corp = generate(40, 80, seed=5)
    s = stratified_sample(corp.questions, 12)
    assert {q["type"] for q in s} == {q["type"] for q in corp.questions} and 12 <= len(s) <= 16 and s == stratified_sample(corp.questions, 12)
    c, _ = build_system(corp.docs, "structure")
    from app.rag.types import Answer
    c.engine.ask = lambda *a, **k: Answer("q", "insufficient_evidence", "x", degraded=True, warnings=["LLM unavailable (LLMQuotaExhausted: quota exhausted)"])  # type: ignore
    with pytest.raises(QuotaExhausted):
        run_config(c, corp.questions[:3], standard_configs(False)[-1], retry_degraded=2)


def test_metric_v2_accepts_typographic_hyphens_and_equivalent_dates_but_stays_strict_otherwise():
    from app.rag.types import Answer, Claim
    q = {"expected_status": "answered", "gold_answers": ["CHG-0025", "2024-03-28", "Irakli Chkheidze"]}
    a = Answer("q", "answered", "Caused by CHG\u20110025 on 28 March 2024, authored by Irakli\u00a0Chkheidze.", [Claim("x", ["E1"])])
    assert M.answer_correct(q, a) and not M.answer_correct(q, a, strict=True)
    wrong = Answer("q", "answered", "Caused by CHG-0026 on 28 March 2024, authored by Irakli Chkheidze.")
    assert not M.answer_correct(q, wrong)
    wrong_date = Answer("q", "answered", "CHG-0025, 29 March 2024, Irakli Chkheidze")
    assert not M.answer_correct(q, wrong_date)
    assert "March 5, 2024" in M.gold_variants("2024-03-05") and "5 March 2024" in M.gold_variants("2024-03-05")
    unans = {"expected_status": "insufficient_evidence", "gold_answers": []}
    assert not M.answer_correct(unans, a) and M.answer_correct(unans, Answer("q", "insufficient_evidence", "no"))

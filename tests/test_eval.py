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

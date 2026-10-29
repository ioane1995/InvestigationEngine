import math

from app.models import Chunk, Hit
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import minmax, reciprocal_rank_fusion, weighted_score_fusion
from app.retrieval.hybrid import RetrievalConfig
from app.retrieval.rerank import FeatureReranker, _min_window, assess_evidence


def test_bm25_ranks_and_idf():
    ix = BM25Index()
    ix.add("a", "d1", "the cat sat on the mat")
    ix.add("b", "d2", "the dog chased the cat around the garden garden")
    ix.add("c", "d3", "completely unrelated text about finance")
    res = ix.search("garden cat", 3)
    assert res[0][0] == "b" and {r[0] for r in res} == {"a", "b"}
    assert ix.idf("finance") > ix.idf("cat")


def test_bm25_matches_manual_formula():
    ix = BM25Index(k1=1.5, b=0.75)
    ix.add("a", "d1", "alpha beta"); ix.add("b", "d2", "alpha gamma gamma gamma")
    tf, dl, avgdl, n, df = 1, 2, 3.0, 2, 2
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expect = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert abs(dict(ix.search("alpha", 2))["a"] - expect) < 1e-9


def test_bm25_remove_doc_and_update():
    ix = BM25Index()
    ix.add("a::0", "a", "unique zebra"); ix.add("b::0", "b", "common text")
    assert ix.search("zebra", 5)[0][0] == "a::0"
    assert ix.remove_doc("a") == 1 and ix.search("zebra", 5) == [] and len(ix) == 1
    ix.add("b::0", "b", "replaced content zebra")  # same chunk id -> replaced
    assert len(ix) == 1 and ix.search("zebra", 5)[0][0] == "b::0" and ix.search("common", 5) == []


def test_bm25_filters():
    ix = BM25Index()
    ix.add("a", "d1", "error log", {"format": "csv"}); ix.add("b", "d2", "error report", {"format": "md"})
    assert [c for c, _ in ix.search("error", 5, {"format": "md"})] == ["b"]


def test_rrf_prefers_items_ranked_well_by_both():
    s = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60)
    assert max(s, key=s.get) == "a" and s["a"] > s["b"] and s["d"] < s["a"]


def test_weighted_fusion_and_minmax():
    assert minmax({"x": 5, "y": 10}) == {"x": 0.0, "y": 1.0}
    f = weighted_score_fusion([{"a": 1, "b": 0}, {"a": 0, "b": 1}], [3, 1])
    assert f["a"] > f["b"]


def test_min_window():
    assert _min_window(list("xaxxbxc"), {"a", "b", "c"}) == 6
    assert _min_window(["a"], {"a", "z"}) == 0


def _hit(cid, text, score=1.0, title="t"):
    return Hit(Chunk(cid, cid, 0, text, title, "", "s", "md", "x", len(text.split())), score)


def test_reranker_boosts_identifier_match_and_dedupes_with_mmr():
    ix = BM25Index()
    texts = {"a": "Severity SEV2 for INC-0012", "b": "Severity SEV1 for INC-0099", "c": "Severity SEV2 for INC-0012"}
    for k, t in texts.items():
        ix.add(k, k, t)
    hits = [_hit("b", texts["b"], 1.0), _hit("a", texts["a"], 0.9), _hit("c", texts["c"], 0.8)]
    out = FeatureReranker(ix, mmr_lambda=0.5).rerank("severity of INC-0012", hits, 2)
    assert out[0].chunk.chunk_id in ("a", "c") and {h.chunk.chunk_id for h in out} != {"a", "c"}  # MMR avoids two identical chunks


def test_assess_evidence_flags_missing_identifier():
    ix = BM25Index(); ix.add("a", "a", "incident INC-0001 root cause config")
    hits = [_hit("a", "incident INC-0001 root cause config")]
    assert assess_evidence("root cause of INC-0001", hits, ix, 0.3).ok
    bad = assess_evidence("root cause of INC-9999", hits, ix, 0.3)
    assert not bad.ok and bad.missing_ids == ["inc-9999"]


def test_hybrid_search_end_to_end(loaded):
    hits = loaded.retriever.search("Who authored CHG-0123?", RetrievalConfig(top_k=3))
    assert hits[0].chunk.doc_id == "chg-1"
    assert {"fused"} <= set(hits[0].components) and "bm25_rank" in hits[0].components
    for mode in ("dense", "bm25"):
        assert loaded.retriever.search("ledger runbook restart", RetrievalConfig(mode=mode, top_k=2))
    assert loaded.retriever.search("severity", RetrievalConfig(top_k=5), {"format": "md"})


def test_near_duplicate_hits_are_collapsed(container):
    from app.models import RawDocument
    base = "Drain traffic from the payments service, restart the instances one by one and verify the health checks before continuing to the next."
    container.ingestion.ingest_many([RawDocument("x1", "Runbook A", base + " Revision 1.", "a.md", "md"),
                                     RawDocument("x2", "Runbook A", base + " Revision 2.", "b.md", "md")])
    hits = container.retriever.search("restart payments instances health checks", RetrievalConfig(top_k=5))
    assert len(hits) == 1 and set(hits[0].doc_ids) == {"x1", "x2"}


def test_search_falls_back_to_bm25_when_embedding_service_fails(loaded, monkeypatch):
    from app.embeddings import EmbeddingError
    def boom(text): raise EmbeddingError("embedding API unreachable")
    monkeypatch.setattr(loaded.embedder.inner, "embed_query", boom)
    for mode in ("hybrid", "dense"):
        hits = loaded.retriever.search("Who authored CHG-0123?", RetrievalConfig(mode=mode, top_k=3))
        assert hits and hits[0].chunk.doc_id == "chg-1" and "dense_unavailable" in hits[0].components
    ans = loaded.engine.ask("What was the severity of INC-0012?")  # the whole pipeline keeps working, and says so
    assert ans.status == "answered" and any("keyword search only" in w for w in ans.warnings)


def test_embedding_api_client_retries_transient_errors():
    import httpx, pytest
    from app.embeddings import EmbeddingError, OpenAICompatEmbedder
    calls = {"n": 0}
    def h(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})
    sleeps = []
    e = OpenAICompatEmbedder("http://x/v1", "k", "m", 2, client=httpx.Client(transport=httpx.MockTransport(h)), sleep=sleeps.append)
    v = e.embed_query("hi")
    assert calls["n"] == 3 and len(sleeps) == 2 and abs(float(v @ v) - 1.0) < 1e-6  # normalised
    dead = OpenAICompatEmbedder("http://x/v1", "k", "m", 2, client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))), sleep=lambda s: None)
    with pytest.raises(EmbeddingError):
        dead.embed_query("hi")

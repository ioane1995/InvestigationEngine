from app.ingestion.loaders import load_bytes
from app.models import RawDocument
from tests.conftest import DOCS, make_settings
from app.container import build_container


def test_create_unchanged_update_delete(container):
    ing = container.ingestion
    d = DOCS[0]
    assert ing.ingest(d).status == "created"
    assert ing.ingest(d).status == "unchanged"
    changed = RawDocument(d.doc_id, d.title, d.text + "\n\nAn additional paragraph about rollback.", d.source, d.format)
    r = ing.ingest(changed)
    assert r.status == "updated" and ing.docs[d.doc_id].created_at <= ing.docs[d.doc_id].updated_at
    assert any("rollback" in c.text for c in container.corpus.doc_chunks(d.doc_id))
    assert container.store.count_chunks() == len(container.corpus.chunks)  # store and BM25 mirror stay in sync
    assert ing.delete(d.doc_id) and not ing.delete(d.doc_id)
    assert container.store.count_chunks() == 0 and container.corpus.bm25.search("rollback", 3) == []


def test_exact_duplicate_documents_are_rejected(container):
    ing = container.ingestion
    a = RawDocument("a", "A", "Same   content here.", "a.txt", "txt")
    b = RawDocument("b", "B", "same content HERE.", "b.txt", "txt")
    assert ing.ingest(a).status == "created"
    r = ing.ingest(b)
    assert r.status == "duplicate" and r.duplicate_of == "a" and "b" not in ing.docs
    # duplicates inside one batch
    res = ing.ingest_many([RawDocument("c", "C", "brand new text", "c.txt", "txt"), RawDocument("d", "D", "Brand New Text", "d.txt", "txt")])
    assert [r.status for r in res] == ["created", "duplicate"]


def test_update_after_delete_frees_hash(container):
    ing = container.ingestion
    a = RawDocument("a", "A", "text one", "a.txt", "txt")
    ing.ingest(a); ing.delete("a")
    assert ing.ingest(RawDocument("z", "Z", "text one", "z.txt", "txt")).status == "created"


def test_batch_isolates_failures(container, monkeypatch):
    ing = container.ingestion
    ok = RawDocument("ok", "OK", "good document text", "ok.txt", "txt")
    empty = RawDocument("empty", "E", "   ", "e.txt", "txt")
    res = {r.doc_id: r.status for r in ing.ingest_many([ok, empty])}
    assert res == {"ok": "created", "empty": "failed"}


def test_embedding_failure_does_not_corrupt_state(container, monkeypatch):
    ing = container.ingestion
    ing.ingest(DOCS[3])
    def boom(texts): raise RuntimeError("embedding service down")
    monkeypatch.setattr(container.embedder.inner, "embed_documents", boom)
    r = ing.ingest(RawDocument(DOCS[3].doc_id, "Runbook ledger", "updated text that should not land", "rb.md", "md"))
    assert r.status == "failed"
    assert "updated text" not in " ".join(c.text for c in container.corpus.doc_chunks(DOCS[3].doc_id))  # old version intact


def test_persistence_across_restart(tmp_path):
    s = make_settings(qdrant_path=str(tmp_path / "q"))
    c1 = build_container(s, load=False)
    c1.ingestion.ingest_many(DOCS)
    c1.store.client.close()
    c2 = build_container(s, load=True)  # new process: mirrors rebuilt from the persistent store
    assert len(c2.ingestion.docs) == len(DOCS) and len(c2.corpus.chunks) > 0
    assert c2.ingestion.ingest(DOCS[0]).status == "unchanged"
    assert c2.retriever.search("Who authored CHG-0123?")[0].chunk.doc_id == "chg-1"


def test_embedder_mismatch_is_refused(tmp_path):
    import pytest
    from app.stores.qdrant_store import StoreMismatch
    s = make_settings(qdrant_path=str(tmp_path / "q"))
    build_container(s, load=False).store.client.close()
    with pytest.raises(StoreMismatch):
        build_container(make_settings(qdrant_path=str(tmp_path / "q"), embedding_dim=128), load=False)


def test_embedding_store_id_override_is_explicit_and_logged(tmp_path):
    from app.embeddings import build_embedder
    e = build_embedder(make_settings(embedding_store_id="fastembed:BAAI/bge-small-en-v1.5"))
    assert e.name == "fastembed:BAAI/bge-small-en-v1.5" and e.inner.name == "hashing-384"  # pinned identity differs from the real provider
    assert build_embedder(make_settings()).name == "hashing-384"

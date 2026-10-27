import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.container import build_container
from tests.conftest import make_settings

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_files"


@pytest.fixture
def client():
    s = make_settings(admin_api_key="secret", rate_limit_per_min=1000)
    app = create_app(s, build_container(s, load=False))
    with TestClient(app) as c:
        yield c


H = {"X-API-Key": "secret"}


def upload_samples(client, names=None):
    files = [("files", (p.name, p.read_bytes())) for p in sorted(SAMPLES.iterdir()) if p.suffix in (".pdf", ".docx", ".json", ".csv", ".html", ".txt") and p.name != "sample_questions.txt"
             and (names is None or p.name in names)]
    return client.post("/api/documents", files=files, headers=H)


def test_health_stats_ui(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/stats").json()["documents"] == 0
    assert "AI Investigation Engine" in client.get("/").text


def test_writes_require_api_key(client):
    f = [("files", ("a.txt", b"hello"))]
    assert client.post("/api/documents", files=f).status_code == 401
    assert client.post("/api/documents", files=f, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.delete("/api/documents/x").status_code == 401
    assert client.post("/api/documents", files=f, headers=H).status_code == 200


def test_production_without_admin_key_disables_writes():
    s = make_settings(app_env="production", admin_api_key=None)
    with TestClient(create_app(s, build_container(s, load=False))) as c:
        assert c.post("/api/documents", files=[("files", ("a.txt", b"hello"))]).status_code == 403


def test_full_document_lifecycle_multi_format(client):
    r = upload_samples(client).json()["results"]
    assert {x["status"] for x in r} == {"created"}
    assert client.get("/api/stats").json()["documents"] == len(r)
    assert upload_samples(client).json()["results"][0]["status"] == "unchanged"
    docs = client.get("/api/documents?limit=100").json()
    assert {d["format"] for d in docs["items"]} >= {"pdf", "docx", "json", "csv", "html", "txt"}
    pdf = next(d for d in docs["items"] if d["format"] == "pdf")
    detail = client.get(f"/api/documents/{pdf['doc_id']}").json()
    assert detail["chunks"] and detail["metadata"]["pages"] >= 1
    # update (replace) then delete
    rep = client.put(f"/api/documents/{pdf['doc_id']}", files={"file": ("x.txt", b"Severity: SEV9 for INC-0002 replaced")}, data={"chunking": "fixed"}, headers=H)
    assert rep.json()["status"] == "updated"
    assert client.delete(f"/api/documents/{pdf['doc_id']}", headers=H).status_code == 200
    assert client.get(f"/api/documents/{pdf['doc_id']}").status_code == 404
    assert client.delete(f"/api/documents/{pdf['doc_id']}", headers=H).status_code == 404


def test_bad_uploads_are_reported_not_crashed(client):
    r = client.post("/api/documents", files=[("files", ("evil.exe", b"MZ")), ("files", ("broken.pdf", b"nope")), ("files", ("ok.txt", b"fine text"))], headers=H).json()["results"]
    assert [x["status"] for x in r] == ["failed", "failed", "created"]
    assert client.post("/api/documents", files=[("files", ("a.txt", b"x"))], data={"chunking": "bogus"}, headers=H).status_code == 422


def test_ask_search_and_cache(client):
    upload_samples(client)
    q = {"question": "What was the severity of INC-0001?"}
    a = client.post("/api/ask", json=q).json()
    assert a["status"] == "answered" and a["citations"] and a["usage"]["llm_calls"] >= 0
    assert client.post("/api/ask", json=q).json()["cached"] is True
    assert client.post("/api/ask", json={"question": "What was the root cause of INC-8888?"}).json()["status"] == "insufficient_evidence"
    s = client.post("/api/search", json={"query": "payments-api owner team", "top_k": 3}).json()
    assert len(s["hits"]) == 3 and "components" in s["hits"][0]
    m = client.get("/api/metrics").json()
    assert m["counters"]["ask_total"] >= 2 and m["answer_cache"]["hits"] >= 1


def test_cache_invalidated_after_corpus_change(client):
    client.post("/api/documents/text", json={"title": "Notes INC-0042", "text": "Severity: SEV1 for INC-0042."}, headers=H)
    q = {"question": "What was the severity of INC-0042?"}
    assert "SEV1" in client.post("/api/ask", json=q).json()["answer"]
    client.post("/api/documents/text", json={"title": "Notes INC-0042", "text": "Severity: SEV3 for INC-0042."}, headers=H)
    assert "SEV3" in client.post("/api/ask", json=q).json()["answer"]


def test_validation_and_rate_limit():
    s = make_settings(rate_limit_per_min=2)
    with TestClient(create_app(s, build_container(s, load=False))) as c:
        assert c.post("/api/ask", json={"question": "x"}).status_code == 422
        for _ in range(2):
            assert c.post("/api/ask", json={"question": "What is going on here?"}).status_code == 200
        r = c.post("/api/ask", json={"question": "What is going on here?"})
        assert r.status_code == 429 and "retry-after" in r.headers


def test_stream_endpoint_emits_steps_then_answer(client):
    upload_samples(client)
    with client.stream("POST", "/api/ask/stream", json={"question": "Which change caused INC-0003 and who authored it?"}) as r:
        body = "".join(r.iter_text())
    assert body.index("event: step") < body.index("event: answer") and "event: plan" not in body
    assert '"step": "plan"' in body and '"step": "assess"' in body

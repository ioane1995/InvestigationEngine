"""FastAPI application: REST API + SSE streaming + static UI."""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api.support import Metrics, SlidingWindowLimiter, TTLCache, client_ip, make_admin_dependency
from app.config import Settings, get_settings
from app.container import Container, build_container
from app.ingestion.loaders import LoaderError, UnsupportedFormat, load_bytes, stable_doc_id
from app.llm.base import LLMError
from app.models import IngestResult, RawDocument
from app.rag.types import RagConfig
from app.retrieval.hybrid import RetrievalConfig

log = logging.getLogger("api")
STATIC = Path(__file__).resolve().parent.parent / "static"


class AskOptions(BaseModel):
    mode: str = Field("hybrid", pattern="^(hybrid|dense|bm25)$")
    rerank: bool = True
    hops: int = Field(1, ge=0, le=3)
    use_planner: bool = True
    top_k: int = Field(8, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    options: AskOptions = AskOptions()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    mode: str = Field("hybrid", pattern="^(hybrid|dense|bm25)$")
    fusion: str = Field("rrf", pattern="^(rrf|weighted)$")
    rerank: bool = True
    top_k: int = Field(8, ge=1, le=30)
    format: Optional[str] = None


class TextDoc(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=500_000)
    doc_id: Optional[str] = None


def create_app(settings: Optional[Settings] = None, container: Optional[Container] = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format='{"t":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}')

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.c = container or await run_in_threadpool(build_container, settings)
        app.state.ready = True
        yield

    app = FastAPI(title="AI Investigation Engine", version="1.0.0", lifespan=lifespan)
    app.state.ready = False
    limiter = SlidingWindowLimiter(settings.rate_limit_per_min)
    cache = TTLCache(settings.cache_size, settings.cache_ttl_s)
    metrics = Metrics()
    admin = make_admin_dependency(settings)

    def C() -> Container:
        c = getattr(app.state, "c", None)
        if c is None:
            raise HTTPException(503, "service is starting (loading the index); retry in a few seconds")
        return c

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t = time.perf_counter()
        try:
            resp = await call_next(request)
        except Exception:
            log.exception("unhandled error rid=%s path=%s", rid, request.url.path)
            metrics.inc("http_500")
            resp = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        resp.headers["x-request-id"] = rid
        resp.headers["x-content-type-options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            log.info("rid=%s %s %s -> %s %.0fms", rid, request.method, request.url.path, resp.status_code, (time.perf_counter() - t) * 1000)
        return resp

    # ------------------------------------------------------------ system
    @app.get("/health")
    def health():
        return {"status": "ok" if app.state.ready else "starting"}

    @app.get("/api/stats")
    def stats():
        c = C()
        return {"documents": len(c.ingestion.docs), "chunks": len(c.corpus.chunks), "embedder": c.embedder.name, "llm": c.llm.name,
                "llm_is_offline_heuristic": c.llm.name == "heuristic-extractive", "writes_protected": bool(settings.admin_api_key),
                "supported_formats": ["txt", "md", "html", "pdf", "docx", "json", "csv"]}

    @app.get("/api/metrics")
    def get_metrics():
        c = C()
        return {**metrics.snapshot(), "answer_cache": {"hits": cache.hits, "misses": cache.misses},
                "query_embedding_cache": {"hits": getattr(c.embedder, "hits", None), "misses": getattr(c.embedder, "misses", None)}}

    # ------------------------------------------------------------ documents
    def _result(r: IngestResult) -> dict[str, Any]:
        return {"doc_id": r.doc_id, "status": r.status, "chunks": r.n_chunks, "detail": r.detail, "duplicate_of": r.duplicate_of}

    @app.post("/api/documents", dependencies=[Depends(admin)])
    async def upload(files: list[UploadFile] = File(...), chunking: str = Form("structure")):
        if chunking not in ("structure", "fixed"):
            raise HTTPException(422, "chunking must be 'structure' or 'fixed'")
        c = C()
        limit = settings.max_upload_mb * 1024 * 1024
        out: list[dict[str, Any]] = []
        for f in files:
            name = f.filename or "upload"
            data = await f.read(limit + 1)
            if len(data) > limit:
                out.append({"doc_id": stable_doc_id(name), "status": "failed", "detail": f"file larger than {settings.max_upload_mb} MB"})
                continue
            try:
                raw = await run_in_threadpool(load_bytes, data, name)
                res = await run_in_threadpool(c.ingestion.ingest, raw, chunking)
                out.append(_result(res))
            except (UnsupportedFormat, LoaderError) as e:
                out.append({"doc_id": stable_doc_id(name), "status": "failed", "detail": str(e)})
        return {"results": out}

    @app.post("/api/documents/text", dependencies=[Depends(admin)])
    async def upload_text(body: TextDoc, chunking: str = Query("structure", pattern="^(structure|fixed)$")):
        c = C()
        doc_id = body.doc_id or stable_doc_id(body.title + ".txt")
        raw = RawDocument(doc_id, body.title, body.text, f"{body.title}.txt", "txt", {"origin": "pasted"})
        return _result(await run_in_threadpool(c.ingestion.ingest, raw, chunking))

    @app.put("/api/documents/{doc_id}", dependencies=[Depends(admin)])
    async def replace_document(doc_id: str, file: UploadFile = File(...), chunking: str = Form("structure")):
        c = C()
        if doc_id not in c.ingestion.docs:
            raise HTTPException(404, "unknown doc_id")
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        try:
            raw = await run_in_threadpool(load_bytes, data, file.filename or "upload", doc_id)
        except (UnsupportedFormat, LoaderError) as e:
            raise HTTPException(422, str(e))
        return _result(await run_in_threadpool(c.ingestion.ingest, raw, chunking))

    @app.delete("/api/documents/{doc_id}", dependencies=[Depends(admin)])
    async def delete_document(doc_id: str):
        if not await run_in_threadpool(C().ingestion.delete, doc_id):
            raise HTTPException(404, "unknown doc_id")
        return {"deleted": doc_id}

    @app.get("/api/documents")
    def list_documents(limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0), q: str = ""):
        docs = sorted(C().ingestion.docs.values(), key=lambda d: d.updated_at, reverse=True)
        if q:
            ql = q.lower()
            docs = [d for d in docs if ql in d.title.lower() or ql in d.doc_id.lower() or ql in d.source.lower()]
        page = docs[offset: offset + limit]
        return {"total": len(docs), "offset": offset, "items": [d.to_payload() for d in page]}

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: str):
        c = C()
        rec = c.ingestion.docs.get(doc_id)
        if not rec:
            raise HTTPException(404, "unknown doc_id")
        return {**rec.to_payload(), "chunks": [{"chunk_id": k.chunk_id, "index": k.index, "heading": k.heading, "text": k.text, "n_tokens": k.n_tokens}
                                               for k in c.corpus.doc_chunks(doc_id)]}

    # ------------------------------------------------------------ search / ask
    @app.post("/api/search")
    def search(body: SearchRequest):
        c = C()
        cfg = RetrievalConfig(mode=body.mode, fusion=body.fusion, rerank=body.rerank, top_k=body.top_k, candidate_k=max(40, body.top_k * 3))
        hits = c.retriever.search(body.query, cfg, {"format": body.format} if body.format else None)
        return {"query": body.query, "hits": [{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "title": h.chunk.title, "heading": h.chunk.heading,
                                               "text": h.chunk.text, "score": round(h.score, 4), "components": h.components, "also_doc_ids": h.also_doc_ids}
                                              for h in hits]}

    def _configs(o: AskOptions) -> tuple[RagConfig, RetrievalConfig]:
        return (RagConfig(use_planner=o.use_planner, hops=o.hops, min_coverage=settings.min_evidence_coverage, max_tool_rounds=settings.max_tool_rounds,
                          evidence_k=settings.evidence_k),
                RetrievalConfig(mode=o.mode, rerank=o.rerank, top_k=o.top_k, candidate_k=max(settings.candidate_k, o.top_k * 3)))

    def _rate_check(request: Request) -> None:
        ok, retry = limiter.allow(client_ip(request))
        if not ok:
            metrics.inc("rate_limited")
            raise HTTPException(429, f"rate limit exceeded; retry in {retry:.0f}s", headers={"Retry-After": str(int(retry) + 1)})

    @app.post("/api/ask")
    async def ask(body: AskRequest, request: Request):
        _rate_check(request)
        c = C()
        key = (body.question.strip().lower(), body.options.model_dump_json(), c.ingestion.version)
        if (hit := cache.get(key)) is not None:
            return {**hit, "cached": True}
        rag, ret = _configs(body.options)
        t = time.perf_counter()
        try:
            ans = await run_in_threadpool(c.engine.ask, body.question.strip(), rag, ret)
        except LLMError as e:  # should be handled inside the engine; belt and braces
            raise HTTPException(502, f"language model error: {e}")
        metrics.observe((time.perf_counter() - t) * 1000)
        metrics.inc("ask_total")
        metrics.inc(f"status_{ans.status}")
        metrics.inc("llm_prompt_tokens", ans.usage.get("prompt_tokens", 0))
        metrics.inc("llm_completion_tokens", ans.usage.get("completion_tokens", 0))
        metrics.inc("llm_cost_usd", ans.usage.get("cost_usd", 0.0))
        payload = ans.to_dict()
        if not ans.degraded:  # never cache degraded answers
            cache.put(key, payload)
        return payload

    @app.post("/api/ask/stream")
    async def ask_stream(body: AskRequest, request: Request):
        """Server-Sent Events: one `step` event per pipeline stage (plan, retrieve, hop, assess, tool calls...) then `answer`."""
        _rate_check(request)
        c = C()
        rag, ret = _configs(body.options)
        q: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        def work() -> None:
            try:
                ans = c.engine.ask(body.question.strip(), rag, ret, on_event=lambda e: q.put(("step", e)))
                q.put(("answer", ans.to_dict()))
            except Exception as e:  # pragma: no cover
                log.exception("stream failed")
                q.put(("error", {"detail": str(e)}))
            q.put(("end", {}))

        threading.Thread(target=work, daemon=True).start()

        async def gen():
            while True:
                try:
                    kind, data = await asyncio.get_event_loop().run_in_executor(None, lambda: q.get(timeout=120))
                except queue.Empty:
                    yield "event: error\ndata: {\"detail\":\"timeout\"}\n\n"
                    return
                if kind == "end":
                    return
                yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------ UI
    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(STATIC / "index.html")

    return app


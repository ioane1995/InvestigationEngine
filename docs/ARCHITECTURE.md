# Architecture & technical decisions

## 1. System overview

```
                 ┌────────────── ingestion ──────────────┐
 upload / jsonl →│ loaders → chunker → dedup → embed     │→ Qdrant (vectors + payload)  ┐
                 └───────────────────────────────────────┘→ CorpusIndex (own BM25 + chunk mirror, in RAM)
                                                                                         │
 question → plan (LLM tool) → hybrid search per sub-query → RRF merge → hop expansion → evidence assessment ─┐
                                                                                                           │ gate: abstain?
             ┌─────────────────────────────── LLM with tools: search_documents / get_document / submit_answer ◄┘
             ▼
        claim verification → conflict candidates → Answer {status, claims+citations, conflicts, missing_info, trace, usage}
```

| Layer | Module | Responsibility |
|---|---|---|
| Config | `config.py` | typed settings from env / `.env`; no secrets in code |
| Domain | `models.py`, `rag/types.py` | dataclasses shared by all layers |
| Ingestion | `ingestion/loaders.py`, `chunking.py`, `service.py` | parsing, chunking, dedup, CRUD orchestration |
| Storage | `stores/qdrant_store.py` | chunk vectors, document registry, embedder pinning |
| Retrieval | `retrieval/bm25.py`, `fusion.py`, `rerank.py`, `hybrid.py` | own lexical index, fusion, re-ranking, evidence assessment |
| LLM | `llm/base.py`, `openai_compat.py`, `heuristic.py` | provider abstraction, retries, offline fallback |
| RAG | `rag/engine.py`, `conflicts.py`, `verify.py`, `safety.py`, `prompts.py` | agentic pipeline |
| API/UI | `api/`, `static/index.html` | REST + SSE, auth, rate limit, cache, metrics, UI |
| Evaluation | `eval/` | synthetic data, metrics, runner, CLI |

Dependency direction is strictly downward (API → RAG → retrieval → store); `container.py` is the only place that wires concrete classes.

## 2. Ingestion

* **Formats (7):** txt, md, html (BeautifulSoup, scripts removed, headings preserved), pdf (pypdf; page count, author, title), docx (python-docx; heading styles → markdown headings, tables flattened), json (flattened to `key: value` lines), csv (one `col: val; …` line per row). Corrupt / encrypted / empty files raise `LoaderError`; unsupported types raise `UnsupportedFormat`; the API reports these per file instead of failing the batch. Scanned PDFs (no text layer) are rejected explicitly — OCR is out of scope.
* **Metadata:** filename, format, size, first ISO date found, pages/author/rows/columns where the format has them; title from the format's own metadata or first heading.
* **Chunking — two strategies, both implemented and compared (see evaluation):**
  * `fixed`: sliding window of 200 whitespace tokens, 40 overlap. Cheap and uniform, but it collapses line structure and ignores sections.
  * `structure` (default): splits on headings → sentences/lines, packs up to 200 tokens, carries the heading path, overlaps by one sentence, merges tiny tails. Each chunk is embedded as `title | heading \n text` (contextual embedding) so a chunk like `- Duration: 47 minutes` still knows it belongs to *Incident Report INC-0012*.
* **Deduplication (3 levels, see `ingestion/service.py` docstring):** exact document duplicates (normalised-content SHA-256) are rejected and the canonical id is returned; same id + same hash is an idempotent no-op; near-duplicate chunks are *kept* (provenance!) but collapsed at query time with an own 64-bit SimHash (Hamming ≤ 3), and the collapsed documents stay visible as `also_doc_ids` / in citations.
* **Create / update / delete:** `doc_id` is stable (derived from the file name) so re-upload = update; update deletes the old chunks *only after* the new embeddings were computed successfully (a failed embedding call leaves the old version intact — tested). BM25 mirror, Qdrant and registry are updated together under one lock; a `version` counter invalidates the answer cache.

## 3. Retrieval (the "own algorithm" requirement)

All of the following are implemented in this repo without a search/ranking library:

1. **BM25 (Okapi) inverted index** — own tokenizer (identifiers like `INC-0012` are indexed whole *and* split), light stemmer, postings with TF, RSJ-IDF, length normalisation, heap top-k, incremental add/remove-by-document, metadata filters. Verified against the closed-form formula in a unit test.
2. **Rank fusion** — Reciprocal Rank Fusion (k=60) and min-max weighted fusion; both evaluated.
3. **Feature re-ranker** — IDF-weighted query-term coverage, identifier match, bigram match, minimal-window proximity (sliding window), title match and the fused score, combined linearly; followed by **MMR** diversification using Jaccard similarity so the LLM does not receive five paraphrases of one paragraph.
4. **Near-duplicate collapse** — SimHash over unigrams+bigrams.
5. **Evidence assessment / abstention gate** — IDF-weighted share of the question's content terms covered by the top evidence; an identifier in the question that appears in *no* retrieved text forces abstention before any LLM call (saves cost and removes a hallucination path).
6. **Hop expansion** — identifiers found in the best hits but absent from the question (e.g. `CHG-0123` inside an incident report) trigger follow-up searches: a deterministic form of multi-hop retrieval that works even without an LLM.

Dense retrieval uses Qdrant (cosine). The in-process mirror (`CorpusIndex`) is rebuilt from Qdrant at start-up (~tens of seconds for 60k chunks), which keeps the stack to **one** external service.

## 4. LLM / RAG pipeline

1. **Plan** — `submit_plan` tool call → ≤4 sub-queries (identifiers verbatim). Failure ⇒ fall back to the raw question.
2. **Retrieve & merge** — hybrid search per sub-query, merged with RRF (original question weighted 1.5×), then hop expansion.
3. **Gate** — see above.
4. **Reason** — the model receives numbered evidence blocks `E1…En` and tools `search_documents`, `get_document`, `submit_answer`. It may search again (max 2 tool rounds; the last round *forces* `submit_answer`). Malformed tool JSON / invalid status ⇒ an error message is returned *as the tool result* so the model can repair it; a plain-text reply is nudged once.
5. **Verify** — every claim's citations must exist in the evidence book (invented `E99` is dropped); numbers/identifiers in a claim must appear verbatim in the cited text; other content words need ≥60 % overlap. If no claim survives, `answered` is downgraded to `insufficient_evidence`.
6. **Conflicts** — a rule-based detector extracts `field: value` lines per primary entity and reports fields with different normalised values (`1 hour` = `60 minutes`) in **different documents**. They are hints for the LLM *and* attached to the answer if the LLM overlooked them.
7. **Graceful degradation** — rate limit / timeout / provider error after retries ⇒ an extractive answer from the offline reader, flagged `degraded` and never cached.

Prompt-injection hygiene: evidence is wrapped in tags, tag break-outs are neutralised, the system prompt declares evidence to be untrusted data, instruction-like text is flagged (`warnings`), and step 5 drops claims that the sources do not support.

### Decisions & trade-offs

| Decision | Why | Cost / alternative |
|---|---|---|
| Qdrant for vectors, BM25 in RAM | one managed service, free cloud tier; lexical side fully ours | RAM grows with corpus (≈1 GB RSS at 60k chunks); a Postgres FTS/`pgvector` design would persist BM25 but would not be "own" |
| Hashing embedder as a first-class provider | deterministic tests/CI/offline eval; no model download | lexical only — real semantic quality needs `fastembed`/API (documented in the evaluation) |
| Heuristic LLM fallback | demo never shows a stack trace when the key is missing/limits are hit | extractive, no reasoning; flagged `degraded` |
| OpenAI-compatible client written on `httpx` | provider-independent, explicit retry/timeout semantics, easy to unit-test | no SDK conveniences |
| Evidence ids `E1…` instead of chunk ids | fewer tokens, hallucinated citations are trivially detectable | ids are per-request |
| Linear re-ranker with hand-set weights | interpretable, no training data needed, ~ms latency | weights are not learned; a cross-encoder would likely be better (future work) |
| Abstention gate is lexical and conservative (coverage < 0.30 or unknown identifier) | cheap, explainable, no false refusals on paraphrased questions | finer "no answer" judgement is delegated to the LLM + verifier |

## 5. Deployment (free tier)

```
 Browser ──HTTPS──► Hugging Face Space (Docker, 1 worker, 2 vCPU / 16 GB, sleeps when idle)
                         │  FastAPI + own BM25 mirror + fastembed (query embeddings only)
                         ├──HTTPS──► Qdrant Cloud free cluster   (vectors + document registry → survives restarts)
                         └──HTTPS──► LLM provider (OpenAI-compatible, free tier)
```

1. Create a Qdrant Cloud free cluster → `QDRANT_URL`, `QDRANT_API_KEY`.
2. Locally: `python -m scripts.generate_synthetic … --out data/corpus_50k`, then `python -m scripts.ingest_corpus --corpus data/corpus_50k/corpus.jsonl` with the cloud credentials in `.env` and `EMBEDDING_PROVIDER=fastembed` (embedding 60k chunks is done **once, offline**; the Space only embeds queries).
3. Create a Space (SDK: Docker), push this repo (the README front-matter sets `app_port: 7860`), add **Secrets**: `ADMIN_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `LLM_API_KEY`, and variables `APP_ENV=production`, `LLM_PROVIDER=openai`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_PROVIDER=fastembed`.
4. Persistence: everything durable is in Qdrant; the container is stateless. The start-up hook rebuilds the BM25 mirror from Qdrant, so a restart loses nothing (tested: `test_persistence_across_restart`).
5. Secrets: only environment variables; `.env` is git-ignored; write endpoints need `X-API-Key` (constant-time compare) and are **disabled** in production if no key is configured; errors never echo keys; logs contain no secrets.

**Free-tier limits and how they are handled** — *Sleep after inactivity*: first request after idle waits ~1–2 min while the index rebuilds (`/health` reports `starting`, the UI retries). *Qdrant free storage/RAM*: 384-d vectors for ~60k chunks fit (≈0.1 GB raw); if the quota is hit, ingest a subset via `--limit`. *LLM rate limits*: per-IP limiter (default 20 req/min), 10-min answer cache, planner can be disabled (1 LLM call per question instead of 2), automatic degraded mode. *Single worker*: required because the BM25 mirror is per-process; fine for a demo, a scale-out design would move the lexical index into the vector DB (Qdrant sparse vectors) or Postgres FTS.

**Cost model** — infrastructure: $0. LLM: ≈ 2 calls/question (plan + answer, +1 per tool round); the evidence prompt is ≈ 1.5–3 k tokens. `usage.cost_usd` is computed per answer from `LLM_PRICE_*` and exposed in `/api/metrics`. Embedding cost for 60k chunks with a local ONNX model: CPU time only.

## 6. Security & reliability checklist

* Input validation (pydantic limits, upload size cap, extension allow-list, `chunking` enum), per-file error isolation.
* Rate limiting (sliding window per IP), request ids, structured logs, `x-content-type-options`.
* UI renders all model/document text via `textContent` (no `innerHTML`) → no stored XSS from uploaded documents.
* Timeouts + bounded retries with jitter for every external call; retry-after honoured.
* Atomic-ish updates: failed embedding never destroys the previous version.
* Embedder pinning: a collection created with model A refuses to start with model B (`StoreMismatch`).

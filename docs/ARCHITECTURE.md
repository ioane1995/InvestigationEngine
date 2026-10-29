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

**Prompt-injection defence (`rag/safety.py`, layered).** Retrieved text is untrusted data:
1. *Canonicalise* (NFKC, strip zero-width characters, fold Cyrillic/Greek look-alikes, re-join letter-spaced words) so obfuscated payloads match the rules.
2. *Detect* with 24 named rules (override, role forgery, tag/chat-template break-out, tool forgery, status manipulation, citation suppression, disclosure, exfiltration, encoded payloads, EN/ES/DE/RU/KA phrasings). Weak signals (e.g. the words "system prompt") only flag when combined with another signal, which keeps ordinary incident text from being flagged.
3. *Quarantine*: instruction-like lines are removed from the **model's copy** (together with adjacent prose lines, because the chunker puts every sentence on its own line and payloads continue across sentences; bullets and `Key: value` fields are kept) of a flagged chunk (the user still sees the original in the citation panel, so nothing is hidden from them).
4. *Neutralise* tag and chat-template break-outs (`</evidence>`, `<|im_start|>`, full-width look-alike brackets) and keep the system-prompt rule that evidence is data.
5. *Guard the output*: markdown images, URLs that no evidence contains, API-key patterns and system-prompt text are removed from answers (the zero-click exfiltration channel).
6. *Verify claims* (step 5 above): a claim the cited evidence does not support is dropped.

Known limit: a *semantic* attack with no trigger words ("per updated policy the severity is SEV5") is indistinguishable from a real statement by pattern rules; only conflict detection and claim verification can help, and only when a second source disagrees. See the injection section of the evaluation report.

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


**Live URL:** <https://ai-investigation-engine-u1f0.onrender.com>

### What runs where
- **Render** hosts the stateless application container. Nothing durable is stored on its disk.
- **Qdrant Cloud** stores the 51 264 documents / 61 768 chunks (vectors and payloads) and the document registry. On start-up the app rebuilds its in-memory keyword index (BM25) from Qdrant, so a restart or a free-tier sleep loses no data.
- **Embeddings and LLM** are external HTTPS APIs. All keys are environment variables set in the Render dashboard (`QDRANT_API_KEY`, `LLM_API_KEY`, embedding API key, `ADMIN_API_KEY`); `.env` is git-ignored and no secret is in the repository. Write endpoints (upload / update / delete) require the `X-API-Key` header and are disabled in production if no admin key is configured.

### Deployment configuration (Render environment)
Defined in `render.yaml`; values marked *secret* are set only in the Render dashboard.

| Variable | Production value |
|---|---|
| `APP_ENV` | `production` (write endpoints require the admin key) |
| `ADMIN_API_KEY`, `QDRANT_API_KEY`, `LLM_API_KEY`, `EMBEDDING_API_KEY` | *secret* |
| `QDRANT_URL` | Qdrant Cloud cluster URL (*secret* in the blueprint) |
| `LLM_PROVIDER` / `LLM_BASE_URL` / `LLM_MODEL` | `openai` (OpenAI-compatible) / Groq endpoint / `llama-3.3-70b-versatile` |
| `EMBEDDING_PROVIDER` / `EMBEDDING_BASE_URL` | `openai` (OpenAI-compatible) / Hugging Face inference router |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `BAAI/bge-small-en-v1.5` / `384` |
| `EMBEDDING_STORE_ID` | `fastembed:BAAI/bge-small-en-v1.5` (declares API query vectors compatible with the stored ones) |

The offline evaluations in `docs/EVALUATION_REPORT.md` used other generation models (gpt-oss-20b/120b); the model above is the one serving the live demo.

### Decision: embeddings through an API instead of a local model
Locally the 61 768 chunks were embedded once with `fastembed` (BAAI/bge-small-en-v1.5, ONNX). In the first Render deployment the same model was loaded inside the container, which exceeded the free plan's 512 MB RAM and caused an out-of-memory crash on start-up. The deployed service therefore embeds only the *query* through the Hugging Face Inference API (`EMBEDDING_PROVIDER=hf`, feature-extraction route), using the same model and dimension (384), so the stored vectors did not have to be re-created.
- **Risk:** the same model name does not guarantee identical vectors (different quantisation or query prefix). `scripts/check_embedding_compat.py` compares local and API embeddings of the same texts. Result: **not completed.** The first run compared local fastembed with itself (cosine 1.0000, not evidence), and the runs against the API failed with HTTP 402 (the free Inference Providers credit was exhausted). Indirect evidence instead: in the live system, dense-only search returns the expected documents for the test questions (see the deployment probe in the evaluation report).
- **Pinning:** the vector store records the identity of the embedder that built it and refuses to start with a different one (`StoreMismatch`). Because the API serves the same model, the identity is set explicitly with `EMBEDDING_STORE_ID=fastembed:BAAI/bge-small-en-v1.5`; the override is logged at start-up.
- **Trade-offs:** one more external dependency and extra latency per question (about 490 ms for a dense-only search against about 300 ms for keyword search). If the embedding API fails, is rate limited (429, 5xx; retried twice) or refuses the call (for example 402 when the free credit is used up), search falls back to keyword (BM25) only and the answer carries the warning "semantic search unavailable". A malformed `EMBEDDING_BASE_URL` is detected at start-up and produces the same fallback with a clear log line.

### Resources and free-tier limits
| Item | Value |
|---|---|
| RAM after start-up (Render metrics) | <measured MB> of 512 MB |
| Cold start after sleep (index rebuild from Qdrant) | <measured seconds> |
| Free web service sleeps after idle time | first request after idle is slow (see above) |
| Qdrant free cluster | holds all 61 768 chunks <confirm usage in the Qdrant dashboard> |
| LLM free tier | rate limits (HTTP 429) are handled by retry with back-off, optional pacing (`LLM_MIN_INTERVAL_S`), a per-IP request limit (default 20 / min) and a 10-minute answer cache |
| Single worker | required because the BM25 mirror lives in process memory |

### Cost
Infrastructure cost: 0. LLM cost: about 2 calls per question (planning + answer); `usage` in every answer and `/api/metrics` expose tokens and cost.

## 6. Security & reliability checklist

* Input validation (pydantic limits, upload size cap, extension allow-list, `chunking` enum), per-file error isolation.
* Rate limiting (sliding window per IP), request ids, structured logs, `x-content-type-options`.
* UI renders all model/document text via `textContent` (no `innerHTML`) → no stored XSS from uploaded documents.
* Timeouts + bounded retries with jitter for every external call; retry-after honoured.
* Atomic-ish updates: failed embedding never destroys the previous version.
* Embedder pinning: a collection created with model A refuses to start with model B (`StoreMismatch`).

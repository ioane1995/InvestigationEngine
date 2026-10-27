---
title: AI Investigation Engine
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# AI Investigation Engine

A retrieval-augmented investigation system for large document collections. It does **not** just "chat with a PDF": a question is
planned into sub-queries, answered from several documents at once (multi-hop), every claim is cited and machine-verified against its
sources, contradictory sources are reported as such, and the system **abstains** instead of inventing facts.

| Requirement (task PDF) | Where it lives |
|---|---|
| ≥3 file formats, metadata, chunking, dedup, add/update/delete, 2 chunking strategies compared | `app/ingestion/` (7 formats: txt, md, html, pdf, docx, json, csv) · `eval/` chunking comparison |
| Vector DB + embeddings + semantic search | Qdrant (`app/stores/qdrant_store.py`), `app/embeddings.py` |
| **Hybrid search** + **own algorithmic component** | `app/retrieval/` — own BM25 inverted index, own RRF / weighted fusion, own feature re-ranker + MMR, own SimHash near-dup collapse |
| LLM + RAG, multi-doc synthesis, multi-step, citations, conflicts, abstention, error handling, tool calling | `app/rag/engine.py`, `app/llm/` (function calling: `search_documents`, `get_document`, `submit_answer`, `submit_plan`) |
| Own automated evaluation (≥40 Qs, metrics, ≥2 configs, analysis) | `eval/` → [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) |
| Python backend + functional UI | FastAPI (`app/api/`) + single-page UI (`app/static/index.html`) |
| ≥50 000 documents | synthetic incident corpus (50 131 docs) + HotpotQA loader — see *Dataset* |
| Free hosting, persistence, secrets, cost awareness | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §Deployment · `Dockerfile` |
| Tests | `pytest` — 79 tests (`tests/`) |

**Live demo:** `<PUT YOUR HUGGING FACE SPACE URL HERE>` &nbsp;·&nbsp; **Admin key for uploads:** `<shared with reviewers>`
(free tier: the Space sleeps when idle and needs ~1 min to wake up and rebuild the in-memory keyword index.)

## Quick start (local, no accounts needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # defaults below work offline
# offline mode: EMBEDDING_PROVIDER=hashing  LLM_PROVIDER=heuristic  QDRANT_PATH=./data/qdrant
python -m scripts.make_sample_files                      # 50 demo files: pdf/docx/html/csv/json/txt
python -m scripts.ingest_corpus --dir data/sample_files  # or upload them in the UI
uvicorn app.api.asgi:app --reload --port 7860            # http://localhost:7860
pytest -q                                                # 79 tests
```

With real models, set in `.env`: `EMBEDDING_PROVIDER=fastembed` (BAAI/bge-small-en-v1.5, CPU) and
`LLM_PROVIDER=openai` + `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` for any OpenAI-compatible endpoint with tool calling
(Groq, Gemini's OpenAI-compatible endpoint, OpenRouter, OpenAI, a local vLLM/Ollama...). Check the provider's *current* free-tier limits before the demo.
Without an LLM key the app still works in a degraded, extractive mode (clearly labelled in the UI).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| UI shows *degraded (LLM unavailable)*; log says `Request URL is missing an 'http://' or 'https://' protocol` | `LLM_BASE_URL` in `.env` is empty or lacks `https://`. Use e.g. `LLM_BASE_URL=https://api.groq.com/openai/v1` (no quotes, no spaces). On start-up the log prints `LLM: base_url=… model=… api_key_set=yes` — if you see `using the offline heuristic reader` instead, fix the message above it. Variables already set in your shell override `.env` (`echo $env:LLM_BASE_URL` in PowerShell). |
| `LLMAuthError … authentication failed` | wrong/expired `LLM_API_KEY` |
| `429` / `rate limited` in the log | free-tier quota; wait, switch model, or set *Planner* to off in the UI (1 LLM call instead of 2) |
| `StoreMismatch` at start-up | the Qdrant data was built with another embedding model; delete `data/qdrant` (local) or re-ingest |
| Unexpected documents in answers | open the *Documents* tab and delete stray files (e.g. a questions list that was ingested by mistake) |

## Dataset

* **Synthetic incident corpus (primary, 50 131 documents).** `python -m scripts.generate_synthetic --incidents 3200 --filler 37000 --seed 21 --out data/corpus_50k`.
  Changes → incidents → services, with incident reports, post-mortems, change records, service catalogue, alert logs, runbooks, digests, plus *injected*
  exact duplicates (494), near duplicates (185), contradicting report/post-mortem pairs, missing post-mortems and never-recorded attributes.
  Seeded ⇒ byte-identical on every run; every question carries exact gold documents/answers. This mirrors the task's own example (March payment delays).
* **HotpotQA (optional, real multi-hop data).** `python -m scripts.prepare_hotpotqa --n-questions 6000` (needs internet + `pip install datasets`), then
  `python -m eval.run_eval --corpus data/corpus_hotpot`. Supporting paragraphs of a few questions are *removed* so that the right answer is to abstain.

## Evaluate

```bash
python -m scripts.generate_synthetic --incidents 120 --filler 600 --out data/corpus_small
python -m eval.run_eval --corpus data/corpus_small --chunking both --tag baseline        # offline baseline (hashing + extractive)
python -m eval.run_eval --corpus data/corpus_small --embedder fastembed --llm openai --tag real   # real models (needs .env keys)
```

## API (OpenAPI at `/docs`)

| Method & path | Purpose |
|---|---|
| `POST /api/documents` (multipart `files`, `chunking`) | add documents (admin key) |
| `POST /api/documents/text` | add pasted text |
| `PUT /api/documents/{id}` / `DELETE /api/documents/{id}` | update / delete |
| `GET /api/documents`, `GET /api/documents/{id}` | list (paged, filter) / view with chunks |
| `POST /api/search` | retrieval debugger (mode, fusion, rerank, per-hit score components) |
| `POST /api/ask` · `POST /api/ask/stream` (SSE) | investigate a question; stream pipeline steps |
| `GET /api/stats`, `/api/metrics`, `/health` | status, counters, latency percentiles, cache stats |

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — design, algorithms, decisions & trade-offs, deployment, costs, security
* [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) — methodology, metrics, results, error analysis
* [docs/AI_USAGE_REPORT.md](docs/AI_USAGE_REPORT.md) — AI tools used (fill in honestly)
* [docs/INTERVIEW_GUIDE.md](docs/INTERVIEW_GUIDE.md) — walk-through of the code for the technical interview

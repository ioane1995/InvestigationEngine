# AI Investigation Engine

A retrieval-augmented investigation system for large document collections. It is **not** a "chat with a PDF" app: a question is
planned into sub-queries, answered from several documents at once (multi-hop), every claim is cited and machine-verified against its
sources, contradictory sources are reported as such, and the system **abstains** instead of inventing facts.

**Live demo:** <https://ai-investigation-engine-u1f0.onrender.com>
Free Render plan: the service sleeps when idle. The first request after a pause takes about a minute while the in-memory keyword index is
rebuilt from Qdrant Cloud (about 76 s to load 61 768 chunks). Wait for the page to load, then ask a question.

**Corpus in the live system:** 51 264 unique documents (61 768 chunks) of a synthetic incident-investigation dataset, reproducible with
`python -m scripts.generate_synthetic --incidents 3200 --filler 38800 --seed 21`.

**Try these questions** (gold answers are in `eval/questions/scale_50k_seed21.jsonl`):

- Who authored change CHG-16378? *(single document)*
- Which team owns the service that was modified by the root cause change of INC-2622? *(multi-hop: incident → change → service catalogue)*
- Who approved change CHG-18794? *(approver never recorded: the system should abstain)*
- Which change was the root cause of INC-0044? *(incident report and post-mortem disagree: the system should report both)*

**Uploading / deleting documents** requires an admin key. It is given to the reviewers in the submission message and is deliberately not
published here, because anyone holding it can delete documents.

## Requirements map

| Requirement (task PDF) | Where it lives |
|---|---|
| ≥3 file formats, metadata, chunking, dedup, add/update/delete, 2 chunking strategies compared | `app/ingestion/` (7 formats: txt, md, html, pdf, docx, json, csv) · chunking comparison in `eval/` |
| Vector DB + embeddings + semantic search | Qdrant (`app/stores/qdrant_store.py`), `app/embeddings.py` |
| Hybrid search + own algorithmic component | `app/retrieval/`: own BM25 inverted index, own RRF / weighted fusion, own feature re-ranker + MMR, own SimHash near-duplicate collapse |
| LLM + RAG, multi-doc synthesis, multi-step, citations, conflicts, abstention, error handling, tool calling | `app/rag/engine.py`, `app/llm/` (function calls: `submit_plan`, `search_documents`, `get_document`, `submit_answer`) |
| Own automated evaluation (≥40 questions, metrics, ≥2 configurations, analysis) | `eval/` → [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) · questions in `eval/questions/` |
| Python backend + functional UI | FastAPI (`app/api/`) + single-page UI (`app/static/index.html`) |
| ≥50 000 documents | synthetic incident corpus (51 264 unique docs) + HotpotQA loader, see *Dataset* |
| Free hosting, persistence, secrets, cost awareness | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §5–6 · `Dockerfile` · `render.yaml` |
| Automated tests | `pytest -q`: 115 tests in `tests/` (no network or API keys needed) · CI in `.github/workflows/ci.yml` |

## Architecture in short

```
upload / jsonl → loaders → chunker → dedup → embed → Qdrant (vectors + payload)
                                                   └→ in-RAM BM25 index + chunk mirror
question → plan (LLM tool) → hybrid search per sub-query → RRF merge → hop expansion → evidence gate (abstain?)
        → LLM with tools (search_documents / get_document / submit_answer)
        → claim verification → conflict detection → Answer {status, claims + citations, conflicts, missing_info, trace, usage}
```

Dependency direction is strictly downward (API → RAG → retrieval → store); `app/container.py` is the only place that wires concrete classes.
Design decisions, trade-offs, deployment, costs and security are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Deployment:** Docker image on Render (free web service) + Qdrant Cloud (free cluster) for persistent vectors and the document registry.
All secrets are environment variables (`sync: false` in `render.yaml`); `.env` is git-ignored. Because the free plan has 512 MB RAM, query
embeddings are served through an API for the same BGE model rather than a local ONNX model (see ARCHITECTURE §5).

## Quick start (local, no accounts needed)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
cp .env.example .env            # offline mode: EMBEDDING_PROVIDER=hashing  LLM_PROVIDER=heuristic  QDRANT_PATH=./data/qdrant
python -m scripts.make_sample_files                      # 50 demo files: pdf/docx/html/csv/json/txt
python -m scripts.ingest_corpus --dir data/sample_files  # or upload them in the UI
uvicorn app.api.asgi:app --reload --port 7860            # http://localhost:7860
pytest -q
```

With real models, set in `.env`: `EMBEDDING_PROVIDER=fastembed` (BAAI/bge-small-en-v1.5, CPU) and `LLM_PROVIDER=openai` with
`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` for any OpenAI-compatible endpoint with tool calling (Groq, Gemini's compatible endpoint,
OpenRouter, OpenAI, local vLLM/Ollama). The live demo uses Groq with `openai/gpt-oss-120b`. Check the provider's *current* model list and
free-tier limits first, because model names are retired over time.
Without an LLM key the app still works in a clearly labelled degraded, extractive mode.

## Dataset

- **Synthetic incident corpus (primary, 51 958 generated documents, 51 264 unique after exact duplicates are rejected at ingest).**
  `python -m scripts.generate_synthetic --incidents 3200 --filler 38800 --seed 21 --out data/corpus_50k`.
  Changes → incidents → services, with incident reports, post-mortems, change records, a service catalogue, alert logs, runbooks and digests,
  plus *injected* exact duplicates, near duplicates, contradicting report/post-mortem pairs, missing post-mortems and never-recorded attributes.
  It is seeded, so it is byte-identical on every run, and every question carries exact gold documents and answers. It mirrors the task's own
  example (payment-processing delays).
- **HotpotQA (optional, real multi-hop data).** `python -m scripts.prepare_hotpotqa --n-questions 6000` (needs internet and `pip install datasets`),
  then `python -m eval.run_eval --corpus data/corpus_hotpot`. Supporting paragraphs of some questions are removed so that abstaining is correct.

## Evaluation

```bash
python -m scripts.generate_synthetic --incidents 120 --filler 600 --out data/corpus_small
python -m eval.run_eval --corpus data/corpus_small --chunking both --tag baseline                 # offline baseline
python -m eval.run_eval --corpus data/corpus_heldout --embedder fastembed --llm openai \
       --chunking structure --configs full --sample 25 --pause 15 --retry-degraded 2 --resume --tag real_120b
python -m eval.tables eval/results/real_20b.json eval/results/real_120b.json
```

54 questions per corpus (simple, multi-hop, natural-language multi-hop, unanswerable, conflicting), 7 retrieval/pipeline configurations,
2 chunking strategies, 2 embedders and 2 LLMs. The question sets with gold documents and answers are committed in `eval/questions/`.
Metrics: Recall@1/5/10, MRR, evidence recall, answer correctness, abstention recall/precision,
conflict-detection recall, citation precision/recall, faithfulness, latency and LLM calls. Full methodology, results and error analysis:
[docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md).

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

## Known limitations

- The free Render plan sleeps when idle and has 512 MB RAM, so the first request is slow and only one worker runs.
- The BM25 index lives in RAM and is rebuilt from Qdrant at start-up; very large corpora need more memory.
- Query embeddings come from the Hugging Face Inference API free tier. If its credit runs out, the token lacks permission, or the service is
  unreachable (for example HTTP 402/403/429/5xx), semantic search is skipped and the system answers from keyword search only, with a visible warning.
- Free LLM tiers are rate limited (for example Groq's tokens-per-minute and tokens-per-day caps) and model names can be retired; the pipeline
  retries, paces requests, and falls back to a labelled offline reader, which is reported as a *degraded* answer.
- Evaluation uses a synthetic corpus with rule-based scoring; it measures this pipeline well but is not a substitute for human review.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| UI shows *degraded (LLM unavailable)*; log says `Request URL is missing an 'http://' or 'https://' protocol` | `LLM_BASE_URL` is empty or lacks `https://`. Use e.g. `https://api.groq.com/openai/v1` (no quotes). Variables already set in your shell override `.env` (`echo $env:LLM_BASE_URL` in PowerShell). |
| `404 Not Found` from `.../chat/completions` | the model name no longer exists at the provider. List models with `GET /openai/v1/models` and set `LLM_MODEL` to one that is listed (e.g. `openai/gpt-oss-120b`). |
| `403 Forbidden` or `402` from the embeddings URL; log says `dense retrieval unavailable` | the Hugging Face token lacks the "Inference Providers" permission or its free credit is used up. Create a new token with that permission and set `EMBEDDING_API_KEY`. |
| `LLMAuthError … authentication failed` | wrong or expired `LLM_API_KEY` |
| `429` / `rate limited` in the log | free-tier quota: wait, switch model, set *Planner* to off in the UI (1 LLM call instead of 2), or raise `--pause` in evaluations |
| `StoreMismatch` at start-up | the Qdrant data was built with another embedding model: delete `data/qdrant` (local) or re-ingest |
| Unexpected documents in answers | open the *Documents* tab and delete stray files (e.g. a questions list ingested by mistake) |

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): design, algorithms, decisions, deployment, costs, security
- [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md): methodology, metrics, results, error analysis
- [docs/AI_USAGE_REPORT.md](docs/AI_USAGE_REPORT.md): AI tools used and what I did myself

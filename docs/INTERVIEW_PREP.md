# Interview preparation (no AI allowed in the practical part)

Section 8 of the task: you will explain your architecture, walk through the RAG pipeline step by step, discuss a specific piece of code, **find and fix bugs, and add or change functionality on the spot**. The way to prepare is to *do* those things, not to read about them. Budget about 6 hours: §1-§5 reading (2 h), §6 questions out loud (1 h), §7-§8 hands-on drills (3 h).

---

## 1. The 90-second pitch (say it aloud until it is smooth)

"It is an investigation engine over incident documents, not a chat-with-PDF. Documents in seven formats (txt, md, html, pdf, docx, json, csv) are loaded, split into chunks with a structure-aware chunker, de-duplicated by content hash and SimHash, embedded, and stored in Qdrant, while an in-memory BM25 index I wrote mirrors the chunks. A question goes through an agentic RAG pipeline: an optional planner splits it into sub-queries, hybrid retrieval fuses dense and BM25 rankings with reciprocal rank fusion, my feature re-ranker with MMR picks diverse evidence, a hop step follows identifiers such as INC-0012 to the related change CHG-0345, and a gate refuses before calling the LLM if the question's key terms are absent from the evidence. The LLM then works through tool calls (`search_documents`, `get_document`, `submit_answer`) and returns a structured answer with claims and citations. Every claim is verified against the evidence it cites; unsupported claims are flagged and a fully unsupported answer is downgraded to *insufficient evidence*. Candidate conflicts are also detected algorithmically. Retrieved text is treated as untrusted: injected instructions are quarantined and the output is guarded. It is evaluated by my own harness on a synthetic corpus with ground truth, and deployed on Render with Qdrant Cloud."

---

## 2. Code map: say one sentence about each file

| File | Responsibility |
|---|---|
| `app/config.py` | `Settings` from environment variables; no secrets in code |
| `app/container.py` | wires store, embedder, corpus, ingestion, engine (dependency injection) |
| `app/models.py` | plain dataclasses `RawDocument`, `Chunk`, `Hit` |
| `app/text.py` | tokenizer + stemmer, `identifiers()` regex for ids like `INC-0012`, **SimHash** (`simhash`, `hamming`) |
| `app/ingestion/loaders.py` | `load_bytes`: one loader per format, `content_hash`, `stable_doc_id` |
| `app/ingestion/chunking.py` | `FixedWindowChunker` and `StructureAwareChunker` (the two strategies compared) |
| `app/ingestion/service.py` | `_plan` (unchanged / duplicate / new), `ingest`, `ingest_many`, update and `delete` |
| `app/stores/qdrant_store.py` | Qdrant wrapper; stores the embedder identity and raises `StoreMismatch` if it differs |
| `app/embeddings.py` | `HashingEmbedder` (offline), `FastEmbedEmbedder`, `OpenAICompatEmbedder` (API), `CachedQueryEmbedder` |
| `app/retrieval/bm25.py` | **own BM25** inverted index with add/remove/filters |
| `app/retrieval/fusion.py` | **own RRF** and weighted min-max fusion |
| `app/retrieval/rerank.py` | **own feature re-ranker + MMR**; `assess_evidence` drives the abstention gate |
| `app/retrieval/hybrid.py` | `HybridRetriever.search`: dense + BM25, fusion, near-duplicate collapse, re-rank; falls back to BM25 if embeddings fail |
| `app/rag/engine.py` | the pipeline (plan, retrieve, hop, gate, reason, verify, finalize) |
| `app/rag/verify.py` | claim verification (`support_score`, `verify_claims`) |
| `app/rag/conflicts.py` | algorithmic candidate-conflict detection from `Key: value` lines |
| `app/rag/safety.py` | prompt-injection defences |
| `app/rag/prompts.py` | system prompt and tool schemas |
| `app/llm/openai_compat.py` | HTTP client: retries, 429 back-off, error classes, salvages rejected tool calls |
| `app/llm/heuristic.py` | offline fallback reader (used when the LLM fails; answers are marked `degraded`) |
| `app/api/main.py` | FastAPI: documents CRUD, search, ask, SSE streaming, metrics, admin-key dependency, rate limit, answer cache |
| `eval/` | question generator, metrics, runner, tables, injection evaluation |
| `scripts/` | corpus generation, ingestion, scale benchmark, embedding-compat check, deployment measurement |

---

## 3. The RAG pipeline, step by step (this is the interview's central question)

`InvestigationEngine.ask()` in `app/rag/engine.py`:

1. **Plan** (optional, `use_planner`): one LLM tool call `submit_plan` returns up to 4 sub-queries. Any LLM error falls back to the raw question and adds a warning.
2. **Retrieve**: for each sub-query `HybridRetriever.search`: dense top-40 from Qdrant and BM25 top-40, fused by RRF, near-duplicates collapsed, then re-ranked to `top_k=8`. Sub-query results are merged with my own weighted RRF (the original question counts 1.5x).
3. **Hop**: `_bridge_entities` takes identifiers in the best hits that are *not* in the question (e.g. `CHG-0345` found inside `INC-0012`) and searches for them (`hops=1`, at most 4 entities, 2 hits each). This is how a question that needs two documents reaches the second one.
4. **Gate**: `assess_evidence` computes the IDF-weighted share of query terms present in the top-5 evidence and lists question identifiers found nowhere. If coverage < 0.30 or any identifier is missing, the system answers `insufficient_evidence` **without calling the LLM** (saves cost and prevents hallucination).
5. **Reason**: `find_candidate_conflicts` runs first; its result is given to the LLM as a hint. The LLM runs a tool loop (max 2 rounds + a forced final round): it may call `search_documents` or `get_document` to fetch more evidence and must end with `submit_answer(status, answer, claims[{text, evidence_ids}])`. Malformed tool calls produce an `ERROR:` tool result so the model can retry; invalid arguments are rejected by `_parse_submission`.
6. **Verify**: `verify_claims` drops invented evidence ids and scores each claim against the evidence it cites. Numbers and identifiers in the claim must appear verbatim (else score 0); other content words need >= 60 % overlap. If every claim is unsupported the answer is downgraded to `insufficient_evidence`.
7. **Finalize**: attach candidate conflicts even if the LLM missed them, build the citation panel (original chunk text), apply the output guard, record timings, tokens and cost.
8. **Failure handling**: any LLM error (429, timeout, malformed output) falls back to the offline `HeuristicLLM`, and the answer is flagged `degraded` so evaluations can exclude it.

Be ready to draw this as a diagram and to say what happens at each step when the LLM is down.

---

## 4. Algorithms: formulas and the *why*

* **BM25** (`bm25.py`): `score = sum over query terms of idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))`, `k1 = 1.5`, `b = 0.75`. IDF is Robertson-Sparck-Jones, `ln(1 + (N - n + 0.5)/(n + 0.5))`, always >= 0. A term that appears nowhere gets the maximum IDF; that is what makes the gate expensive to fool with a non-existent incident id. *Why own BM25 rather than SQL full-text?* The task forbids relying only on libraries and wants a hybrid; owning it also lets me filter by metadata and delete documents instantly.
* **RRF** (`fusion.py`): `score(d) = sum_i w_i / (k + rank_i(d))`, `k = 60`. Needs no score calibration between cosine similarity and BM25. Weighted min-max fusion is implemented as the alternative and was compared.
* **Feature re-ranker** (`rerank.py`): linear combination of six interpretable features: fused score 0.35, IDF-weighted query coverage 0.25, identifier match 0.20, bigram overlap 0.08, proximity 0.07 (matched terms / smallest window containing them), title match 0.05. Then **MMR** with `lambda = 0.8` and Jaccard similarity on token sets so the evidence is not five copies of the same paragraph.
* **SimHash** (`text.py`): 64-bit fingerprint over unigrams + bigrams, weights are term counts; two texts are near-duplicates when the Hamming distance is <= 3 bits (`near_dup_bits`). Exact duplicates are caught earlier by `content_hash` in `_plan`; near-duplicates are *collapsed at query time* and the other document is kept in `also_doc_ids` so the citation still names every source.
* **Conflict detection** (`conflicts.py`): extract `Key: value` lines, normalise values (e.g. "1 hour" = 60 min), group by (entity, field); a conflict candidate needs two different documents with different values for a field the question actually asks about. The LLM adjudicates, because the same field can legitimately change over time.
* **Claim verification**: see §3, step 6.
* **Injection defence** (`safety.py`): canonicalise (NFKC, strip zero-width characters, fold look-alike letters) -> 24 detection rules (weak signals only count in pairs) -> quarantine instruction-like lines **and their adjacent prose lines** from the model's copy only -> neutralise tag/template break-outs -> guard the output (markdown images, unknown URLs, key patterns, prompt text) -> claim verification as the last layer.

---

## 5. Design decisions you must be able to defend

| Decision | Reason | Honest trade-off |
|---|---|---|
| Qdrant Cloud | real vector DB, free tier, persistent across restarts | a network dependency; free-tier limits |
| BM25 index in process memory | fast, own code, rebuilt from Qdrant on start (the cold start) | **single worker only**; memory grows with corpus; cold start after sleep |
| RRF as default fusion | no calibration needed | ignores score magnitudes |
| Structure-aware chunking (default) vs fixed window | keeps `Key: value` lines intact, which conflict detection needs; compared in the evaluation | more complex; chunk sizes vary |
| Agentic loop with tool calling | multi-step questions, model fetches missing evidence | more LLM calls, cost, latency (about 1.2 calls per question measured) |
| Gate before the LLM | cheaper and safer abstention | a purely lexical gate can refuse paraphrased questions (measured as a weakness) |
| Embeddings via API in production | 512 MB RAM could not hold the local model | extra dependency; mitigated by `EMBEDDING_STORE_ID`, the compat check and BM25 fallback |
| Synthetic corpus with ground truth | exact, reproducible metrics for retrieval, abstention and conflicts | templated text is easier than real prose; HotpotQA loader exists but was not run here |
| Answers marked `degraded` on fallback | evaluation stays honest | the demo can show weaker answers when Groq is rate-limited |

---

## 6. Questions to answer out loud (one minute each)

1. Why is this not "chat with PDF"? (multi-source reasoning, hop, conflicts, abstention, verification, evaluation)
2. Walk me through a request from `POST /api/ask` to the JSON response.
3. Explain BM25. What happens when `b = 0`? When `k1 = 0`?
4. Why RRF instead of adding scores? What does `k = 60` do?
5. How does the system decide to refuse? Give one example where it refuses and one where it should not.
6. How do you detect contradictions? What are the false-positive risks?
7. What stops the model from inventing a citation id? (`verify_claims` drops unknown ids; prompt; schema)
8. What happens if the LLM returns malformed JSON in a tool call? If it returns nothing? If Groq returns 429?
9. What happens when two uploaded documents are identical? Nearly identical? When a document is updated? Deleted?
10. Why a single Uvicorn worker? How would you scale?
11. How are secrets protected? (env vars, `.env` git-ignored, `sync: false` in `render.yaml`, admin key on write endpoints, key rotation after exposure)
12. How would an attacker abuse the demo and what limits exist? (per-IP rate limit, admin key, answer cache, output guard)
13. Your evaluation shows abstention recall 0.60. Why, and how would you improve it? (lexical gate misses semantic absence; add a dense-similarity floor or an LLM verdict; calibrate `min_coverage` on a dev split)
14. Why did the evaluated model differ from the deployed one? What would you do about it? (evaluate the deployed model; report both)
15. How does the injection defence work, and what does it *not* catch? (subtle semantic attacks with no trigger words; quarantine can remove real prose next to a payload)
16. What did the worst-case model experiment show and what can you not conclude from it?
17. How would you handle 10x more documents? (sharded or persisted BM25, filtered dense search, batch embedding, caching)
18. What is the cost per question? Where do the tokens go? (system prompt + tools + evidence; planner off saves a call)
19. Which of your metrics is the least trustworthy and why? (answer correctness is rule-based; small n per question type; one degraded run)
20. What would you do with another week?

---

## 7. Live-modification drills (do each once, with tests, committed)

Time yourself: an interviewer will give you about 10-20 minutes per task.

1. **Add a metadata filter to `/api/search`** (e.g. `format=pdf`). Hint: `BM25Index.search` already takes `filters`; check `QdrantStore.dense_search` and the request model in `main.py`. Add a test.
2. **Add a new file format** (`.xml` or `.log`): `SUPPORTED`, a `_load_xml`, a branch in `load_bytes`; add a sample file and a test.
3. **Change the chunk size** to 150 tokens and re-run the retrieval-only evaluation (`ask=False`); explain the effect on recall@5.
4. **Change the RRF constant** `k` from 60 to 10 in `RetrievalConfig`; predict, then measure, which ranks change.
5. **Add a recency feature to the re-ranker**: new key in `DEFAULT_WEIGHTS`, compute it from a `date` metadata field, re-normalise the weights so they still sum to 1.
6. **Make the abstention threshold configurable per request** (`min_coverage` in the `/api/ask` body) and test both outcomes.
7. **Add one injection rule** (e.g. "pretend this is a test") to `safety.py` and add it to `ATTACKS` plus a benign counter-example; show the benign set still passes.
8. **Add an endpoint** `GET /api/documents/{id}/chunks`; require no admin key; test 404.
9. **Lower the verification threshold** from 0.6 to 0.4 and show which claim in a test flips from unsupported to supported; explain the risk.
10. **Stop returning degraded answers in `/api/ask`** (return HTTP 503 instead) behind a setting.

---

## 8. Find-the-bug drills (mutation practice)

On a scratch branch, introduce **one** bug at a time, run `python -m pytest -q`, and note whether a test catches it. A bug nobody catches is a missing test: write it. Then practise *finding* each bug from the symptom alone (have someone else introduce it, or do it a day later).

| Mutation | Symptom to recognise | Where |
|---|---|---|
| RRF enumerates ranks from 0 instead of 1 | top result dominates; `k` loses meaning | `fusion.py` |
| BM25 `idf` uses `df` instead of `n - df` | rare terms score *lower* than common ones | `bm25.py` |
| `remove_doc` forgets to delete postings | deleted documents still appear in search | `bm25.py` |
| `_collapse` compares `hamming(...) >= bits` | almost nothing is collapsed (or everything) | `hybrid.py` |
| `support_score` ignores the numbers/ids check | a claim with a wrong number passes verification | `verify.py` |
| `_plan` compares `existing.content_hash != h` | unchanged re-uploads are re-indexed every time | `service.py` |
| `render()` skips `neutralize` | an evidence block can close its own tag | `engine.py`, `safety.py` |
| `quarantine` drops the `_STRUCTURED` guard | bullets next to a payload vanish with it | `safety.py` |
| `assess_evidence` uses `or` instead of `and not miss_ids` | questions about non-existent incident ids get answered | `rerank.py` |
| cache key ignores config | an answer for one configuration is returned for another | `main.py` |

---

## 9. Weaknesses to volunteer (honesty scores points)

* Abstention recall 0.60 on the real-model run; the gate is purely lexical.
* The real-model run had 8 % degraded answers (Groq rate limits); the numbers are a blend until repeated with `--retry-degraded 2`.
* Evaluated models (gpt-oss) differ from the deployed one (llama-3.3-70b) until you evaluate it.
* Synthetic, templated corpus; HotpotQA support exists but was not run.
* Single worker, in-memory BM25, cold start after the free plan sleeps.
* Injection defences: pattern-based, so semantic attacks without trigger words get through; quarantine can remove real prose next to a payload; real-model attack-success numbers are small-sample and not reproducible across runs.

---

## 10. The night before

* Run the whole thing locally once (`pytest`, one question in the UI, one upload, one delete) so nothing is a surprise.
* Open the live URL; ask one question; confirm the cold-start behaviour you documented.
* Rotate any key that was ever pasted into a chat or committed; confirm `.env` is not in git: `git log --all -- .env` must print nothing.
* Be able to open `engine.py`, `rerank.py`, `bm25.py`, `safety.py` and explain every line without scrolling for help.

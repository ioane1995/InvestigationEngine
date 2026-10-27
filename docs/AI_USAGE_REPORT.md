# AI usage report

> **Template prepared with the first draft — you must edit it so that it is true.** The assignment explicitly evaluates what you did yourself, and in the
> interview you may not use AI. Keep only statements that are accurate for *you*, and make sure you can explain every file below without help.

## 1. Tools used

| Tool | What for | Where |
|---|---|---|
| Claude (Anthropic) | generated the **first complete draft** of the repository: architecture proposal, Python modules, tests, UI, synthetic-data generator, evaluation harness and the documents in `docs/`; ran the test suite and the offline evaluations in a sandbox and fixed the bugs they exposed | whole repo |
| *(add: your IDE assistant / other LLMs, if any)* | | |

## 2. What AI did (draft stage)

* Proposed the architecture (Qdrant + own BM25 + RRF + re-ranker; agentic RAG with tool calling; verification & abstention; synthetic corpus for ground-truth evaluation).
* Wrote the code and 79 unit/integration tests; iterated on failures it found by running the offline evaluation (examples: the extractive reader was matching identifier *parts* like `chg`/`0414` as content words; heading text leaked into evidence; the fixed-window chunker destroyed line structure which broke conflict detection; "how long … last?" questions have no lexical overlap with the `Duration` field).
* Produced the evaluation report from real measured output (`eval/results/*.json`).

## 3. What was **not** done / needs you

* **No real LLM or neural embedding model was run by the AI** (sandbox had no network/API keys). Real-model evaluation numbers, the Live Demo deployment (Hugging Face Space + Qdrant Cloud), the HotpotQA run, and the 50k ingestion into your cloud Qdrant must be done by you (see `docs/ARCHITECTURE.md` §5 and `docs/EVALUATION_REPORT.md` §8).
* Project history: commit in meaningful steps yourself (e.g. one commit per module/feature as you review it) — do not squash everything into one commit.

## 4. What I changed / verified myself  *(fill in — be specific and honest)*

- [ ] Read and understood every module (use `docs/INTERVIEW_GUIDE.md`).
- [ ] Changes I made on my own (examples to replace with real ones): tuned `RerankWeights` after looking at failures; changed chunk size; added/removed a loader; rewrote prompt; …
- [ ] Bugs I found and fixed myself: …
- [ ] Decisions I disagreed with the draft on and why: …
- [ ] Real-model evaluation I ran (model names, dates, results): …
- [ ] Deployment steps I performed: …

## 5. Why I can defend the code

Key algorithms are small and readable (BM25 ≈ 100 lines, RRF 10 lines, SimHash 20 lines, sliding-window proximity 20 lines, claim verifier 30 lines). Every design choice has a stated reason and a measured effect in the evaluation report.

# AI usage report

> **How to finish this document (delete this box before submitting).** Everything not in square brackets is a fact that can be verified from the repository
> history and the evaluation files. Every `[YOU: ...]` item can only be answered by you; replace it with your own true statement or delete the line.
> A claim you cannot defend in the interview does more damage than a short report. Section 8 of the task states that AI help is allowed but that you are
> fully responsible for the code and will explain and modify it live without AI.

## 1. Tools used

| Tool | What for |
|---|---|
| Claude (Anthropic), chat and sandbox | First complete draft of the repository (architecture proposal, Python modules, tests, UI, synthetic-data generator, evaluation harness, documents); later sessions: prompt-injection hardening, attack test set, evaluation scripts, deployment-measurement script, documentation corrections |
| Groq API (`gpt-oss-20b`, `gpt-oss-120b`, `llama-3.3-70b-versatile`) | The LLMs *inside* the product (tool calling, answer generation), not development tools |
| `BAAI/bge-small-en-v1.5` via fastembed (local) and Hugging Face router (deployment) | Embedding model |
| [YOU: IDE assistant / other tools, or "none"] | |

## 2. What the AI did

**Draft stage.** Proposed the architecture (Qdrant + own BM25 + reciprocal rank fusion + own feature re-ranker with MMR; agentic RAG with tool calling; claim verification and abstention; synthetic corpus with ground truth); wrote the modules and the first test suite; found and fixed bugs by running the offline evaluation (e.g. the extractive reader matched identifier *parts*; heading text leaked into evidence; the fixed-window chunker destroyed line structure and broke conflict detection).

**Prompt-injection hardening (later session).** Rewrote `app/rag/safety.py` (canonicalisation, 24 detection rules, quarantine of instruction-like text, tag neutralisation, output guard), added the 38-attack set (`eval/injection_cases.py`), the end-to-end attack evaluation with defences on/off and a worst-case "obedient" model (`eval/injection_eval.py`), 13 unit tests and 2 pipeline tests.
Errors made by the AI that the evaluation exposed and that were corrected, which are recorded in `docs/EVALUATION_REPORT.md`:
* the first unit tests used unchunked text and reported 36/36; through the real chunker, 3 payloads leaked (payloads continue across the sentence-per-line chunk boundary); the quarantine was changed and the tests now run through the real chunker;
* the first worst-case model cited every claim to evidence `E1`, which understated utility (0.868 instead of 0.947);
* an "only real-model attack success" statement was withdrawn because the result did not reproduce on a second run.

## 3. What the AI could not do

It never had network access or API keys: all runs against real LLMs, the real embedding model, Render, Groq and Qdrant Cloud were performed by me on my machine and the logs were pasted back for analysis.

## 4. What I did myself

*(Items below are things the project history shows. Keep only those that are true for you, and add commit hashes.)*

- Installed and ran the code; the suite is currently **110 tests, all passing** on my machine. Generated the synthetic corpus and ingested 51 264 unique documents (61 768 chunks) into Qdrant Cloud.
- Ran every evaluation with real models (`real_20b`, `real_120b`, the injection runs on `gpt-oss-20b`, `gpt-oss-120b` and the worst-case model) and returned the logs. I decided how to spend the free-tier quota (e.g. switching model after the 200 000 tokens-per-day limit of `gpt-oss-120b` was exhausted; using the free worst-case model for the full on/off comparison).
- Debugged the real LLM integration from the logs: a malformed `LLM_BASE_URL`, an unavailable model name, `tool_use_failed` and `output_parse_failed` errors, HTTP 429 limits. These drove: non-retryable configuration errors, an output-token floor for reasoning models, salvaging rejected tool calls, request pacing, resumable evaluation, retry of answers that fell back to the offline reader.
- Deployed to Render with Qdrant Cloud, configured every secret as an environment variable, and switched query embeddings to an API after the 512 MB free plan ran out of memory with the local model; added `EMBEDDING_STORE_ID` so the store accepts API vectors of the same model.
- Detected that results can be invalidated by stale files: one injection run executed with an outdated `safety.py`; I now verify file sizes/test results before running evaluations. [YOU: keep only if true]
- **Own code changes:** [YOU: list each concrete change you made yourself with its commit, e.g. "changed chunk size from 200 to 150 and re-ran the evaluation", "tuned DEFAULT_WEIGHTS of the re-ranker", "fixed ...". If the list is empty, do the exercises in `docs/INTERVIEW_PREP.md` §7 first and commit them: that is both preparation and honest history.]
- **Things I disagreed with or changed in the AI's proposals:** [YOU: list, or write "none"].
- **Chose the models and why:** [YOU: e.g. "gpt-oss-120b for evaluation because ...; llama-3.3-70b for the demo because of its higher free-tier token budget"].

## 5. What I can explain without help

[YOU: tick only what is true after working through `docs/INTERVIEW_PREP.md`] The pipeline end to end (plan, hybrid retrieval, fusion, re-ranking, hop expansion, abstention gate, tool-calling loop, claim verification, conflict detection, output guard), and the algorithms I am responsible for: BM25 scoring, reciprocal rank fusion, the feature re-ranker with MMR, SimHash near-duplicate detection, the injection quarantine.

## 6. Verification of AI output

Nothing the AI wrote was accepted on its word: every module has tests; the retrieval and answer quality are measured on a held-out synthetic corpus with ground truth; the attack set measures the injection defences; and where a measurement contradicted the AI's own claims (section 2), the claim was corrected in the report rather than the measurement.

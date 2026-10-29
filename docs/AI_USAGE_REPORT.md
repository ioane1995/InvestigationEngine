# AI usage report

## 1. Tools used
| Tool | What for |
|---|---|
| Claude (Anthropic), chat/sandbox | First complete draft of the repository (architecture proposal, Python modules, tests, UI, synthetic-data generator, evaluation harness, documents); later: prompt-injection hardening, attack test set, evaluation and deployment-measurement scripts, documentation corrections, final review of the repository against the task PDF, CI workflow, git/PowerShell commands, and help diagnosing deployment errors |
| Groq API (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`; earlier `llama-3.3-70b-versatile`, since retired by Groq) | LLMs inside the product (tool calling, answer generation), not development tools |
| BAAI/bge-small-en-v1.5 (fastembed locally, Hugging Face router in deployment) | Embedding model |
| VS code | For Coding, file management and docs.

## 2. What the AI did
Draft stage: proposed the architecture (Qdrant + own BM25 + RRF + own feature re-ranker with MMR; agentic RAG with tool calling; claim verification and abstention; synthetic corpus with ground truth), wrote the modules and first test suite, and fixed bugs found by the offline evaluation (extractive reader matching identifier parts; heading text leaking into evidence; fixed-window chunker destroying line structure and breaking conflict detection).

Prompt-injection hardening: rewrote `app/rag/safety.py`, added the 38-attack set and the on/off evaluation, plus unit and pipeline tests. Errors in the AI's work that the evaluation exposed and that were corrected (recorded in EVALUATION_REPORT.md): unit tests used unchunked text (36/36 became 33/36 through the real chunker); the worst-case model cited every claim to E1; one "real-model attack success" claim was withdrawn because it did not reproduce.

Final review: It wrote the CI workflow and README/documentation corrections, and explained the deployment errors found in the Render logs. I applied and verified these changes.

## 3. What the AI could not do
It had no network access or API keys. All runs against real LLMs, the real embedder, Render, Groq and Qdrant Cloud were done by me; logs and command output were pasted back for analysis.

## 4. What I did myself
- Installed and ran the code; test suite: 115 tests, all passing, 1 warning(`pytest -q`).
- Generated the synthetic corpus and ingested 51,264 unique documents (61,768 chunks) into Qdrant Cloud.
- Ran the real-model evaluations and injection runs; decided how to spend free-tier quota (switched model after the `gpt-oss-120b` daily token limit was reached).
- Debugged the real LLM integration from logs, with AI assistance: malformed `LLM_BASE_URL`, unavailable model name, `tool_use_failed` / `output_parse_failed`, HTTP 429.
- Deployed to Render + Qdrant Cloud, set every secret as an environment variable, and moved query embeddings to an API after the 512 MB plan ran out of memory.
- Found from the live logs that Groq returned HTTP 404 for `llama-3.3-70b-versatile`, confirmed with a model-list test against my own key that the model had been retired and that `openai/gpt-oss-120b` and `openai/gpt-oss-20b` worked, and switched the deployment and documents to `openai/gpt-oss-120b`.
- Why I chose these models: I use `openai/gpt-oss-120b` for the demo because it supports tool calling, it is the model my evaluations were run with, and the previously deployed `llama-3.3-70b-versatile` no longer exists on Groq. `openai/gpt-oss-20b` is the fallback when the 120b daily token limit is reached.

## 5. Verification of AI output
Every module has tests; retrieval and answer quality are measured on a held-out synthetic corpus with ground truth; the attack set measures the injection defences. Where a measurement contradicted an AI claim (section 2), the claim was corrected rather than the measurement.

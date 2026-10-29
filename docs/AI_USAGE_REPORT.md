# AI usage report

## 1. Tools used
| Tool | What for |
|---|---|
| Claude (Anthropic), chat/sandbox | First complete draft of the repository (architecture proposal, Python modules, tests, UI, synthetic-data generator, evaluation harness, documents); later: prompt-injection hardening, attack test set, evaluation and deployment-measurement scripts, documentation corrections |
| Groq API (gpt-oss-20b, gpt-oss-120b, llama-3.3-70b-versatile) | LLMs inside the product (tool calling, answer generation), not development tools |
| BAAI/bge-small-en-v1.5 (fastembed locally, Hugging Face router in deployment) | Embedding model |
| [FILL: any IDE assistant / other tool, or write "none"] | |

## 2. What the AI did
Draft stage: proposed the architecture (Qdrant + own BM25 + RRF + own feature re-ranker with MMR; agentic RAG with tool calling; claim verification and abstention; synthetic corpus with ground truth), wrote the modules and first test suite, and fixed bugs found by the offline evaluation (extractive reader matching identifier parts; heading text leaking into evidence; fixed-window chunker destroying line structure and breaking conflict detection).

Prompt-injection hardening: rewrote `app/rag/safety.py`, added the 38-attack set and the on/off evaluation, plus unit and pipeline tests. Errors in the AI's work that the evaluation exposed and that were corrected (recorded in EVALUATION_REPORT.md): unit tests used unchunked text (36/36 became 33/36 through the real chunker); the worst-case model cited every claim to E1; one "real-model attack success" claim was withdrawn because it did not reproduce.

## 3. What the AI could not do
It had no network access or API keys. All runs against real LLMs, the real embedder, Render, Groq and Qdrant Cloud were done by me; logs were pasted back for analysis.

## 4. What I did myself
- Installed and ran the code; test suite: [FILL: number] tests, all passing (`pytest -q`).
- Generated the synthetic corpus and ingested 51,264 unique documents (61,768 chunks) into Qdrant Cloud.
- Ran the real-model evaluations and injection runs; decided how to spend free-tier quota (switched model after the gpt-oss-120b daily token limit was reached).
- Debugged the real LLM integration from logs: malformed `LLM_BASE_URL`, unavailable model name, `tool_use_failed` / `output_parse_failed`, HTTP 429.
- Deployed to Render + Qdrant Cloud, set every secret as an environment variable, and moved query embeddings to an API after the 512 MB plan ran out of memory.
- **Own code changes:** [FILL: each change you personally made + commit hash, e.g. "changed chunk size 200 -> 150, re-ran eval, commit abc123". Do the exercises in docs/INTERVIEW_PREP.md first if this is empty.]
- **Where I disagreed with / changed the AI's proposals:** [FILL or "none"]
- **Why I chose these models:** [FILL, e.g. "llama-3.3-70b for the demo because of its larger free-tier token budget"]

## 5. What I can explain without help
[FILL: keep only what is true after working through docs/INTERVIEW_PREP.md]

## 6. Verification of AI output
Every module has tests; retrieval and answer quality are measured on a held-out synthetic corpus with ground truth; the attack set measures the injection defences. Where a measurement contradicted an AI claim (section 2), the claim was corrected rather than the measurement.

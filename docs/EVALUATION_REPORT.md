# Evaluation report

> **Read this first (honesty note).** The numbers below were produced in a sandbox **without internet and without an LLM API key**, so every
> run uses the project's *offline baseline stack*: the own hashing embedder (lexical, not semantic) and the extractive "heuristic LLM".
> They therefore measure **our own retrieval/ranking/verification/abstention components and the pipeline logic**, and give a *lower bound* for
> generation quality. They do **not** measure how a neural embedder or a real LLM performs. Section 8 gives the one-command re-run with real
> models and a table to fill in; do that before submitting and report those numbers next to these.

## 1. Method

**Data.** Deterministic synthetic incident corpus (`eval/synthetic.py`) modelled on the task's own example (changes → incidents → services; incident
reports vs. post-mortems). Three independent corpora:

| corpus | seed | documents | unique indexed | purpose |
|---|---|---|---|---|
| `dev_seed7` | 7 | 1 112 | 1 102 | development; reader/gate heuristics were debugged on it |
| `heldout_seed11` | 11 | 1 119 | 1 108 | **held-out**: generated after the code was frozen, never inspected while tuning |
| `scale_50k` | 21 | 51 958 | **51 264** | scale run (≥ 50 000 unique documents), different seed |

Injected by the generator: exact duplicate documents (rejected at ingest: 10 / 11 / 694), near-duplicate runbooks, contradicting
incident-report ↔ post-mortem pairs (duration or root-cause change), incidents whose post-mortem is missing, attributes that are never recorded,
and non-existent entities. Every question has exact gold documents and gold answer strings.

**Questions (54 on each 1.1k corpus, 48 on the 50k corpus; requirement ≥ 40).**

| type | what it tests | example |
|---|---|---|
| `simple` (16) | one document lookup | "What was the severity of INC-0007?" |
| `multi_hop` (13) | 2–3 documents must be linked (incident → change → service catalogue) | "Which change caused INC-0030 and who authored it?" |
| `multi_hop_nl` (6) | no identifier; month + paraphrased impact; the task's own example | "Which system change caused delays in processing payments in April 2025, which services were affected and which documents confirm this?" |
| `unanswerable` (11) | must abstain: unrecorded attribute / non-existent id / missing post-mortem | "Who approved change CHG-0036?" |
| `conflict` (8) | report and post-mortem disagree; must report *both* values | "How long did INC-0014 last?" |

**Metrics (all in `eval/metrics.py`, unit-tested).**
*Retrieval* (first stage, question text only, doc-level, near-duplicates credited): Recall@1/3/5/10, MRR, nDCG@10; *Evidence recall* = share of gold documents the
**whole pipeline** gathered (after hop expansion / tool calls). *Answer correctness* (rule-based, strict): status must equal the expected status
(`answered` / `insufficient_evidence` / `conflicting_evidence`) **and** every gold string must occur in the answer (for conflicts: both values).
*Abstention recall/precision*, *conflict-detection recall*. *Citation precision* = cited documents that are relevant / cited; *citation recall* = gold docs that are cited / gold.
*Faithfulness* = share of claims passing the verifier. *Cost/latency*: p50 latency, LLM calls and tokens per question.

**Configurations compared (≥ 2 required; 7 here, plus 2 chunking strategies):**
`dense_only` · `bm25_only` · `hybrid_rrf` · `hybrid_rrf_rerank` · `hybrid_weighted_rerank` (min-max fusion instead of RRF) · `full_no_gate` (+ hop expansion) · `full` (+ abstention gate; production default).

## 2. Results — held-out corpus (seed 11, 1 119 docs, 54 questions), structure-aware chunking

| config | R@1 | R@5 | MRR | Evid.recall | Answer correct. | Abstain recall | Abstain prec. | Conflict recall | Cite prec. | Cite recall | Faithful. | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense_only` | 0.543 | 0.74 | 0.841 | 0.752 | 0.593 | 0.818 | 0.692 | 0.5 | 0.764 | 0.601 | 1.0 | 3.7 |
| `bm25_only` | 0.578 | 0.748 | 0.884 | 0.787 | 0.685 | 0.818 | 0.818 | 1.0 | 0.717 | 0.725 | 1.0 | 2.4 |
| `hybrid_rrf` | 0.554 | 0.771 | 0.868 | 0.771 | 0.667 | 0.818 | 0.75 | 0.875 | 0.764 | 0.702 | 1.0 | 4.6 |
| `hybrid_rrf_rerank` | 0.589 | 0.748 | 0.895 | 0.771 | 0.685 | 0.818 | 0.818 | 1.0 | 0.711 | 0.725 | 1.0 | 13.0 |
| `hybrid_weighted_rerank` | 0.578 | 0.748 | 0.884 | 0.771 | 0.685 | 0.818 | 0.818 | 1.0 | 0.711 | 0.725 | 1.0 | 12.7 |
| `full_no_gate` | 0.589 | 0.748 | 0.895 | 0.938 | 0.796 | 0.818 | 0.818 | 1.0 | 0.653 | 0.833 | 1.0 | 21.8 |
| `full` | 0.589 | 0.748 | 0.895 | 0.938 | 0.796 | 0.818 | 0.818 | 1.0 | 0.653 | 0.833 | 1.0 | 22.8 |

## 3. Results — development corpus (seed 7), structure-aware chunking

| config | R@1 | R@5 | MRR | Evid.recall | Answer correct. | Abstain recall | Abstain prec. | Conflict recall | Cite prec. | Cite recall | Faithful. | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense_only` | 0.581 | 0.748 | 0.888 | 0.748 | 0.574 | 0.818 | 0.692 | 0.75 | 0.764 | 0.616 | 1.0 | 4.0 |
| `bm25_only` | 0.62 | 0.748 | 0.934 | 0.76 | 0.667 | 0.818 | 0.818 | 1.0 | 0.69 | 0.713 | 1.0 | 2.9 |
| `hybrid_rrf` | 0.581 | 0.748 | 0.899 | 0.76 | 0.648 | 0.818 | 0.75 | 0.875 | 0.75 | 0.717 | 1.0 | 4.6 |
| `hybrid_rrf_rerank` | 0.597 | 0.76 | 0.912 | 0.76 | 0.667 | 0.818 | 0.818 | 1.0 | 0.69 | 0.713 | 1.0 | 13.2 |
| `hybrid_weighted_rerank` | 0.585 | 0.748 | 0.899 | 0.76 | 0.667 | 0.818 | 0.818 | 1.0 | 0.69 | 0.713 | 1.0 | 13.2 |
| `full_no_gate` | 0.597 | 0.76 | 0.912 | 0.95 | 0.815 | 0.818 | 0.9 | 1.0 | 0.621 | 0.845 | 1.0 | 23.3 |
| `full` | 0.597 | 0.76 | 0.912 | 0.95 | 0.815 | 0.818 | 0.818 | 1.0 | 0.636 | 0.845 | 1.0 | 26.6 |

The two corpora agree (full pipeline: answer correctness 0.815 on `dev`, 0.796 on held-out), i.e. the fixes made on `dev` did not over-fit it.

## 4. Results — 51 264 unique documents (seed 21, 48 questions), structure-aware chunking

| config | R@1 | R@5 | MRR | Evid.recall | Answer correct. | Abstain recall | Abstain prec. | Conflict recall | Cite prec. | Cite recall | Faithful. | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense_only` | 0.608 | 0.739 | 0.827 | 0.766 | 0.438 | 0.818 | 0.562 | 0.0 | 0.812 | 0.297 | 1.0 | 126.6 |
| `bm25_only` | 0.676 | 0.802 | 0.973 | 0.802 | 0.792 | 0.818 | 1.0 | 1.0 | 0.756 | 0.802 | 1.0 | 27.5 |
| `hybrid_rrf_rerank` | 0.658 | 0.802 | 0.946 | 0.802 | 0.792 | 0.818 | 1.0 | 1.0 | 0.756 | 0.802 | 1.0 | 165.8 |
| `full` | 0.658 | 0.802 | 0.946 | 0.955 | 0.958 | 0.818 | 1.0 | 1.0 | 0.701 | 0.937 | 1.0 | 345.6 |

Throughput / latency on the same corpus (`scripts/benchmark_scale.py`, 1 vCPU sandbox, offline embedder, Qdrant *local* mode):
ingest 1143.8 docs/s (61768 chunks, peak RSS 1046 MB, BM25 vocabulary 58859); retrieval p50/p95 —
bm25 22.3/36.9 ms, dense 107.4/124.2 ms (local mode scans vectors brute-force; a Qdrant server/Cloud uses HNSW),
hybrid+rerank 135.4/166.0 ms, full pipeline without a network LLM 266.7/420.9 ms.

## 5. Chunking comparison (held-out corpus, same questions)

structure-aware (above) vs. fixed window — `full` configuration:

| config | R@1 | R@5 | MRR | Evid.recall | Answer correct. | Abstain recall | Abstain prec. | Conflict recall | Cite prec. | Cite recall | Faithful. | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 0.484 | 0.771 | 0.814 | 0.938 | 0.722 | 0.818 | 0.818 | 0.0 | 0.549 | 0.795 | 1.0 | 28.3 |
| `hybrid_rrf_rerank` | 0.484 | 0.771 | 0.814 | 0.771 | 0.611 | 0.818 | 0.818 | 0.0 | 0.708 | 0.725 | 1.0 | 15.3 |

By question type, `full`, structure-aware:

| question type | n | answer correct. | R@5 (1st stage) | MRR |
|---|---:|---:|---:|---:|
| simple | 16 | 1.0 | 1.0 | 1.0 |
| multi_hop | 13 | 0.769 | 0.436 | 0.731 |
| multi_hop_nl | 6 | 0.0 | 0.417 | 0.833 |
| unanswerable | 11 | 0.818 | nan | nan |
| conflict | 8 | 1.0 | 1.0 | 1.0 |

By question type, `full`, fixed window:

| question type | n | answer correct. | R@5 (1st stage) | MRR |
|---|---:|---:|---:|---:|
| simple | 16 | 1.0 | 1.0 | 0.875 |
| multi_hop | 13 | 0.846 | 0.513 | 0.615 |
| multi_hop_nl | 6 | 0.5 | 0.417 | 0.833 |
| unanswerable | 11 | 0.818 | nan | nan |
| conflict | 8 | 0.0 | 1.0 | 1.0 |

**Reading.** Structure-aware chunking wins ranking (MRR 0.895 vs 0.814 with re-ranking; R@1 0.589 vs 0.484) and is the only one that supports conflict detection
(recall 1.0 vs 0.0): the fixed window joins words with spaces, destroying the line structure (`Duration: 47 minutes`) that the field-level conflict detector and the
heading/title context rely on. The fixed window wins the *natural-language, several-facts-in-one-answer* questions (`multi_hop_nl`, 0.5 vs 0.0 for the extractive reader) because it keeps a whole
short report in one chunk, whereas structure-aware chunking splits it into 3 chunks that the extractive reader must stitch together; a real LLM reads all evidence blocks and is expected to remove that gap (to be verified in §8). Decision: structure-aware is the default.

## 6. What the ablation shows

* **Hybrid vs. single retriever.** On the 1.1k corpora BM25 alone is already strong because questions contain identifiers (`INC-0012`) that dense hashing vectors blur
  (dense-only MRR 0.841 vs BM25 0.884 vs hybrid+rerank 0.895 on held-out; on 50k: 0.827 / 0.973 / 0.946). **Hybrid+rerank had the best MRR on the held-out corpus (0.895) but not on `dev` (BM25 0.934 vs 0.912) nor at 50k (BM25 0.973 vs 0.946)**: with an offline embedder the dense side adds noise rather than semantics, so on identifier-heavy questions BM25 alone is a very strong baseline. We expect hybrid to pay off with a neural embedder on paraphrased questions (`multi_hop_nl`); that is *not* demonstrated by these runs.
* **RRF vs. weighted fusion**: indistinguishable here (MRR 0.895 vs 0.884, R@5 equal) — RRF is kept because it needs no score calibration.
* **Re-ranker + MMR** raises MRR over plain RRF (0.868 → 0.895 held-out; 0.899 → 0.912 dev) at ≈ +7 ms.
* **Hop expansion** is the biggest single lever for multi-document questions: evidence recall 0.771 → 0.938 and answer correctness 0.685 → 0.796 (held-out). First-stage Recall@K is capped near 0.75–0.80 *by construction*: the change record and the service page are not retrievable from the question text alone; they are only reachable through the identifier found in the incident report.
* **Abstention gate**: on these sets it changes little beyond skipping LLM calls (0.944 calls/question instead of 1.0, i.e. the gate refuses ~5 % of questions *before* spending tokens: non-existent identifiers). It was deliberately kept conservative (coverage < 0.30): an earlier, stricter threshold (0.55) produced false refusals on paraphrased questions ("How long did X last?") — the lexical coverage of answerable and unanswerable questions overlaps (0.29–0.73 vs 0.22–0.65), so finer judgement has to come from the LLM and the verifier.
* **Faithfulness = 1.0** everywhere is expected for an extractive reader (each claim is a sentence of its evidence); it becomes informative only with a generative LLM (§8).

## 7. Error analysis (held-out, `full`, structure; 11 of 54 questions wrong)

| failure class | n | examples | root cause | improvement |
|---|---:|---|---|---|
| `multi_hop_nl` answered incompletely or wrong incident | 4 (`q031,q033,q034,q035`) | month + impact questions | many incidents share the same impact; the month token is the only discriminator and BM25/hashing weigh it like any word; one question retrieved nothing relevant (R@5 = 0) | temporal parsing → metadata filter on `first_date`; cross-encoder reranker; with a neural embedder paraphrases ("payments were delayed") match the report text |
| false abstention | 2 (`q030,q032`) | "outdated search results" / "errors during checkout" | the paraphrase word occurs nowhere in the corpus (docs say "stale search results", "checkout errors"); the *offline reader's* rule "≥ 2 question words absent from the corpus → refuse" fires | real semantic embeddings + LLM judgement; query rewriting/synonym expansion by the planner |
| `answered_unanswerable` | 2 (`q042,q043`) | "Who approved change CHG-…?" | the attribute word (`approved`) is a single rare term; evidence for the *entity* exists, so neither gate nor reader can tell that the *attribute* is missing | LLM-side judgement (system prompt rule 4) + claim verification; add an "attribute presence" check (expected key in a `key: value` line) |
| multi-hop answer incomplete | 2 (`q023,q024`) | "which change… *and who authored it*" | the extractive reader picked the two `Root cause change` lines and ignored the author line in the hop document | generative reader (planned) or per-document slot filling |
| 3-hop retrieval miss | 1 (`q027`) | incident → change → *service* → team | the bridge entity `wallet` is a plain word, not an identifier, so identifier-based hop expansion cannot follow it | LLM tool loop (`search_documents("service wallet owner")`), or entity-linking against document titles |

Other observations: (1) *conflict* questions are solved (8/8) **only with structure-aware chunking** (§5); (2) identical-text documents are credited through `also_doc_ids`, so de-duplication does not hurt recall; (3) citation precision is lowest for `full` (0.653) because hop expansion adds the change/service documents that are relevant but sometimes cited beyond what the question required, while citation *recall* rises 0.725 → 0.833.

## 8. Re-running with real models (required before submission)

```bash
# .env: EMBEDDING_PROVIDER=fastembed, LLM_PROVIDER=openai, LLM_BASE_URL/LLM_API_KEY/LLM_MODEL (and price variables for cost)
python -m eval.run_eval --corpus data/corpus_heldout --embedder fastembed --llm openai --chunking structure --tag real_heldout
python -m eval.tables eval/results/real_heldout.json
```
Fill the table below with your measured values (and add a second LLM or embedding model for the bonus comparison — change `LLM_MODEL` / `EMBEDDING_MODEL` and re-run).

| stack | config | R@5 | MRR | Evid. recall | Answer correct. | Abstain recall | Conflict recall | Cite prec. | Faithful. | LLM calls | tokens/q | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed bge-small + `<LLM>` | `hybrid_rrf_rerank` | | | | | | | | | | | |
| fastembed bge-small + `<LLM>` | `full` | | | | | | | | | | | |

Hypotheses to confirm or refute with those runs: dense+hybrid beats BM25 on `multi_hop_nl`; the LLM tool loop fixes the 3-hop and the "who authored it" cases; false abstentions on paraphrases disappear; faithfulness drops below 1.0 for a generative model and the verifier catches the difference.
Optional LLM-as-judge for free-text correctness is *not* implemented — correctness here is rule-based and strict (a correct paraphrase without the gold string counts as wrong), which is conservative for generative models.

## 9. Threats to validity

* Synthetic, templated text: easier than real prose in vocabulary, harder in that facts are scattered over documents. HotpotQA (real, multi-hop) support is provided (`scripts/prepare_hotpotqa.py`) but was **not run here** (no network).
* Rule-based correctness and small question counts (n = 6–16 per type) → differences of one or two questions are noise (one question = 1.9 % overall).
* Offline stack only (see top note). The hand-set re-ranker weights and the 0.30 gate threshold were chosen by inspecting `dev`, then confirmed on `held-out` and `50k`.
* Latencies are from a single shared vCPU with Qdrant local mode and no network LLM; production numbers will be dominated by LLM latency.

## 10. Reproduce

```bash
python -m scripts.generate_synthetic --incidents 120 --filler 600 --seed 7  --out data/corpus_small
python -m scripts.generate_synthetic --incidents 120 --filler 600 --seed 11 --out data/corpus_heldout
python -m scripts.generate_synthetic --incidents 3200 --filler 38800 --seed 21 --out data/corpus_50k
python -m eval.run_eval --corpus data/corpus_small   --chunking both --tag dev_seed7
python -m eval.run_eval --corpus data/corpus_heldout --chunking both --tag heldout_seed11
python -m eval.run_eval --corpus data/corpus_50k --chunking structure --configs dense_only,bm25_only,hybrid_rrf_rerank,full --tag scale_50k
python -m scripts.benchmark_scale --corpus data/corpus_50k
```
Raw per-question results (JSON) are written to `eval/results/`.

"""Are two embedding providers interchangeable for ONE existing vector store?
Run this before pinning EMBEDDING_STORE_ID. It embeds the same texts with the local ONNX model (used at ingestion)
and with the provider configured in .env (EMBEDDING_PROVIDER=openai ...), then compares the vectors.

   python -m scripts.check_embedding_compat --local-model BAAI/bge-small-en-v1.5

Reading the result: cosine >= 0.98 for every text -> interchangeable. 0.90-0.98 -> retrieval works but ranking drifts: re-evaluate.
< 0.90 -> NOT compatible (different model/quantisation/prefix): re-ingest with the provider you deploy."""
from __future__ import annotations

import argparse

import numpy as np

from app.config import get_settings
from app.embeddings import FastEmbedEmbedder, build_embedder

TEXTS = ["Incident Report INC-0012 severity SEV2 duration 47 minutes", "id: CHG-0123 author: Nika Beridze service: payments-api",
         "Postmortem: root cause change CHG-0123, pool size reduced", "Weekly operations digest for week 12 of 2024",
         "Runbook for ledger: restart procedure and escalation"]
QUERIES = ["Which change caused INC-0003 and who authored it?", "payment processing delays in June 2024", "How long did INC-0006 last?"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-model", default="BAAI/bge-small-en-v1.5")
    a = ap.parse_args()
    s = get_settings().model_copy(update={"embedding_store_id": None})
    remote = build_embedder(s)
    local = FastEmbedEmbedder(a.local_model, remote.dim)
    print(f"local : {local.name}\nremote: {remote.name}\n")
    worst = 1.0
    rd, ld = remote.embed_documents(TEXTS), local.embed_documents(TEXTS)
    for t, r, l in zip(TEXTS, rd, ld):
        c = float(np.dot(r, l)); worst = min(worst, c); print(f"doc   cos={c:.4f}  {t[:60]}")
    for q in QUERIES:
        c = float(np.dot(remote.embed_query(q), local.embed_query(q))); worst = min(worst, c); print(f"query cos={c:.4f}  {q[:60]}")
    verdict = "COMPATIBLE" if worst >= 0.98 else "DRIFT - re-evaluate" if worst >= 0.90 else "NOT COMPATIBLE - re-ingest"
    print(f"\nworst cosine {worst:.4f} -> {verdict}")


if __name__ == "__main__":
    main()

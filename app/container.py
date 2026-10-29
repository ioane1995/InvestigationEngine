"""Dependency wiring: builds the object graph from Settings (used by the API, scripts and tests)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.config import Settings
from app.embeddings import Embedder, build_embedder
from app.ingestion.service import IngestionService
from app.llm.base import LLM, LLMConfigError
from app.llm.heuristic import HeuristicLLM
from app.llm.openai_compat import OpenAICompatLLM
from app.rag.engine import InvestigationEngine
from app.retrieval.hybrid import CorpusIndex, HybridRetriever, RetrievalConfig
from app.stores.qdrant_store import QdrantStore

log = logging.getLogger(__name__)


def build_llm(s: Settings, strict: bool = False) -> LLM:
    """Factory. In the app (strict=False) a bad configuration never crashes start-up: it logs an ERROR and falls back to
    the offline reader. In evaluation (strict=True) it raises instead, so a run can never silently use the wrong model."""
    provider = s.llm_provider.lower()
    if provider not in ("openai", "heuristic"):
        msg = f"unknown LLM_PROVIDER {s.llm_provider!r}: use 'openai' (any OpenAI-compatible API, e.g. Groq) or 'heuristic'"
        if strict:
            raise ValueError(msg)
        log.error("%s -> using the offline heuristic reader", msg)
        return HeuristicLLM()
    if provider == "heuristic":
        log.info("LLM: offline heuristic reader")
        return HeuristicLLM()
    if not s.llm_api_key:
        if strict:
            raise ValueError("LLM_PROVIDER=openai but LLM_API_KEY is empty")
        log.error("LLM_PROVIDER=openai but LLM_API_KEY is empty -> using the offline heuristic reader")
        return HeuristicLLM()
    try:
        llm = OpenAICompatLLM(s.llm_base_url, s.llm_api_key, s.llm_model, s.llm_timeout_s, s.llm_max_retries,
                              min_output_tokens=s.llm_min_output_tokens, reasoning_effort=s.llm_reasoning_effort, min_interval_s=s.llm_min_interval_s)
    except LLMConfigError as e:
        if strict:
            raise ValueError(str(e)) from e
        log.error("%s -> using the offline heuristic reader", e)
        return HeuristicLLM()
    log.info("LLM: base_url=%s model=%s api_key_set=yes", s.llm_base_url, s.llm_model)
    return llm


@dataclass
class Container:
    settings: Settings
    embedder: Embedder
    store: QdrantStore
    corpus: CorpusIndex
    ingestion: IngestionService
    retriever: HybridRetriever
    engine: InvestigationEngine
    llm: LLM

    def default_retrieval(self) -> RetrievalConfig:
        s = self.settings
        return RetrievalConfig(mode=s.retrieval_mode, fusion=s.fusion, candidate_k=s.candidate_k, top_k=s.top_k, rerank=s.rerank)


def build_container(s: Settings, embedder: Optional[Embedder] = None, llm: Optional[LLM] = None, load: bool = True) -> Container:
    embedder = embedder or build_embedder(s)
    store = QdrantStore(s, embedder.dim, embedder.name)
    corpus = CorpusIndex()
    ingestion = IngestionService(store, embedder, corpus, s)
    if load:
        ingestion.load_from_store()
    retriever = HybridRetriever(store, embedder, corpus)
    llm = llm or build_llm(s)
    if isinstance(llm, HeuristicLLM) and llm.idf is None:
        llm.idf = corpus.bm25.idf
    engine = InvestigationEngine(retriever, llm, s.llm_price_in_per_m, s.llm_price_out_per_m)
    return Container(s, embedder, store, corpus, ingestion, retriever, engine, llm)

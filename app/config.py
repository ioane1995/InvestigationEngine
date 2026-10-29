"""Central, validated configuration. All secrets come from environment variables / .env,
never from source code. `get_settings()` is cached; tests build `Settings(...)` directly."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    admin_api_key: Optional[str] = None  # protects write endpoints when set

    # ---- vector store (Qdrant) ----
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_path: str = "./data/qdrant"  # used when qdrant_url is empty; ":memory:" for tests
    collection_prefix: str = "ie"

    # ---- embeddings ----
    embedding_provider: str = "hashing"  # hashing | fastembed | openai (OpenAI-shaped API) | hf (Hugging Face feature-extraction)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None
    # Identity pinned in the vector store. Set it ONLY when a different provider serves the very same model the
    # collection was built with (e.g. 'fastembed:BAAI/bge-small-en-v1.5' served through an API) and you have verified
    # compatibility with scripts/check_embedding_compat.py.
    embedding_store_id: Optional[str] = None

    # ---- LLM ----
    llm_provider: str = "heuristic"  # heuristic | openai (OpenAI-compatible)
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: Optional[str] = None
    llm_model: str = "llama-3.3-70b-versatile"
    llm_timeout_s: float = 45.0
    llm_max_retries: int = 3
    llm_min_interval_s: float = 0.0  # minimum seconds between LLM calls (set 2-6 on strict free tiers)
    llm_min_output_tokens: int = 2048  # headroom for reasoning models (gpt-oss, qwen3...) whose hidden thinking counts against max_tokens
    llm_reasoning_effort: Optional[str] = None  # e.g. 'low' for gpt-oss on Groq; empty = parameter is not sent
    llm_price_in_per_m: float = 0.0
    llm_price_out_per_m: float = 0.0

    # ---- ingestion ----
    chunk_strategy: str = "structure"  # structure | fixed
    chunk_max_tokens: int = 200
    chunk_overlap_tokens: int = 40
    max_upload_mb: int = 15

    # ---- retrieval / RAG defaults ----
    retrieval_mode: str = "hybrid"  # hybrid | dense | bm25
    fusion: str = "rrf"  # rrf | weighted
    candidate_k: int = 40
    top_k: int = 8
    rerank: bool = True
    hops: int = 1
    use_planner: bool = True
    max_tool_rounds: int = 2
    evidence_k: int = 10
    min_evidence_coverage: float = 0.30  # hard abstention gate (see retrieval/rerank.assess_evidence)

    # ---- API protection / perf ----
    rate_limit_per_min: int = Field(default=20, ge=1)
    cache_ttl_s: int = 600
    cache_size: int = 256

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()

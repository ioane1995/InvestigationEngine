import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.container import build_container  # noqa: E402
from app.models import RawDocument  # noqa: E402


def make_settings(**kw) -> Settings:
    base = dict(qdrant_path=":memory:", embedding_provider="hashing", llm_provider="heuristic", admin_api_key=None, rate_limit_per_min=1000, _env_file=None)
    base.update(kw)
    return Settings(**base)


@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture
def container(settings):
    return build_container(settings, load=False)


DOCS = [
    RawDocument("inc-1", "Incident Report INC-0012", "# Incident Report INC-0012\n\n- Date: 2024-03-14\n- Severity: SEV2\n- Duration: 47 minutes\n"
                "- Services affected: payments-api, ledger\n- Root cause change: CHG-0123\n- Customer impact: payment processing delays\n\n"
                "## Summary\nPayment processing delays were observed after a configuration change.", "inc1.md", "md"),
    RawDocument("pm-1", "Postmortem INC-0012", "Postmortem: INC-0012\n\n- Duration: 75 minutes\n- Root cause change: CHG-0123\n\nThe pool size was reduced.", "pm1.txt", "txt"),
    RawDocument("chg-1", "Change CHG-0123", "id: CHG-0123\nauthor: Nika Beridze\nservice: payments-api\nsummary: reduced connection pool size from 50 to 10", "chg.json", "json"),
    RawDocument("rb-1", "Runbook ledger", "Restart the ledger service by running the standard procedure. Check dashboards.", "rb.md", "md"),
]


@pytest.fixture
def loaded(container):
    container.ingestion.ingest_many(DOCS)
    return container

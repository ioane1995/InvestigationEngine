"""ASGI entrypoint: `uvicorn app.api.asgi:app`. Kept separate so importing app.api.main has no side effects."""
from app.api.main import create_app

app = create_app()

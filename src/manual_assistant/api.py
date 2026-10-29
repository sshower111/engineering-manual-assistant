"""FastAPI app: ``POST /ask`` plus a minimal web page.

Run: uvicorn manual_assistant.api:app
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from manual_assistant.assistant import Assistant
from manual_assistant.config import Settings, get_settings
from manual_assistant.schemas import AskRequest, AskResponse

STATIC_DIR = Path(__file__).parent / "static"


def require_index(store, collection: str) -> None:
    """Fail at startup on an empty collection instead of refusing every question."""
    if store.count() == 0:
        raise RuntimeError(
            f"Chroma collection {collection!r} is empty. "
            "Build it first: python -m manual_assistant.ingest"
        )


def build_assistant(settings: Settings) -> Assistant:
    from manual_assistant.embeddings import SentenceTransformerEmbedder
    from manual_assistant.llm import ClaudeAnswerGenerator
    from manual_assistant.query_log import QueryLogger
    from manual_assistant.store import VectorStore

    store = VectorStore(
        settings.chroma_dir,
        settings.collection_name,
        SentenceTransformerEmbedder(settings.embedding_model),
    )
    require_index(store, settings.collection_name)
    generator = ClaudeAnswerGenerator(
        settings.llm_model, settings.llm_max_tokens, settings.llm_effort
    )
    return Assistant(
        store,
        generator,
        QueryLogger(settings.log_file),
        top_k=settings.top_k,
        min_score=settings.min_score,
        snippet_chars=settings.snippet_chars,
    )


def create_app(assistant: Assistant | None = None) -> FastAPI:
    """Build the app. Tests pass a ready ``assistant``; otherwise it is built at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.assistant = assistant or build_assistant(get_settings())
        yield

    app = FastAPI(title="Engineering Manual Assistant", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Sync endpoint: FastAPI runs it in a thread pool, so the blocking
    # embedding and Claude calls don't stall the event loop.
    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> AskResponse:
        return request.app.state.assistant.ask(body.question)

    return app


app = create_app()

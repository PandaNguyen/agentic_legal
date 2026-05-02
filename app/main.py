import logging
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routers.chat import router as chat_router
from app.api.routers.search import router as search_router
from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()
log_path = setup_logging(log_level=settings.log_level, log_file=settings.log_file)
logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title=settings.app_name)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(chat_router, prefix="/v1")
app.include_router(search_router, prefix="/v1")
logger.info("Application started app=%s env=%s log_file=%s", settings.app_name, settings.app_env, log_path)
logger.info(
    "Retrieval config qdrant_url=%s collection=%s dense_model=%s",
    settings.qdrant_url,
    settings.qdrant_collection_hybrid,
    settings.dense_embedding_model,
)
logger.info(
    "LLM config base_url=%s model=%s api_style=%s",
    settings.openai_base_url or "https://api.openai.com/v1",
    settings.openai_llm_model or settings.openai_model,
    "chat_completions"
    if settings.openai_base_url and "api.openai.com" not in settings.openai_base_url
    else settings.openai_api_style,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (perf_counter() - start) * 1000
        logger.exception(
            "Unhandled request error method=%s path=%s elapsed_ms=%.2f",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (perf_counter() - start) * 1000
    logger.info(
        "Request completed method=%s path=%s status_code=%s elapsed_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/healthz")
def healthz():
    return {"status": "ok", "env": settings.app_env}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")

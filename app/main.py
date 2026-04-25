from fastapi import FastAPI
from app.api.routers.chat import router as chat_router
from app.api.routers.search import router as search_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)
app.include_router(chat_router, prefix="/v1")
app.include_router(search_router, prefix="/v1")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "env": settings.app_env}

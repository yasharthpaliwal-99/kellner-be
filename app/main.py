from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.conversation import router as conversation_router
from app.api.device_auth import router as device_auth_router
from app.api.kitchen import router as kitchen_router
from app.db.mongo import ensure_order_indexes
from app.services.device_auth_service import ensure_device_auth_indexes


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_order_indexes()
    ensure_device_auth_indexes()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversation_router, prefix="/api")
app.include_router(kitchen_router, prefix="/api")
app.include_router(device_auth_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/health")
async def health_check():
    return {"status": "ok"}

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.conversation import router as conversation_router
from app.api.device_auth import router as device_auth_router
from app.api.face_local import router as face_local_router
from app.api.kitchen import router as kitchen_router
from app.config import config
from app.db.face_schema import ensure_face_schema
from app.db.mongo import ensure_order_indexes
from app.services.device_auth_service import ensure_device_auth_indexes


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_order_indexes()
    ensure_device_auth_indexes()
    try:
        ensure_face_schema()
    except Exception as e:
        print(f"[lifespan] face schema skip: {e}")
    yield


app = FastAPI(lifespan=lifespan)

# Regex: localhost + private LAN IPs (Vite from phone: http://192.168.x.x:5175).
# If the SPA is served from the VM public IP (same browser tab as API host), set CORS_EXTRA_ORIGINS
# e.g. http://74.249.2.119 — Origin is that URL (default http port omitted in header).
_CORS_DEV_HOST_REGEX = (
    r"^https?://("
    r"localhost|127\.0\.0\.1|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}"
    r")(:\d+)?$"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS or ["*"],
    allow_origin_regex=_CORS_DEV_HOST_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversation_router, prefix="/api")
app.include_router(kitchen_router, prefix="/api")
app.include_router(device_auth_router, prefix="/api")
app.include_router(face_local_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/health")
async def health_check():
    return {"status": "ok"}

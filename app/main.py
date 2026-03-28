from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.conversation import router as conversation_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversation_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Kellner

Voice-based restaurant assistant: Azure Speech (STT/TTS), Azure OpenAI, PostgreSQL + pgvector, FastAPI, WebSocket.

## Setup

1. Python 3.11+
2. `cp .env.example .env` and add your Azure / Postgres credentials.
3. `pip install -r requirements.txt`
4. Database (once): `python scripts/setup_db.py` then `python scripts/embed_menu.py`
5. Run: `uvicorn app.main:app --reload`
6. Open `http://127.0.0.1:8000` — use **Start voice chat** for continuous voice.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/setup_db.py` | Create tables + seed menu & customers |
| `scripts/embed_menu.py` | Add `embedding` column + HNSW index + vectors |
| `scripts/test_ws.py` | Batch WAV WebSocket test |
| `scripts/test_azure_connections.py` | Smoke-test Azure endpoints |

## License

Proprietary / internal use unless stated otherwise.

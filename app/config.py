import os
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        v = os.getenv(name)
        if v:
            return v
    return None


def _csv_env_list(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    items = [v.strip() for v in (raw or "").split(",")]
    return [v for v in items if v]


class Config:
    """
    Required in .env:

    Speech:   AZURE_SPEECH_KEY, AZURE_SPEECH_REGION
    Reasoning: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME

    Optional: AZURE_OPENAI_API_VERSION, AZURE_TTS_VOICE_NAME
    """

    AZURE_SPEECH_KEY: Optional[str] = _first_env("AZURE_SPEECH_KEY", "STT_API_KEY", "TTS_API_KEY")
    AZURE_SPEECH_REGION: Optional[str] = _first_env("AZURE_SPEECH_REGION", "AZURE_REGION")
    AZURE_TTS_VOICE_NAME: str = _first_env("AZURE_TTS_VOICE_NAME") or "en-US-JennyNeural"

    # Azure OpenAI audio (gpt-4o-mini-tts) — separate from chat if needed
    AZURE_OPENAI_TTS_ENDPOINT: Optional[str] = _first_env("AZURE_OPENAI_TTS_ENDPOINT")
    AZURE_OPENAI_TTS_API_KEY: Optional[str] = _first_env("AZURE_OPENAI_TTS_API_KEY")
    AZURE_OPENAI_TTS_DEPLOYMENT: str = _first_env("AZURE_OPENAI_TTS_DEPLOYMENT") or "gpt-4o-mini-tts"
    AZURE_OPENAI_TTS_API_VERSION: str = _first_env("AZURE_OPENAI_TTS_API_VERSION") or "2025-04-01-preview"
    AZURE_OPENAI_TTS_VOICE: str = _first_env("AZURE_OPENAI_TTS_VOICE") or "nova"

    AZURE_OPENAI_ENDPOINT: Optional[str] = _first_env("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: Optional[str] = _first_env("AZURE_OPENAI_API_KEY", "LLM_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = _first_env("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION: str = _first_env("AZURE_OPENAI_API_VERSION") or "2025-01-01-preview"

    AZURE_EMBEDDING_ENDPOINT: Optional[str] = _first_env("AZURE_EMBEDDING_ENDPOINT")
    AZURE_EMBEDDING_API_KEY: Optional[str] = _first_env("AZURE_EMBEDDING_API_KEY")
    AZURE_EMBEDDING_DEPLOYMENT: str = _first_env("AZURE_EMBEDDING_DEPLOYMENT") or "text-embedding-ada-002"

    PGSQL_ENDPOINT: Optional[str] = _first_env("PGSQL_ENDPOINT")
    PGSQL_DB_NAME: Optional[str] = _first_env("PGSQL_DB_NAME")
    PGSQL_ADMIN_USERNAME: Optional[str] = _first_env("PGSQL_ADMIN_USERNAME")
    PGSQL_ADMIN_PASSWORD: Optional[str] = _first_env("PGSQL_ADMIN_PASSWORD")

    # MongoDB (draft orders) — MONGO_URL / DATABASE_NAME supported for legacy .env
    MONGODB_URI: Optional[str] = _first_env("MONGODB_URI", "MONGO_URL")
    MONGODB_DB_NAME: str = _first_env("MONGODB_DB_NAME", "DATABASE_NAME") or "kellner"
    MONGODB_ORDERS_COLLECTION: str = _first_env("MONGODB_ORDERS_COLLECTION") or "orders"

    # Default tenant / guest for voice session until auth exists
    DEFAULT_HOTEL_ID: int = int(_first_env("DEFAULT_HOTEL_ID") or "1")
    DEFAULT_CUSTOMER_ID: int = int(_first_env("DEFAULT_CUSTOMER_ID") or "1")

    # CORS allowlist for browser clients (SWA, localhost dev, etc.)
    CORS_ALLOW_ORIGINS: List[str] = _csv_env_list(
        "CORS_ALLOW_ORIGINS",
        "https://nice-ground-03e95fd0f.4.azurestaticapps.net,http://localhost:5173,http://127.0.0.1:5173",
    )

    # Local face (InsightFace + pgvector). Optional.
    INSIGHTFACE_ROOT: Optional[str] = _first_env("INSIGHTFACE_ROOT")
    FACE_MATCH_MAX_DISTANCE: float = float(_first_env("FACE_MATCH_MAX_DISTANCE") or "0.45")


config = Config()

import os
from pathlib import Path
from typing import Optional

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


config = Config()

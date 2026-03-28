#!/usr/bin/env python3
"""
Connectivity smoke test for kellner.

Checks:
1. Azure Speech TTS
2. Azure OpenAI (list deployments / simple completion)

Run from the kellner directory:
  python scripts/test_azure_connections.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def test_speech_tts() -> bool:
    from app.config import config
    from app.services.tts_service import TTSService

    if not config.AZURE_SPEECH_KEY or not config.AZURE_SPEECH_REGION:
        print("SKIP Speech: set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env")
        return True

    try:
        out_path = TTSService().synthesize("Connection test.")
        size = Path(out_path).stat().st_size
        Path(out_path).unlink(missing_ok=True)
        if size == 0:
            print("FAIL Speech TTS: empty audio response")
            return False
        print(f"OK   Speech TTS ({size} bytes)")
        return True
    except Exception as e:
        print(f"FAIL Speech TTS: {e}")
        return False


def test_openai() -> bool:
    from openai import OpenAI
    from app.config import config

    if not config.AZURE_OPENAI_ENDPOINT or not config.AZURE_OPENAI_API_KEY:
        print("SKIP OpenAI: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
        return True
    if not config.AZURE_OPENAI_DEPLOYMENT_NAME:
        print("SKIP OpenAI: set AZURE_OPENAI_DEPLOYMENT_NAME in .env")
        return True

    try:
        client = OpenAI(
            base_url=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
        )
        response = client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        reply = response.choices[0].message.content or ""
        print(f"OK   Azure OpenAI — reply: {reply.strip()!r}")
        return True
    except Exception as e:
        print(f"FAIL Azure OpenAI: {e}")
        return False


def main() -> int:
    load_env()
    args = sys.argv[1:]
    if "--llm-only" in args:
        ok = test_openai()
    elif "--speech-only" in args:
        ok = test_speech_tts()
    else:
        ok = test_speech_tts() and test_openai()
    print("\nAll connectivity checks passed." if ok else "\nConnectivity check failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

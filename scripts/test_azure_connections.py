#!/usr/bin/env python3
"""
Connectivity smoke test for kellner.

Checks:
1. Azure OpenAI TTS (gpt-4o-mini-tts)
2. Azure OpenAI chat (deployment smoke test)

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


def test_openai_tts() -> bool:
    from app.config import config
    from app.services.tts_streaming_service import StreamingTTSService

    if not config.AZURE_OPENAI_TTS_ENDPOINT or not config.AZURE_OPENAI_TTS_API_KEY:
        print("SKIP OpenAI TTS: set AZURE_OPENAI_TTS_ENDPOINT and AZURE_OPENAI_TTS_API_KEY in .env")
        return True

    try:
        tts = StreamingTTSService()
        pcm = tts.synthesize_segment_pcm("Connection test.")
        if len(pcm) < 200:
            print("FAIL OpenAI TTS: empty or tiny PCM response")
            return False
        print(f"OK   Azure OpenAI TTS segment ({len(pcm)} PCM bytes @ 16k mono)")
        pcm2 = tts.synthesize_full_turn_pcm("First sentence. Second sentence.")
        if len(pcm2) < 200:
            print("FAIL OpenAI TTS: full-turn synthesis empty or tiny")
            return False
        print(f"OK   Azure OpenAI TTS full-turn ({len(pcm2)} PCM bytes @ 16k mono)")
        return True
    except Exception as e:
        print(f"FAIL OpenAI TTS: {e}")
        return False


def test_openai() -> bool:
    from openai import AzureOpenAI
    from app.config import config

    if not config.AZURE_OPENAI_ENDPOINT or not config.AZURE_OPENAI_API_KEY:
        print("SKIP OpenAI: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
        return True
    if not config.AZURE_OPENAI_DEPLOYMENT_NAME:
        print("SKIP OpenAI: set AZURE_OPENAI_DEPLOYMENT_NAME in .env")
        return True

    try:
        from app.services.llm_service import completion_limit_kwargs

        client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
        response = client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            **completion_limit_kwargs(10),
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
    elif "--tts-only" in args:
        ok = test_openai_tts()
    else:
        ok = test_openai_tts() and test_openai()
    print("\nAll connectivity checks passed." if ok else "\nConnectivity check failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

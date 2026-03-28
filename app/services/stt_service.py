from pathlib import Path

import requests

from app.config import config

_STT_URL = (
    "https://{region}.stt.speech.microsoft.com"
    "/speech/recognition/conversation/cognitiveservices/v1"
    "?language=en-IN&format=simple"
)

_FORMAT_MAP = {
    ".wav":  "audio/wav; codecs=audio/pcm; samplerate=16000",
    ".webm": "audio/webm; codecs=opus",
    ".ogg":  "audio/ogg; codecs=opus",
}


class STTService:
    def __init__(self):
        if not config.AZURE_SPEECH_KEY or not config.AZURE_SPEECH_REGION:
            raise ValueError("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env")
        self._url = _STT_URL.format(region=config.AZURE_SPEECH_REGION)
        self._key = config.AZURE_SPEECH_KEY

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes audio via Azure Speech REST API.
        Supports .wav (PCM 16kHz), .webm and .ogg (opus) from browser.
        Returns recognized text, or empty string if nothing was recognised.
        """
        suffix = Path(audio_path).suffix.lower()
        content_type = _FORMAT_MAP.get(suffix, _FORMAT_MAP[".wav"])

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        response = requests.post(
            self._url,
            headers={
                "Ocp-Apim-Subscription-Key": self._key,
                "Content-Type": content_type,
            },
            data=audio_data,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"STT request failed: {response.status_code} — {response.text[:200]}"
            )

        result = response.json()
        if result.get("RecognitionStatus") == "Success":
            return result.get("DisplayText", "")
        return ""

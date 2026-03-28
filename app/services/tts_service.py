import logging
import tempfile

import requests

from app.config import config

logger = logging.getLogger(__name__)

_TTS_URL = "https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

_SSML = """<speak version='1.0' xml:lang='en-IN'>
  <voice name='{voice}'>{text}</voice>
</speak>"""


class TTSService:
    def __init__(self):
        if not config.AZURE_SPEECH_KEY or not config.AZURE_SPEECH_REGION:
            raise ValueError("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env")
        self._url = _TTS_URL.format(region=config.AZURE_SPEECH_REGION)
        self._session = requests.Session()
        self._session.headers.update({
            "Ocp-Apim-Subscription-Key": config.AZURE_SPEECH_KEY,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
        })

    def synthesize(self, text: str) -> str:
        """
        Converts text to speech via Azure REST API.
        Returns path to a temp .wav file (caller should delete after use).
        """
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ssml = _SSML.format(voice=config.AZURE_TTS_VOICE_NAME, text=safe_text)

        response = self._session.post(self._url, data=ssml.encode("utf-8"))
        if response.status_code != 200:
            raise RuntimeError(
                f"TTS request failed: {response.status_code} — {response.text[:200]}"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(response.content)
            logger.info("TTS synthesis completed, wrote %d bytes.", len(response.content))
            return f.name

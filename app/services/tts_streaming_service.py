"""
TTS via Azure OpenAI gpt-4o-mini-tts (audio/speech API).
Returns raw PCM16 mono at 16 kHz for the WebSocket client.
Barge-in: stop() sets a flag; in-flight HTTP may still finish (short segments).
"""
from __future__ import annotations

import io
import threading
import wave
from array import array

import requests

from app.config import config


def _resample_int16_mono(pcm: bytes, src_hz: int, dst_hz: int = 16000) -> bytes:
    if src_hz == dst_hz or not pcm:
        return pcm
    samples = array("h")
    samples.frombytes(pcm)
    if len(samples) < 2:
        return pcm
    ratio = dst_hz / src_hz
    out_len = max(1, int(len(samples) * ratio))
    out = array("h", [0] * out_len)
    for i in range(out_len):
        x = i / ratio
        i0 = int(x)
        i1 = min(i0 + 1, len(samples) - 1)
        f = x - i0
        v = samples[i0] * (1 - f) + samples[i1] * f
        vi = int(v)
        if vi > 32767:
            vi = 32767
        elif vi < -32768:
            vi = -32768
        out[i] = vi
    return out.tobytes()


def _wav_bytes_to_pcm16_mono_16k(wav_bytes: bytes) -> bytes:
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        fr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"Expected 16-bit WAV from TTS, got sample width {sw}")
    if nch == 2:
        s = array("h", raw)
        mono = array("h", ((s[i] + s[i + 1]) // 2 for i in range(0, len(s), 2)))
        raw = mono.tobytes()
    elif nch != 1:
        raise ValueError(f"Expected mono or stereo WAV, got {nch} channels")
    return _resample_int16_mono(raw, fr, 16000)


class StreamingTTSService:
    def __init__(self) -> None:
        endpoint = (config.AZURE_OPENAI_TTS_ENDPOINT or "").rstrip("/")
        key = config.AZURE_OPENAI_TTS_API_KEY
        dep = config.AZURE_OPENAI_TTS_DEPLOYMENT
        if not endpoint or not key or not dep:
            raise ValueError(
                "Set AZURE_OPENAI_TTS_ENDPOINT, AZURE_OPENAI_TTS_API_KEY, "
                "and AZURE_OPENAI_TTS_DEPLOYMENT in .env for gpt-4o-mini-tts."
            )
        ver = config.AZURE_OPENAI_TTS_API_VERSION
        self._url = f"{endpoint}/openai/deployments/{dep}/audio/speech?api-version={ver}"
        self._voice = config.AZURE_OPENAI_TTS_VOICE
        self._session = requests.Session()
        self._session.headers.update({"api-key": key})
        self._lock = threading.Lock()
        self._cancelled = False

    def stop(self) -> None:
        with self._lock:
            self._cancelled = True

    def reset_for_new_turn(self) -> None:
        with self._lock:
            self._cancelled = False

    def synthesize_segment_pcm(self, text: str) -> bytes:
        text = (text or "").strip()
        if not text:
            return b""

        with self._lock:
            if self._cancelled:
                return b""

        body = {
            "model": config.AZURE_OPENAI_TTS_DEPLOYMENT,
            "input": text[:4096],
            "voice": self._voice,
            "response_format": "wav",
        }
        try:
            r = self._session.post(
                self._url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
        except requests.RequestException:
            return b""

        with self._lock:
            if self._cancelled:
                return b""

        if r.status_code != 200:
            return b""

        try:
            return _wav_bytes_to_pcm16_mono_16k(r.content)
        except Exception:
            return b""

"""
TTS via Azure OpenAI gpt-4o-mini-tts (audio/speech API).
Returns raw PCM16 mono at 16 kHz for the WebSocket client.

- synthesize_segment_pcm: short phrases (e.g. filler) — one WAV request each.
- synthesize_full_turn_pcm: full assistant reply in one prosody (split only past 4096-char API limit).

Barge-in: stop() sets a flag; in-flight HTTP may still finish (short segments).
"""
from __future__ import annotations

import io
import threading
import wave
from array import array

import requests

from app.config import config

TTS_INPUT_MAX_CHARS = 4096


def _split_tts_input(text: str, max_len: int = TTS_INPUT_MAX_CHARS) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts: list[str] = []
    rest = text
    while rest:
        if len(rest) <= max_len:
            parts.append(rest)
            break
        chunk = rest[:max_len]
        cut = chunk.rfind(" ")
        if cut < max_len // 2:
            cut = max_len
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return [p for p in parts if p]


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

    def _speech_json_body(self, input_text: str, *, is_filler: bool) -> dict:
        """Same voice + instruction policy for filler and main (config read each call)."""
        voice = (config.AZURE_OPENAI_TTS_VOICE or "nova").strip().strip("\"'")
        inst = (config.AZURE_OPENAI_TTS_INSTRUCTIONS or "").strip()
        if is_filler:
            inst = (
                f"{inst} This is a very short holding phrase; use the same voice and timbre "
                "as the main assistant reply, not a different character."
            ).strip()
        body = {
            "model": config.AZURE_OPENAI_TTS_DEPLOYMENT,
            "input": input_text[:TTS_INPUT_MAX_CHARS],
            "voice": voice,
            "response_format": "wav",
        }
        if inst:
            body["instructions"] = inst[:4096]
        return body

    def synthesize_segment_pcm(self, text: str, *, is_filler: bool = False) -> bytes:
        text = (text or "").strip()
        if not text:
            return b""

        with self._lock:
            if self._cancelled:
                return b""

        body = self._speech_json_body(text, is_filler=is_filler)
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

    def synthesize_full_turn_pcm(self, text: str) -> bytes:
        """
        One neural read per part; parts only if text exceeds the API input limit.
        Uses the same WAV path as segment synthesis.
        """
        pieces = _split_tts_input(text)
        if not pieces:
            return b""

        out = bytearray()
        for part in pieces:
            with self._lock:
                if self._cancelled:
                    break
            chunk = self.synthesize_segment_pcm(part)
            if not chunk:
                break
            out.extend(chunk)
        return bytes(out)

"""
TTS via Azure OpenAI gpt-4o-mini-tts (audio/speech API).

- HTTP **streaming** (`stream=True`): read Transfer-Encoding chunks from Azure, resample
  PCM (API default 24 kHz int16 LE for `response_format=pcm`) to **16 kHz** for the client.
- `iter_pcm16_chunks_from_http_stream` yields 16 kHz PCM pieces for piping to WebSockets.

Barge-in: `stop()` sets a flag; in-flight reads exit early and close the response body.
"""
from __future__ import annotations

import io
import threading
import wave
from array import array
from typing import Iterator

import requests

from app.config import config

TTS_INPUT_MAX_CHARS = 4096
# OpenAI / Azure speech `pcm` format: raw s16le mono at this rate (no WAV header).
TTS_STREAM_PCM_INPUT_HZ = 24000
TTS_OUTPUT_PCM_HZ = 16000


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


def _resample_int16_mono(pcm: bytes, src_hz: int, dst_hz: int = TTS_OUTPUT_PCM_HZ) -> bytes:
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
    return _resample_int16_mono(raw, fr, TTS_OUTPUT_PCM_HZ)


class _StreamingLinear24kTo16k:
    """Stateful int16 mono resampler for chunked 24 kHz → 16 kHz (same curve as batch)."""

    __slots__ = ("_in", "_base", "_out_k")

    def __init__(self) -> None:
        self._in: array = array("h")
        self._base = 0
        self._out_k = 0

    def feed(self, data: bytes) -> bytes:
        if not data:
            return b""
        n = (len(data) // 2) * 2
        if n == 0:
            return b""
        add = array("h")
        add.frombytes(data[:n])
        self._in.extend(add)
        return self._drain()

    def flush(self) -> bytes:
        if len(self._in) == 1:
            self._in.append(int(self._in[0]))
        out = self._drain()
        self._in = array("h")
        self._base = 0
        return out

    def _drain(self) -> bytes:
        SRC, DST = TTS_STREAM_PCM_INPUT_HZ, TTS_OUTPUT_PCM_HZ
        out = array("h")
        while True:
            x = self._out_k * SRC / DST
            i0_abs = int(x)
            i1_abs = i0_abs + 1
            loc0 = i0_abs - self._base
            loc1 = i1_abs - self._base
            if loc1 >= len(self._in):
                break
            f = x - i0_abs
            v = int(self._in[loc0] * (1 - f) + self._in[loc1] * f)
            if v > 32767:
                v = 32767
            elif v < -32768:
                v = -32768
            out.append(v)
            self._out_k += 1
        trim_abs = max(0, int(self._out_k * SRC / DST) - 4)
        trim_local = trim_abs - self._base
        if trim_local > 0:
            del self._in[:trim_local]
            self._base += trim_local
        return out.tobytes() if out else b""


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
        self._api_lock = threading.Lock()
        self._cancelled = False

    def stop(self) -> None:
        with self._lock:
            self._cancelled = True

    def reset_for_new_turn(self) -> None:
        with self._lock:
            self._cancelled = False

    def is_stopped(self) -> bool:
        with self._lock:
            return self._cancelled

    def _speech_json_body(
        self,
        input_text: str,
        *,
        is_filler: bool,
        agent_language: str = "en",
        response_format: str = "pcm",
    ) -> dict:
        voice = (config.AZURE_OPENAI_TTS_VOICE or "nova").strip().strip("\"'")
        inst = (config.AZURE_OPENAI_TTS_INSTRUCTIONS or "").strip()
        lang = (agent_language or "en").lower()
        if lang in ("hinglish", "hi", "hindi"):
            inst = (
                f"{inst} Speak in natural Indian Hinglish: Hindi–English code-mix with clear, "
                "authentic Indian pronunciation (not formal textbook Hindi only)."
            ).strip()
        if is_filler:
            inst = (
                f"{inst} This is a very short holding phrase; use the same voice and timbre "
                "as the main assistant reply, not a different character."
            ).strip()
        body: dict = {
            "model": config.AZURE_OPENAI_TTS_DEPLOYMENT,
            "input": input_text[:TTS_INPUT_MAX_CHARS],
            "voice": voice,
            "response_format": response_format,
        }
        if inst:
            body["instructions"] = inst[:4096]
        return body

    def iter_pcm16_chunks_from_http_stream(
        self,
        text: str,
        *,
        agent_language: str = "en",
        is_filler: bool = False,
    ) -> Iterator[bytes]:
        """
        Synchronous generator: one or more HTTP streaming calls (4096-char split), each yielding
        16 kHz int16 mono PCM chunks suitable for WebSocket `audio_delta` payloads.
        """
        text = (text or "").strip()
        if not text:
            return
        pieces = _split_tts_input(text)
        for part in pieces:
            with self._lock:
                if self._cancelled:
                    break
            yield from self._iter_stream_one_part(part, agent_language=agent_language, is_filler=is_filler)

    def _iter_stream_one_part(
        self,
        part: str,
        *,
        agent_language: str,
        is_filler: bool,
    ) -> Iterator[bytes]:
        body = self._speech_json_body(
            part, is_filler=is_filler, agent_language=agent_language, response_format="pcm"
        )
        resampler = _StreamingLinear24kTo16k()
        pending = bytearray()

        with self._api_lock:
            with self._lock:
                if self._cancelled:
                    return
            try:
                r = self._session.post(
                    self._url,
                    json=body,
                    headers={"Content-Type": "application/json"},
                    timeout=120,
                    stream=True,
                )
            except requests.RequestException:
                return

            try:
                if r.status_code != 200:
                    return

                for raw in r.iter_content(chunk_size=8192):
                    with self._lock:
                        if self._cancelled:
                            break
                    if not raw:
                        continue
                    out16 = resampler.feed(raw)
                    if out16:
                        pending.extend(out16)
                        while len(pending) >= 4096:
                            yield bytes(pending[:4096])
                            del pending[:4096]

                tail = resampler.flush()
                if tail:
                    pending.extend(tail)
                while len(pending) >= 4096:
                    yield bytes(pending[:4096])
                    del pending[:4096]
                if pending:
                    yield bytes(pending)
            finally:
                try:
                    r.close()
                except Exception:
                    pass

    def synthesize_segment_pcm(
        self, text: str, *, is_filler: bool = False, agent_language: str = "en"
    ) -> bytes:
        """Non-streaming WAV path (e.g. scripts); full buffer in memory."""
        text = (text or "").strip()
        if not text:
            return b""

        with self._lock:
            if self._cancelled:
                return b""

        body = self._speech_json_body(
            text, is_filler=is_filler, agent_language=agent_language, response_format="wav"
        )
        with self._api_lock:
            with self._lock:
                if self._cancelled:
                    return b""
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

    def synthesize_full_turn_pcm(
        self,
        text: str,
        *,
        agent_language: str = "en",
        is_filler: bool = False,
    ) -> bytes:
        """Buffer entire streamed output (tests / callers that still want bytes)."""
        return b"".join(
            self.iter_pcm16_chunks_from_http_stream(
                text, agent_language=agent_language, is_filler=is_filler
            )
        )

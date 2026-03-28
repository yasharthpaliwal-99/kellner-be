"""
Azure Speech SDK TTS — synthesize to a temp WAV, return raw PCM (16 kHz mono) for the client.
Supports stop_speaking_async for barge-in.
"""
from __future__ import annotations

import io
import os
import tempfile
import threading
import wave
from typing import Optional

import azure.cognitiveservices.speech as speechsdk

from app.config import config

_SSML = """<speak version='1.0' xml:lang='en-IN'>
  <voice name='{voice}'>{text}</voice>
</speak>"""


class StreamingTTSService:
    def __init__(self) -> None:
        if not config.AZURE_SPEECH_KEY or not config.AZURE_SPEECH_REGION:
            raise ValueError("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env")
        self._speech_config = speechsdk.SpeechConfig(
            subscription=config.AZURE_SPEECH_KEY,
            region=config.AZURE_SPEECH_REGION,
        )
        self._speech_config.speech_synthesis_voice_name = config.AZURE_TTS_VOICE_NAME
        self._speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
        )
        self._lock = threading.Lock()
        self._active_synth: Optional[speechsdk.SpeechSynthesizer] = None

    def stop(self) -> None:
        with self._lock:
            synth = self._active_synth
        if synth is None:
            return
        try:
            synth.stop_speaking_async().get()
        except Exception:
            pass
        finally:
            with self._lock:
                if self._active_synth is synth:
                    self._active_synth = None

    def synthesize_segment_pcm(self, text: str) -> bytes:
        """Blocking: one SSML utterance → raw PCM bytes (mono 16 kHz)."""
        safe = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        ssml = _SSML.format(voice=config.AZURE_TTS_VOICE_NAME, text=safe)

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=path)
        synth = speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=audio_config,
        )
        with self._lock:
            self._active_synth = synth
        try:
            result = synth.speak_ssml_async(ssml).get()
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                return b""
            with open(path, "rb") as f:
                wav = f.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
            with self._lock:
                if self._active_synth is synth:
                    self._active_synth = None

        if not wav:
            return b""
        try:
            with wave.open(io.BytesIO(wav), "rb") as wf:
                return wf.readframes(wf.getnframes())
        except Exception:
            if len(wav) > 44 and wav[:4] == b"RIFF":
                return wav[44:]
            return wav

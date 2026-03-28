"""
Long-lived streaming STT: push PCM continuously; `recognized` fires per utterance (silence segmentation).
"""
from __future__ import annotations

from typing import Callable, Optional

import azure.cognitiveservices.speech as speechsdk

from app.config import config


class STTContinuousSession:
    """
    16 kHz 16-bit mono PCM via write(). Call start() once, write until close().
    on_recognized(text) is invoked for each finalized phrase (SDK thread).
    """

    def __init__(
        self,
        on_partial: Optional[Callable[[str], None]] = None,
        on_recognized: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not config.AZURE_SPEECH_KEY or not config.AZURE_SPEECH_REGION:
            raise ValueError("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env")
        self._on_partial = on_partial
        self._on_recognized_cb = on_recognized

        stream_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000,
            bits_per_sample=16,
            channels=1,
        )
        self._push = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self._push)

        speech_config = speechsdk.SpeechConfig(
            subscription=config.AZURE_SPEECH_KEY,
            region=config.AZURE_SPEECH_REGION,
        )
        speech_config.speech_recognition_language = "en-IN"

        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )
        self._recognizer.recognizing.connect(self._on_recognizing)
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.canceled.connect(self._on_canceled)

        self._started = False
        self._closed = False

    def _on_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech and self._on_partial:
            t = evt.result.text or ""
            if t:
                self._on_partial(t)

    def _on_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            t = (evt.result.text or "").strip()
            if t and self._on_recognized_cb:
                self._on_recognized_cb(t)

    def _on_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        pass

    def start(self) -> None:
        if self._started or self._closed:
            return
        self._recognizer.start_continuous_recognition_async().get()
        self._started = True

    def write(self, data: bytes) -> None:
        if data and not self._closed:
            self._push.write(data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._push.close()
        try:
            self._recognizer.stop_continuous_recognition_async().get()
        except Exception:
            pass

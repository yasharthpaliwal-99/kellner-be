"""
Long-lived streaming STT: push PCM continuously; `recognized` fires per utterance (silence segmentation).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import azure.cognitiveservices.speech as speechsdk

from app.config import config

logger = logging.getLogger(__name__)


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
        reason = evt.result.reason
        if reason == speechsdk.ResultReason.RecognizedSpeech:
            t = (evt.result.text or "").strip()
            if t and self._on_recognized_cb:
                logger.warning("stt_recognized text_len=%s preview=%r", len(t), t[:120])
                self._on_recognized_cb(t)
            elif not t:
                logger.warning("stt_recognized_empty (silence or no speech)")
            return
        if reason == speechsdk.ResultReason.NoMatch:
            logger.warning("stt_no_match (no speech detected for this segment)")
            return
        logger.warning("stt_recognized_other reason=%s", reason)

    def _on_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        details = evt.cancellation_details
        logger.warning(
            "stt_canceled reason=%s error=%s",
            details.reason,
            (details.error_details or "").strip(),
        )

    def start(self) -> None:
        if self._started or self._closed:
            return
        self._recognizer.start_continuous_recognition_async().get()
        self._started = True
        logger.warning("stt_session_started region=%s", config.AZURE_SPEECH_REGION)

    def write(self, data: bytes) -> None:
        if data and not self._closed:
            self._push.write(data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.warning("stt_session_closing")
        self._push.close()
        try:
            self._recognizer.stop_continuous_recognition_async().get()
        except Exception as exc:
            logger.warning("stt_session_stop_failed: %s", exc)

import asyncio
import base64
import json
import os
import random
import tempfile
import time
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import config
from app.services.device_auth_service import validate_device_session
from app.services.face_local_service import customer_belongs_to_hotel
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService
from app.services.stt_continuous_service import STTContinuousSession
from app.services.stt_service import STTService
from app.services.session_context import ConversationSession, attach_session, reset_session
from app.services.tts_streaming_service import StreamingTTSService

router = APIRouter()

retrieval_service = RetrievalService()
llm_service = LLMService()
stt_service = STTService()
streaming_tts = StreamingTTSService()


def _log_assistant_task(task: asyncio.Task) -> None:
    if not task.done() or task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        import traceback

        traceback.print_exception(type(exc), exc, exc.__traceback__)


async def _run_assistant_turn(
    websocket: WebSocket,
    history: list,
    transcript: str,
    turn_id: int,
    conv_session: ConversationSession,
    enable_filler: bool = True,
) -> None:
    if not transcript.strip():
        return

    streaming_tts.reset_for_new_turn()

    await websocket.send_json({
        "type": "transcript",
        "text": transcript,
        "turn_id": turn_id,
    })

    # UX filler: optional single generic line while tools + LLM run.
    # Filler must NOT be added to `full_segments/history`.
    filler_task: Optional[asyncio.Task] = None
    real_output_started = asyncio.Event()
    tool_started = asyncio.Event()
    if enable_filler:
        async def _filler_controller() -> None:
            try:
                await tool_started.wait()
                if real_output_started.is_set():
                    return
                phrases = [
                    "Alright, I got this.",
                    "Let me check that for you.",
                    "One moment, please.",
                    "Sure, checking now.",
                    "I am on it.",
                    "Got it, give me a second.",
                    "Let me verify that.",
                    "Thanks, checking this now.",
                    "Okay, I am checking.",
                    "Let me quickly confirm that.",
                ]
                phrase = random.choice(phrases)
                pcm = await asyncio.to_thread(
                    streaming_tts.synthesize_segment_pcm, phrase, is_filler=True
                )
                if real_output_started.is_set() or not pcm:
                    return

                await websocket.send_json({
                    "type": "assistant_text_delta",
                    "text": phrase,
                    "turn_id": turn_id,
                })

                chunk_size = 4096
                for i in range(0, len(pcm), chunk_size):
                    chunk = pcm[i : i + chunk_size]
                    await websocket.send_json({
                        "type": "audio_delta",
                        "turn_id": turn_id,
                        "b64": base64.b64encode(chunk).decode("ascii"),
                    })

            except asyncio.CancelledError:
                # Do not call streaming_tts.stop() here: it would cancel real narration.
                return

        filler_task = asyncio.create_task(_filler_controller())

    # ContextVar must be set before _build_messages + resolve_tools so tools + profile use real hotel/customer.
    ctx_token = attach_session(conv_session)
    context = retrieval_service.retrieve(transcript)
    messages = llm_service._build_messages(transcript, history, context)
    menu_recommendations: list = []
    tools_called = False
    loop = asyncio.get_running_loop()
    try:
        def _on_first_tool_call() -> None:
            loop.call_soon_threadsafe(tool_started.set)

        messages, direct, menu_recommendations, tools_called = await asyncio.to_thread(
            llm_service.resolve_tools, messages, _on_first_tool_call
        )
    except Exception as e:
        if "431" in str(e) or "tool" in str(e).lower():
            messages = llm_service._build_messages(transcript, history, context)
            direct = None
            menu_recommendations = []
            tools_called = False
        else:
            await websocket.send_json({"type": "error", "message": str(e), "turn_id": turn_id})
            return
    finally:
        reset_session(ctx_token)

    if not tools_called and filler_task and not filler_task.done():
        filler_task.cancel()

    if menu_recommendations:
        await websocket.send_json({
            "type": "recommendations",
            "turn_id": turn_id,
            "items": menu_recommendations,
        })

    text_q: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    async def llm_producer() -> None:
        try:
            async for seg in llm_service.astream_tts_segments_after_tools(messages, direct):
                if seg.strip():
                    await text_q.put(seg.strip())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await text_q.put(("__error__", str(exc)))
        finally:
            try:
                await text_q.put(SENTINEL)
            except Exception:
                pass

    producer_task = asyncio.create_task(llm_producer())
    full_segments: list[str] = []
    t0 = time.perf_counter()
    first_text_segment = True
    turn_had_error = False

    try:
        while True:
            item = await text_q.get()
            if item is SENTINEL:
                break
            if isinstance(item, tuple) and item[0] == "__error__":
                turn_had_error = True
                await websocket.send_json({
                    "type": "error",
                    "message": item[1],
                    "turn_id": turn_id,
                })
                break

            seg = item
            full_segments.append(seg)

            if first_text_segment:
                # First LLM chunk arrived; stop filler (we still buffer text until the stream ends).
                real_output_started.set()
                if filler_task and not filler_task.done():
                    filler_task.cancel()
                first_text_segment = False

        # Full reply first: one text message, then one TTS pass (stable voice for short answers).
        if not turn_had_error and full_segments:
            full_text = " ".join(full_segments).strip()
            if full_text:
                await websocket.send_json({
                    "type": "assistant_text_delta",
                    "text": full_text,
                    "turn_id": turn_id,
                })
                t_tts = time.perf_counter()
                pcm = await asyncio.to_thread(streaming_tts.synthesize_full_turn_pcm, full_text)
                print(
                    f"[TIMING] full-turn TTS synth: {time.perf_counter() - t_tts:.2f}s "
                    f"(turn {turn_id}, {len(full_text)} chars)"
                )
                print(f"[TIMING] Total to first audio ready: {time.perf_counter() - t0:.2f}s")

                chunk_size = 4096
                for i in range(0, len(pcm), chunk_size):
                    chunk = pcm[i : i + chunk_size]
                    await websocket.send_json({
                        "type": "audio_delta",
                        "turn_id": turn_id,
                        "b64": base64.b64encode(chunk).decode("ascii"),
                    })
    except asyncio.CancelledError:
        streaming_tts.stop()
        producer_task.cancel()
        raise
    finally:
        if not producer_task.done():
            producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass
        if filler_task and not filler_task.done():
            filler_task.cancel()

    if full_segments:
        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": " ".join(full_segments)})
        if len(history) > 20:
            history = history[-20:]

    await websocket.send_json({"type": "done", "turn_id": turn_id})
    print(f"[TIMING] Turn {turn_id} total: {time.perf_counter() - t0:.2f}s")


async def _batch_wav_turn(
    websocket: WebSocket,
    history: list,
    audio_data: bytes,
    turn_id: int,
    conv_session: ConversationSession,
) -> None:
    t0 = time.perf_counter()
    suffix = ".wav" if audio_data[:4] == b"RIFF" else ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    try:
        transcript = await asyncio.to_thread(stt_service.transcribe, tmp_path)
    finally:
        os.unlink(tmp_path)
    print(f"[TIMING] STT (batch): {time.perf_counter() - t0:.2f}s → '{transcript}'")
    await _run_assistant_turn(
        websocket,
        history,
        transcript,
        turn_id,
        conv_session,
        enable_filler=False,
    )


@router.websocket("/ws/conversation")
async def ws_conversation(websocket: WebSocket):
    """
    Voice session:
      - Continuous PCM16 16kHz mono (binary frames) → STT segments on silence (SDK).
      - Optional `{"type":"interrupt"}` → stop TTS + cancel in-flight assistant turn.
      - Legacy: single binary WAV (RIFF…) → batch STT.

    Server → client:
      transcript, recommendations (optional, after menu search), assistant_text_delta,
      audio_delta (raw PCM base64), done, error (all with turn_id where relevant).
    """
    sid = (websocket.query_params.get("session_id") or "").strip()
    sess_doc, sess_err = validate_device_session(sid, role="device")
    if sess_err:
        await websocket.close(code=1008, reason=sess_err)
        return

    await websocket.accept()
    hid = int(sess_doc.get("hotel_id") or config.DEFAULT_HOTEL_ID)
    customer_id = int(config.DEFAULT_CUSTOMER_ID)
    cid_raw = (websocket.query_params.get("customer_id") or "").strip()
    if cid_raw:
        try:
            c = int(cid_raw)
            if c > 0 and customer_belongs_to_hotel(c, hid):
                customer_id = c
        except ValueError:
            pass
    conv_session = ConversationSession(
        session_id=str(uuid.uuid4()),
        hotel_id=hid,
        customer_id=customer_id,
    )
    history: list = []
    turn_seq = 0
    assistant_task: Optional[asyncio.Task] = None
    stt: Optional[STTContinuousSession] = None
    loop = asyncio.get_running_loop()

    async def cancel_assistant() -> None:
        nonlocal assistant_task
        streaming_tts.stop()
        if assistant_task and not assistant_task.done():
            assistant_task.cancel()
            try:
                await assistant_task
            except asyncio.CancelledError:
                pass
        assistant_task = None

    def schedule_utterance(text: str) -> None:
        if not (text or "").strip():
            return

        async def _go() -> None:
            nonlocal assistant_task, turn_seq
            await cancel_assistant()
            turn_seq += 1
            tid = turn_seq
            assistant_task = asyncio.create_task(
                _run_assistant_turn(
                    websocket,
                    history,
                    text,
                    tid,
                    conv_session,
                    enable_filler=True,
                )
            )
            assistant_task.add_done_callback(_log_assistant_task)

        asyncio.run_coroutine_threadsafe(_go(), loop)

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "interrupt":
                    await cancel_assistant()
                    await websocket.send_json({"type": "interrupted", "turn_id": turn_seq})
                    continue
                if data.get("type") == "voice_session":
                    if stt is None:
                        stt = STTContinuousSession(
                            on_partial=lambda t: asyncio.run_coroutine_threadsafe(
                                websocket.send_json(
                                    {"type": "transcript_partial", "text": t, "turn_id": turn_seq}
                                ),
                                loop,
                            ),
                            on_recognized=schedule_utterance,
                        )
                        await asyncio.to_thread(stt.start)
                    continue

            if msg.get("bytes") is not None:
                b = msg["bytes"]
                if b[:4] == b"RIFF" and stt is None:
                    await cancel_assistant()
                    turn_seq += 1
                    await _batch_wav_turn(websocket, history, b, turn_seq, conv_session)
                    continue

                if stt is None:
                    stt = STTContinuousSession(
                        on_partial=lambda t: asyncio.run_coroutine_threadsafe(
                            websocket.send_json(
                                {"type": "transcript_partial", "text": t, "turn_id": turn_seq}
                            ),
                            loop,
                        ),
                        on_recognized=schedule_utterance,
                    )
                    await asyncio.to_thread(stt.start)
                stt.write(b)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await cancel_assistant()
        if stt is not None:
            stt.close()

import asyncio
import base64
import json
import logging
import os
import random
import tempfile
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import config
from app.db.transactions import insert_transaction_row
from app.services.device_auth_service import agent_language_for_session, validate_device_session
from app.services.face_local_service import customer_belongs_to_hotel
from app.services.llm_service import LLMService
from app.services.response_format import (
    append_format_instruction,
    assistant_history_content,
    detect_response_mode,
    history_after_show,
    parse_show_payload,
    parse_speak_show,
)
from app.services.retrieval_service import RetrievalService
from app.services.stt_continuous_service import STTContinuousSession
from app.services.stt_service import STTService
from app.services.session_context import ConversationSession, attach_session, reset_session
from app.services.tts_streaming_service import StreamingTTSService

router = APIRouter()

# Words spoken aloud per assistant turn (UI/history still get full streamed text).
ASSISTANT_TTS_MAX_WORDS = 30


def _complete_sentences_within_budget(text: str, max_words: int) -> str:
    """Return the longest prefix of `text` that ends on a sentence boundary and fits in max_words.

    Sentence boundaries: `.` `?` `!` `।` (Hindi purna-viram).
    If even the first sentence exceeds the budget, return it anyway (better one full sentence
    than nothing). If no sentence boundary found, return nothing (caller skips TTS for this chunk).
    """
    if max_words <= 0 or not (text or "").strip():
        return ""
    boundaries = []
    for i, ch in enumerate(text):
        if ch in ".?!।":
            boundaries.append(i)
    if not boundaries:
        return ""
    best = ""
    for end in boundaries:
        candidate = text[: end + 1].strip()
        if len(candidate.split()) <= max_words:
            best = candidate
        else:
            break
    else:
        best = text[: boundaries[-1] + 1].strip()
    # If nothing fit, use first sentence anyway so TTS says *something* complete.
    if not best and boundaries:
        best = text[: boundaries[0] + 1].strip()
    return best or ""


logger = logging.getLogger(__name__)

# Synthetic user message for LLM on `guest_greeting` (Call waiter / proactive welcome). Not shown as user text when source=guest_greeting.
_GUEST_GREETING_EN = (
    "The guest has just started a voice session. Greet them warmly in one or two short sentences, "
    "introduce yourself as the hotel assistant, and offer help with the menu, ordering, or room service. "
    "Do not ask them to repeat anything; keep it natural and welcoming."
)
_GUEST_GREETING_HINGLISH = (
    "Guest abhi voice session shuru kiya hai. Unhe warmly greet karo — ek ya do short sentences mein, "
    "khud ko hotel assistant batao, aur menu, order ya room service mein help offer karo. "
    "Natural Indian Hinglish mein bolo. Kuch repeat karne ko mat kaho."
)


def _guest_greeting_prompt(agent_language: str) -> str:
    custom = (config.GUEST_GREETING_PROMPT or "").strip()
    if custom:
        return custom
    lang = (agent_language or "en").lower()
    if lang == "hinglish":
        return _GUEST_GREETING_HINGLISH
    return _GUEST_GREETING_EN


retrieval_service = RetrievalService()
llm_service = LLMService()
stt_service = STTService()
streaming_tts = StreamingTTSService()

# Holding phrases while tools + LLM run (not added to history). Picked by hotel `agent_language`.
FILLERS_EN = [
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
    "Just a second.",
    "Working on that now.",
    "Bear with me for a moment.",
    "I will look that up right away.",
    "Give me just a moment.",
    "Checking the details for you.",
    "Almost there.",
    "Let me pull that up.",
    "One sec while I check.",
    "On it — one moment.",
]
FILLERS_HINGLISH = [
    "Thik hai, abhi check kar raha hoon.",
    "Ek second, dekh raha hoon.",
    "Sure, abhi verify kar leta hoon.",
    "Haanji, menu dekh raha hoon.",
    "Theek hai, ruko zara.",
    "Samajh gaya, abhi dekhta hoon.",
    "Just a moment, check kar raha hoon.",
    "Ji, abhi confirm karta hoon.",
    "Ek minute, details nikal raha hoon.",
    "Bilkul, abhi dekh leta hoon.",
    "On it — thoda wait kariye.",
    "Haan, abhi check karta hoon.",
    "Ruko, abhi pata karta hoon.",
    "Theek, abhi dekh raha hoon aapke liye.",
    "Sure, ek second.",
    "Abhi pull kar raha hoon.",
    "Thik, verify ho raha hai.",
    "Bas, almost ho gaya check.",
    "Ji haan, dekh raha hoon.",
    "Theek hai, ek pal.",
]


async def _send_pcm_audio_deltas(websocket: WebSocket, pcm: bytes, turn_id: int) -> None:
    chunk_size = 4096
    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i : i + chunk_size]
        await websocket.send_json({
            "type": "audio_delta",
            "turn_id": turn_id,
            "b64": base64.b64encode(chunk).decode("ascii"),
        })


_TTS_STREAM_END = object()


async def _stream_tts_http_to_websocket(
    websocket: WebSocket,
    turn_id: int,
    text: str,
    *,
    agent_language: str,
    is_filler: bool,
    tts: StreamingTTSService,
    first_audio_perf: Optional[list] = None,
    cancel_if_event: Optional[asyncio.Event] = None,
) -> None:
    """Read TTS HTTP body in a worker thread; forward 16 kHz PCM to WebSocket as it arrives."""
    text = (text or "").strip()
    if not text:
        return
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=64)

    def worker() -> None:
        try:
            for pcm_chunk in tts.iter_pcm16_chunks_from_http_stream(
                text, agent_language=agent_language, is_filler=is_filler
            ):
                fut = asyncio.run_coroutine_threadsafe(q.put(pcm_chunk), loop)
                fut.result(timeout=180)
        except Exception as e:
            logger.exception("tts_stream_worker_failed: %s", e)
        finally:
            try:
                asyncio.run_coroutine_threadsafe(q.put(_TTS_STREAM_END), loop).result(timeout=15)
            except Exception:
                pass

    worker_task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        while True:
            item = await q.get()
            if item is _TTS_STREAM_END:
                break
            if cancel_if_event is not None and cancel_if_event.is_set():
                tts.stop()
                break
            if tts.is_stopped():
                break
            if first_audio_perf is not None and first_audio_perf[0] is None:
                first_audio_perf[0] = time.perf_counter()
            await _send_pcm_audio_deltas(websocket, item, turn_id)
    finally:
        await worker_task


async def _send_show_and_structured(
    websocket: WebSocket,
    mode: str,
    turn_id: int,
    show_text: str,
) -> Optional[dict]:
    """
    Emit assistant_text_delta (JSON string if SHOW parses) and assistant_structured when JSON is valid.
    Returns parsed payload dict or None.
    """
    payload, canonical = parse_show_payload(show_text)
    logger.warning(
        "speak_show show_payload turn_id=%s mode=%s parsed=%s show_preview=%r",
        turn_id,
        mode,
        payload is not None,
        (show_text or "")[:280],
    )
    if (show_text or "").strip():
        text_out = canonical if payload is not None else show_text.strip()
        await websocket.send_json({
            "type": "assistant_text_delta",
            "text": text_out,
            "turn_id": turn_id,
        })
    if payload is not None:
        await websocket.send_json({
            "type": "assistant_structured",
            "mode": mode,
            "turn_id": turn_id,
            "payload": payload,
        })
    return payload


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
    *,
    stt_seconds: Optional[float] = None,
    proactive_source: Optional[str] = None,
) -> None:
    if not transcript.strip():
        return

    # For proactive turns, transcript is the LLM instruction; history uses a short stub instead of the full prompt.
    user_history_stub = transcript if not proactive_source else "(Guest session)"

    transaction_id = str(uuid.uuid4())
    t_turn_start = time.perf_counter()
    timing_events: List[dict] = []
    full_segments: list[str] = []
    retrieval_seconds: Optional[float] = None
    llm_tools_wall_seconds: Optional[float] = None
    llm_stream_wait_seconds = 0.0
    tts_seconds = 0.0
    first_audio_seconds: Optional[float] = None
    had_error = False
    menu_rec_count = 0
    tools_called = False
    filler_task: Optional[asyncio.Task] = None
    producer_task: Optional[asyncio.Task] = None

    streaming_tts.reset_for_new_turn()

    _tr_payload: dict = {
        "type": "transcript",
        "text": "" if proactive_source else transcript,
        "turn_id": turn_id,
    }
    if proactive_source:
        _tr_payload["source"] = proactive_source
    await websocket.send_json(_tr_payload)

    # UX filler: optional single generic line while tools + LLM run.
    # Filler must NOT be added to `full_segments/history`.
    # First real LLM segment cancels filler; TTS runs per segment (not one full-turn batch).
    real_output_started = asyncio.Event()
    tool_started = asyncio.Event()
    if enable_filler:
        async def _filler_controller() -> None:
            try:
                await tool_started.wait()
                if real_output_started.is_set():
                    return
                lang = (conv_session.agent_language or "en").lower()
                if lang not in ("en", "hinglish"):
                    lang = "en"
                phrases = FILLERS_HINGLISH if lang == "hinglish" else FILLERS_EN
                phrase = random.choice(phrases)
                if real_output_started.is_set():
                    return

                await websocket.send_json({
                    "type": "assistant_text_delta",
                    "text": phrase,
                    "turn_id": turn_id,
                })

                await _stream_tts_http_to_websocket(
                    websocket,
                    turn_id,
                    phrase,
                    agent_language=lang,
                    is_filler=True,
                    tts=streaming_tts,
                    cancel_if_event=real_output_started,
                )

            except asyncio.CancelledError:
                # Release streaming HTTP lock so main-line TTS can run; interrupt uses stop() elsewhere.
                streaming_tts.stop()
                streaming_tts.reset_for_new_turn()
                return

        filler_task = asyncio.create_task(_filler_controller())

    ctx_token = None
    try:
        # ContextVar must stay set through resolve_tools + streaming LLM so tools/profile stay correct.
        ctx_token = attach_session(conv_session)
        tr0 = time.perf_counter()
        context = retrieval_service.retrieve(transcript)
        retrieval_seconds = time.perf_counter() - tr0
        messages = llm_service._build_messages(transcript, history, context)
        menu_recommendations: list = []
        loop = asyncio.get_running_loop()

        def _on_first_tool_call() -> None:
            loop.call_soon_threadsafe(tool_started.set)

        try:
            lt0 = time.perf_counter()
            messages, direct, menu_recommendations, tools_called, tools_used = await asyncio.to_thread(
                llm_service.resolve_tools, messages, _on_first_tool_call, timing_events
            )
            llm_tools_wall_seconds = time.perf_counter() - lt0
        except Exception as e:
            if "431" in str(e) or "tool" in str(e).lower():
                messages = llm_service._build_messages(transcript, history, context)
                direct = None
                menu_recommendations = []
                tools_called = False
                tools_used = []
            else:
                had_error = True
                await websocket.send_json({"type": "error", "message": str(e), "turn_id": turn_id})
                return

        menu_rec_count = len(menu_recommendations)
        mode = detect_response_mode(tools_used, menu_rec_count)
        logger.warning(
            "speak_show mode_selected turn_id=%s mode=%s tools_called=%s tools_used=%s rec_count=%s direct=%s",
            turn_id,
            mode,
            tools_called,
            tools_used,
            menu_rec_count,
            direct is not None,
        )

        if not tools_called and filler_task and not filler_task.done():
            filler_task.cancel()

        if menu_recommendations:
            await websocket.send_json({
                "type": "recommendations",
                "turn_id": turn_id,
                "items": menu_recommendations,
            })

        lang = (conv_session.agent_language or "en").lower()
        if lang not in ("en", "hinglish"):
            lang = "en"

        turn_had_error = False

        async def _phase2_tts_from_plain(full_text: str) -> None:
            nonlocal tts_seconds, first_audio_seconds
            if not full_text.strip():
                return
            spoken = _complete_sentences_within_budget(full_text, ASSISTANT_TTS_MAX_WORDS)
            if not spoken:
                return
            tt0 = time.perf_counter()
            first_audio_ref: list = [None]
            await _stream_tts_http_to_websocket(
                websocket,
                turn_id,
                spoken,
                agent_language=lang,
                is_filler=False,
                tts=streaming_tts,
                first_audio_perf=first_audio_ref,
            )
            tts_seconds = time.perf_counter() - tt0
            if first_audio_ref[0] is not None:
                first_audio_seconds = first_audio_ref[0] - t_turn_start

        # Tagged reply from tool phase (rare; no format appendix was applied).
        if direct is not None:
            sp0, sh0, ok_tag_direct = parse_speak_show(direct)
            logger.warning(
                "speak_show direct_path turn_id=%s mode=%s tagged=%s raw_preview=%r",
                turn_id,
                mode,
                ok_tag_direct,
                (direct or "")[:280],
            )
            if ok_tag_direct:
                if mode != "none":
                    await websocket.send_json({
                        "type": "assistant_reply_mode",
                        "mode": mode,
                        "turn_id": turn_id,
                    })
                real_output_started.set()
                if filler_task and not filler_task.done():
                    filler_task.cancel()
                show_ui = (sh0 or "").strip()
                pl_direct: Optional[dict] = None
                if show_ui:
                    pl_direct = await _send_show_and_structured(websocket, mode, turn_id, show_ui)
                full_segments.append(
                    json.dumps(pl_direct, ensure_ascii=False)
                    if pl_direct is not None
                    else (show_ui or direct.strip())
                )
                speak_line = (sp0 or "").strip()
                if speak_line:
                    await _phase2_tts_from_plain(speak_line)
                assistant_history_text = history_after_show(True, sh0, direct, pl_direct)
                history.append({"role": "user", "content": user_history_stub})
                history.append({"role": "assistant", "content": assistant_history_text})
                if len(history) > 20:
                    history = history[-20:]
                await websocket.send_json({"type": "done", "turn_id": turn_id})
                return

        # Structured mode: raw stream so [SPEAK]/[SHOW] are not split by sentence chunking.
        if mode != "none":
            await websocket.send_json({
                "type": "assistant_reply_mode",
                "mode": mode,
                "turn_id": turn_id,
            })
            msgs_tagged = append_format_instruction(messages, mode)
            raw_parts: list[str] = []
            try:
                t_raw_stream = time.perf_counter()
                async for delta in llm_service.astream_raw_text_after_tools(
                    msgs_tagged, None, timing_events
                ):
                    if delta:
                        raw_parts.append(delta)
                llm_stream_wait_seconds += time.perf_counter() - t_raw_stream
            except asyncio.CancelledError:
                streaming_tts.stop()
                raise
            except Exception as exc:
                turn_had_error = True
                had_error = True
                await websocket.send_json({"type": "error", "message": str(exc), "turn_id": turn_id})
            else:
                full_raw = "".join(raw_parts)
                assistant_history_text = None
                sp1, sh1, ok_tag = parse_speak_show(full_raw)
                logger.warning(
                    "speak_show structured_path turn_id=%s mode=%s tagged=%s raw_preview=%r",
                    turn_id,
                    mode,
                    ok_tag,
                    (full_raw or "")[:280],
                )
                if ok_tag:
                    real_output_started.set()
                    if filler_task and not filler_task.done():
                        filler_task.cancel()
                    show_ui = (sh1 or "").strip()
                    pl_stream: Optional[dict] = None
                    if show_ui:
                        pl_stream = await _send_show_and_structured(websocket, mode, turn_id, show_ui)
                    full_segments.append(
                        json.dumps(pl_stream, ensure_ascii=False)
                        if pl_stream is not None
                        else (show_ui or full_raw.strip())
                    )
                    speak_line = (sp1 or "").strip()
                    if speak_line and not turn_had_error:
                        await _phase2_tts_from_plain(speak_line)
                    assistant_history_text = history_after_show(True, sh1, full_raw, pl_stream)
                elif full_raw.strip() and not turn_had_error:
                    # Fallback: model did not return tags even though a structured mode was requested.
                    logger.warning(
                        "speak_show fallback_to_legacy turn_id=%s mode=%s raw_preview=%r",
                        turn_id,
                        mode,
                        (full_raw or "")[:280],
                    )
                    # Fallback: legacy sentence chunks + budget TTS (same as default path).
                    real_output_started.set()
                    if filler_task and not filler_task.done():
                        filler_task.cancel()
                    for seg in llm_service._segments_from_plain(full_raw):
                        if not seg.strip():
                            continue
                        s = seg.strip()
                        full_segments.append(s)
                        await websocket.send_json({
                            "type": "assistant_text_delta",
                            "text": s,
                            "turn_id": turn_id,
                        })
                    if full_segments:
                        await _phase2_tts_from_plain(" ".join(full_segments))
                    assistant_history_text = assistant_history_content(False, None, full_raw)
                if assistant_history_text and not turn_had_error:
                    history.append({"role": "user", "content": user_history_stub})
                    history.append({"role": "assistant", "content": assistant_history_text})
                    if len(history) > 20:
                        history = history[-20:]
            if filler_task and not filler_task.done():
                filler_task.cancel()
            await websocket.send_json({"type": "done", "turn_id": turn_id})
            return

        text_q: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        async def llm_producer() -> None:
            try:
                async for seg in llm_service.astream_tts_segments_after_tools(
                    messages, direct, timing_events
                ):
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

        # Phase 1: stream all text deltas to the client (low latency for UI).
        first_text_segment = True
        try:
            while True:
                w0 = time.perf_counter()
                item = await text_q.get()
                llm_stream_wait_seconds += time.perf_counter() - w0
                if item is SENTINEL:
                    break
                if isinstance(item, tuple) and item[0] == "__error__":
                    turn_had_error = True
                    had_error = True
                    await websocket.send_json({
                        "type": "error",
                        "message": item[1],
                        "turn_id": turn_id,
                    })
                    break

                seg = item
                full_segments.append(seg)

                if first_text_segment:
                    real_output_started.set()
                    if filler_task and not filler_task.done():
                        filler_task.cancel()
                    first_text_segment = False

                if not turn_had_error and seg:
                    await websocket.send_json({
                        "type": "assistant_text_delta",
                        "text": seg,
                        "turn_id": turn_id,
                    })
        except asyncio.CancelledError:
            streaming_tts.stop()
            if producer_task and not producer_task.done():
                producer_task.cancel()
            raise
        finally:
            if producer_task and not producer_task.done():
                producer_task.cancel()
            try:
                if producer_task:
                    await producer_task
            except asyncio.CancelledError:
                pass
            if filler_task and not filler_task.done():
                filler_task.cancel()

        # Phase 2: TTS — one synthesis of the first complete sentences within the word budget.
        if not turn_had_error and full_segments:
            await _phase2_tts_from_plain(" ".join(full_segments))

        if full_segments:
            history.append({"role": "user", "content": user_history_stub})
            history.append({"role": "assistant", "content": " ".join(full_segments)})
            if len(history) > 20:
                history = history[-20:]

        await websocket.send_json({"type": "done", "turn_id": turn_id})
    finally:
        if ctx_token is not None:
            reset_session(ctx_token)
        turn_total_seconds = time.perf_counter() - t_turn_start
        assistant_text = " ".join(full_segments)
        row = {
            "transaction_id": transaction_id,
            "session_id": conv_session.session_id,
            "customer_id": conv_session.customer_id,
            "hotel_id": conv_session.hotel_id,
            "turn_id": turn_id,
            "stt_seconds": stt_seconds,
            "retrieval_seconds": retrieval_seconds,
            "llm_tools_wall_seconds": llm_tools_wall_seconds,
            "llm_stream_wait_seconds": llm_stream_wait_seconds,
            "tts_seconds": tts_seconds,
            "first_audio_seconds": first_audio_seconds,
            "turn_total_seconds": turn_total_seconds,
            "tools_called": tools_called,
            "recommendation_count": menu_rec_count,
            "transcript_chars": len(transcript),
            "assistant_chars": len(assistant_text),
            "had_error": had_error,
            "events": timing_events,
        }
        insert_transaction_row(row)
        logger.info(
            "turn_timing transaction_id=%s session_id=%s customer_id=%s hotel_id=%s turn_id=%s "
            "stt_s=%s retrieval_s=%s llm_tools_wall_s=%s llm_stream_wait_s=%s tts_s=%s "
            "first_audio_s=%s total_s=%s tools_called=%s rec_count=%s had_error=%s",
            transaction_id,
            conv_session.session_id,
            conv_session.customer_id,
            conv_session.hotel_id,
            turn_id,
            stt_seconds,
            retrieval_seconds,
            llm_tools_wall_seconds,
            llm_stream_wait_seconds,
            tts_seconds,
            first_audio_seconds,
            turn_total_seconds,
            tools_called,
            menu_rec_count,
            had_error,
        )


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
    stt_s = time.perf_counter() - t0
    print(f"[TIMING] STT (batch): {stt_s:.2f}s → '{transcript}'")
    await _run_assistant_turn(
        websocket,
        history,
        transcript,
        turn_id,
        conv_session,
        enable_filler=False,
        stt_seconds=stt_s,
    )


@router.websocket("/ws/conversation")
async def ws_conversation(websocket: WebSocket):
    """
    Voice session:
      - Continuous PCM16 16kHz mono (binary frames) → STT segments on silence (SDK).
      - Optional `{"type":"interrupt"}` → stop TTS + cancel in-flight assistant turn.
      - `{"type":"guest_greeting"}` (after `voice_session`, e.g. Call waiter) → proactive welcome turn (LLM + TTS).
      - Legacy: single binary WAV (RIFF…) → batch STT.

    Server → client:
      transcript, recommendations (optional, after menu search), assistant_text_delta,
      assistant_reply_mode (optional: bill | order_confirmation | recommendations — sent once
      before structured assistant output for that turn), assistant_structured (optional JSON
      payload when SHOW parses), audio_delta (PCM16 mono **16 kHz** base64; chunks arrive as
      Azure TTS HTTP stream is read, not after a full-buffer wait), done, error (turn_id where relevant).
      transcript may include `"source":"guest_greeting"` with empty `text` for proactive welcome turns.
    """
    sid = (websocket.query_params.get("session_id") or "").strip()
    sess_doc, sess_err = validate_device_session(sid, role="device")
    if sess_err:
        await websocket.close(code=1008, reason=sess_err)
        return

    await websocket.accept()
    hid = int(sess_doc.get("hotel_id") or config.DEFAULT_HOTEL_ID)
    # No customer_id on the socket = anonymous guest — do not fall back to DEFAULT_CUSTOMER_ID
    # (that points at seeded rows like customer_id=1 "Aarav" and wrongly personalises the agent).
    customer_id = 0
    cid_raw = (websocket.query_params.get("customer_id") or "").strip()
    if cid_raw:
        try:
            c = int(cid_raw)
            if c > 0 and customer_belongs_to_hotel(c, hid):
                customer_id = c
        except ValueError:
            pass
    agent_lang = agent_language_for_session(sess_doc)
    if agent_lang not in ("en", "hinglish"):
        agent_lang = "en"
    conv_session = ConversationSession(
        session_id=str(uuid.uuid4()),
        hotel_id=hid,
        customer_id=customer_id,
        agent_language=agent_lang,
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
                if data.get("type") == "guest_greeting":
                    await cancel_assistant()
                    turn_seq += 1
                    tid = turn_seq
                    greet_prompt = _guest_greeting_prompt(conv_session.agent_language)
                    assistant_task = asyncio.create_task(
                        _run_assistant_turn(
                            websocket,
                            history,
                            greet_prompt,
                            tid,
                            conv_session,
                            enable_filler=True,
                            proactive_source="guest_greeting",
                        )
                    )
                    assistant_task.add_done_callback(_log_assistant_task)
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

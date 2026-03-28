"""
Terminal WebSocket test client.
Sends one WAV utterance (batch STT); plays assistant reply from audio_delta PCM chunks.

Usage:
    python scripts/test_ws.py
"""
import asyncio
import base64
import json
import os
import queue
import struct
import subprocess
import tempfile
import threading

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import websockets

WS_URL = "ws://localhost:8000/api/ws/conversation"
SAMPLERATE = 16000


def pcm16_to_wav_bytes(pcm: bytes) -> bytes:
    n = len(pcm)
    hdr = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + n,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        SAMPLERATE,
        SAMPLERATE * 2,
        2,
        16,
        b"data",
        n,
    )
    return hdr + pcm


def play_wav_bytes(data: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(data)
        path = f.name
    try:
        subprocess.run(["afplay", path], check=True)
    except Exception:
        pass
    finally:
        os.unlink(path)


async def conversation_loop():
    print(f"Connecting to {WS_URL}…")
    async with websockets.connect(WS_URL) as ws:
        print("Connected. Press Enter to speak, Enter again to send (batch WAV).\n")

        while True:
            input("[ Press Enter to start recording ]")

            chunks = []
            stop_event = threading.Event()

            def callback(indata, frames, time, status):
                if not stop_event.is_set():
                    chunks.append(indata.copy())

            with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype="int16", callback=callback):
                input("  Recording… Press Enter to stop.")
                stop_event.set()

            if not chunks:
                print("  Nothing recorded.\n")
                continue

            audio = np.concatenate(chunks, axis=0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                wav.write(f.name, SAMPLERATE, audio)
                with open(f.name, "rb") as wf:
                    audio_bytes = wf.read()
                os.unlink(f.name)

            print("  Sending…")
            await ws.send(audio_bytes)

            play_q: queue.Queue = queue.Queue()

            def player():
                while True:
                    item = play_q.get()
                    if item is None:
                        break
                    play_wav_bytes(item)

            player_thread = threading.Thread(target=player, daemon=True)
            player_thread.start()

            pcm_acc = bytearray()

            while True:
                raw = await ws.recv()
                msg = json.loads(raw)

                if msg["type"] == "transcript":
                    print(f"\nYou said: {msg['text']}")

                elif msg["type"] == "assistant_text_delta":
                    print(f"Assistant (chunk): {msg.get('text', '')}")

                elif msg["type"] == "audio_delta":
                    pcm_acc.extend(base64.b64decode(msg["b64"]))

                elif msg["type"] == "done":
                    if pcm_acc:
                        play_q.put(pcm16_to_wav_bytes(bytes(pcm_acc)))
                    play_q.put(None)
                    player_thread.join()
                    print()
                    break

                elif msg["type"] == "error":
                    print(f"Error: {msg['message']}")
                    play_q.put(None)
                    break


if __name__ == "__main__":
    asyncio.run(conversation_loop())

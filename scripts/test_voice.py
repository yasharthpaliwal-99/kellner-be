#!/usr/bin/env python3
"""
Terminal voice test for kellner.

Records from your mic → sends to /api/conversation/voice → plays the response.

Usage:
  python scripts/test_voice.py            # default: http://localhost:8000
  python scripts/test_voice.py --url http://localhost:8000
"""
import argparse
import base64
import os
import subprocess
import sys
import tempfile
import threading

API_URL = "http://localhost:8000/api/conversation/voice"
SAMPLE_RATE = 16000
CHANNELS = 1


def record_until_enter() -> bytes:
    try:
        import numpy as np
        import sounddevice as sd
        import scipy.io.wavfile as wavfile
    except ImportError:
        sys.exit("Install recording deps first:\n  pip install sounddevice scipy numpy")

    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    input("Press Enter to start recording...")
    print("Recording... Press Enter to stop.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", callback=callback):
        input()

    if not frames:
        sys.exit("No audio captured.")

    import numpy as np
    audio = np.concatenate(frames, axis=0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wavfile.write(f.name, SAMPLE_RATE, audio)
        return f.name


def send_to_api(wav_path: str, url: str) -> dict:
    import requests
    print("Sending to API...")
    with open(wav_path, "rb") as f:
        response = requests.post(url, files={"audio": ("input.wav", f, "audio/wav")})
    if response.status_code != 200:
        sys.exit(f"API error {response.status_code}: {response.text}")
    return response.json()


def play_audio(audio_b64: str) -> None:
    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        # afplay is macOS built-in; falls back to aplay (Linux)
        player = "afplay" if sys.platform == "darwin" else "aplay"
        subprocess.run([player, tmp_path], check=True)
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=API_URL)
    args = parser.parse_args()

    wav_path = record_until_enter()
    try:
        result = send_to_api(wav_path, args.url)
    finally:
        os.unlink(wav_path)

    print(f"\nYou said : {result['user_query']}")
    print(f"Assistant: {result['assistant_text']}\n")
    print("Playing response...")
    play_audio(result["audio_base64"])


if __name__ == "__main__":
    main()

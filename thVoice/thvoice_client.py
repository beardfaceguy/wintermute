import sys
import sounddevice as sd
import queue
import json
import requests
import threading
import subprocess
import os
from vosk import Model, KaldiRecognizer
import soundfile as sf
import asyncio
import websockets
import time

MODEL_PATH = "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 48000
BLOCK_SIZE = 1096
CHANNELS = 1
STREAM_URL = "http://localhost:8000/chat/stream"

PIPER_BIN = "./piper/build/piper"
PIPER_MODEL = "./piper/models/en_GB-cori-medium.onnx"
OUTPUT_WAV = "output.wav"

WS_URL = "ws://localhost:8000/ws/chat"

q = queue.Queue()
recognizing = True  # global flag


def audio_callback(indata, frames, time, status):
    if status:
        print("Audio error:", status, file=sys.stderr)
    q.put(bytes(indata))


def recognize_speech():
    global recognizing
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    print("Say something...")
    with sd.RawInputStream(
        device=4,
        latency="high",
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=CHANNELS,
        callback=audio_callback,
    ):
        while True:
            if not recognizing:
                continue  # skip recognition while speaking
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text")
                if text:
                    print(f"You said: {text}")
                    recognizing = False  # block further recognition
                    asyncio.run(handle_query(text))
                    recognizing = True


async def handle_query(text):
    print(f"Connecting to {WS_URL}...")
    async with websockets.connect(WS_URL) as websocket:
        await websocket.send(text)
        print("Sent text, waiting for response...")

        response = ""
        try:
            async for message in websocket:
                print(message, end="", flush=True)
                response += message
        except websockets.exceptions.ConnectionClosed:
            print("\nWebSocket closed by server.")
        print(f"LLM response: {response}")
        print("\nGenerating audio with Piper...")
        synthesize_with_piper(response)


def synthesize_and_play(text):
    try:
        piper_cmd = [
            PIPER_BIN,
            "--model",
            PIPER_MODEL,
            "--output_file",
            OUTPUT_WAV,
        ]
        proc = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE)
        proc.communicate(input=text.encode())

        data, samplerate = sf.read(OUTPUT_WAV)
        sd.play(data, samplerate)
        sd.wait()
        print("Ready for next input...")

    except Exception as e:
        print(f"Error synthesizing response: {e}")


# def synthesize_with_piper(text):
#     output_path = "output.wav"
#     if os.path.exists(output_path):
#         os.remove(output_path)

#     command = [
#         "./piper/build/piper",
#         "--model",
#         "./piper/models/en_GB-cori-medium.onnx",
#         "--output_file",
#         output_path,
#     ]

#     print("Generating audio with Piper...")
#     proc = subprocess.run(
#         command,
#         input=text.encode(),
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         cwd=os.path.dirname(__file__),
#     )

#     if proc.returncode != 0:
#         print("Piper failed:")
#         print(proc.stderr.decode())
#         return

#     # Optional: wait to ensure file is flushed to disk
#     time.sleep(0.2)

#     if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
#         print("Piper did not create a valid output.wav file.")
#         return

#     try:
#         data, samplerate = sf.read(output_path)
#         sd.play(data, samplerate)
#         sd.wait()
#     except Exception as e:
#         print(f"Error playing audio: {e}")


def synthesize_with_piper(text):
    subprocess.run(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", OUTPUT_WAV],
        input=text.encode(),
        check=True,
    )

    if not os.path.exists(OUTPUT_WAV):
        print("Piper did not create a valid output.wav file.")
        return

    data, samplerate = sf.read(OUTPUT_WAV)
    sd.play(data, samplerate)
    sd.wait()

    q.queue.clear()  # 🚫 Clear out echo'd speech that Piper just produced


if __name__ == "__main__":
    try:
        recognize_speech()
    except KeyboardInterrupt:
        print("\nExiting.")

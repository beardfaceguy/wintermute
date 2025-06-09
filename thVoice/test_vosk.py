import sys
import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

# Update this path if your model is in a different folder
MODEL_PATH = "vosk-model-small-en-us-0.15"

# Use 16kHz mono for compatibility
SAMPLE_RATE = 16000
CHANNELS = 1

q = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print("Audio error:", status, file=sys.stderr)
    q.put(bytes(indata))


def main():
    print("Loading model...")
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    print("Listening... Press Ctrl+C to stop.")
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=CHANNELS,
        callback=audio_callback,
    ):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = rec.Result()
                print("Recognized:", json.loads(result).get("text"))
            else:
                partial = rec.PartialResult()
                print("Partial:", json.loads(partial).get("partial"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")

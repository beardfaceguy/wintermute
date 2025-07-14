from pywhispercpp.model import Model

w = Model("models/whisper.cpp/models/ggml-base.en.bin")
result = w.transcribe("models/whisper.cpp/samples/jfk.wav")

print("\n".join([seg.text for seg in result]))

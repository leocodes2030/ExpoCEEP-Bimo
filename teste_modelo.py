import time
import numpy as np
import sounddevice as sd
import sherpa_onnx


MICROFONE = 1
SAMPLE_RATE = 16000
TEMPO_GRAVACAO = 5


print("Carregando Whisper Base INT8...")

inicio = time.time()

recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
    encoder="sherpa-onnx-whisper-base/base-encoder.int8.onnx",
    decoder="sherpa-onnx-whisper-base/base-decoder.int8.onnx",
    tokens="sherpa-onnx-whisper-base/base-tokens.txt",
    language="pt",
    task="transcribe",
    num_threads=4,
    provider="cpu",
)

print(f"Whisper carregado em {time.time() - inicio:.2f} segundos.")

print()
print("Fale durante 5 segundos...")
print()

audio = sd.rec(
    int(TEMPO_GRAVACAO * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=MICROFONE
)

sd.wait()

audio = audio.flatten()

print("Transcrevendo...")

stream = recognizer.create_stream()

stream.accept_waveform(
    SAMPLE_RATE,
    audio
)

recognizer.decode_stream(stream)

texto = stream.result.text.strip()

print()
print("==============================")
print("VOCÊ DISSE:")
print(texto)
print("==============================")

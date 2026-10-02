import time
import sherpa_onnx

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

tempo = time.time() - inicio

print(f"Whisper Base carregado em {tempo:.2f} segundos!")

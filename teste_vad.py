import sounddevice as sd
import sherpa_onnx

MICROFONE = 1
SAMPLE_RATE = 16000

config = sherpa_onnx.VadModelConfig()
config.silero_vad.model = "silero_vad.onnx"
config.sample_rate = SAMPLE_RATE

vad = sherpa_onnx.VoiceActivityDetector(
    config,
    buffer_size_in_seconds=30
)

print("Bimo esperando...")
print("Fale alguma coisa. Ctrl+C para sair.")

with sd.InputStream(
    device=MICROFONE,
    channels=1,
    dtype="float32",
    samplerate=SAMPLE_RATE,
    blocksize=1600
) as stream:

    while True:
        samples, _ = stream.read(1600)
        samples = samples.reshape(-1)

        vad.accept_waveform(samples)

        if vad.is_speech_detected():
            print(">>> FALANDO")

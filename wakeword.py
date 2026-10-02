import sounddevice as sd

MICROFONE = 1
SAMPLE_RATE = 16000

print("Testando microfone...")
print("Fale por 5 segundos.")

try:
    with sd.InputStream(
        device=MICROFONE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    ) as stream:

        for i in range(50):
            audio, overflowed = stream.read(1600)

            volume = abs(audio).mean()

            print(f"Volume: {volume:.5f}")

except Exception as e:
    print("ERRO:")
    print(e)

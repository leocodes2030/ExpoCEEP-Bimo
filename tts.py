import subprocess
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal
import os

PIPER = r"C:\BEMO\piper\piper.exe"
MODEL = r"C:\BEMO\piper\pt_BR-cadu-medium.onnx"
ARQUIVO_WAV = r"C:\BEMO\piper\fala.wav"

DISPOSITIVO = 5
NOVO_SAMPLE_RATE = 48000

while True:
    texto = input("Digite (ou 'sair'): ")

    if texto.lower() == "sair":
        break

    if not texto.strip():
        continue

    subprocess.run(
        [
            PIPER,
            "--model",
            MODEL,
            "--output_file",
            ARQUIVO_WAV,
        ],
        input=texto.encode("utf-8"),
        check=True
    )

    audio, sample_rate = sf.read(ARQUIVO_WAV)

    print("Sample rate original:", sample_rate)

    if sample_rate != NOVO_SAMPLE_RATE:
        quantidade = int(len(audio) * NOVO_SAMPLE_RATE / sample_rate)
        audio = signal.resample(audio, quantidade)

    sd.default.device = (None, DISPOSITIVO)

    sd.play(audio, NOVO_SAMPLE_RATE)
    sd.wait()

    if os.path.exists(ARQUIVO_WAV):
        os.remove(ARQUIVO_WAV)
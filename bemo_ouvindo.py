import sounddevice as sd
import numpy as np
import torch
import wave

from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper import WhisperModel


# =========================
# CONFIGURAÇÕES
# =========================

SAMPLE_RATE = 16000
CHANNELS = 1
DEVICE = 1
SILENCE_TIME = 1.5


# =========================
# MODELOS
# =========================

print("Carregando VAD...")

vad_model = load_silero_vad()

print("VAD carregado!")

print("Carregando Whisper...")

whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)
print("Whisper carregado!")

print("BEMO pronto!")

# =========================
# GRAVAR UM TRECHO
# =========================

def gravar_segundos(segundos):

    audio = sd.rec(
        int(segundos * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=DEVICE
    )

    sd.wait()

    return audio.flatten()


# =========================
# TRANSCRIBIR
# =========================

def transcrever(audio):

    # Salva temporariamente
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open("audio.wav", "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(SAMPLE_RATE)
        arquivo.writeframes(audio_int16.tobytes())

    segments, info = whisper_model.transcribe(
        "audio.wav",
        language="pt"
    )

    texto = ""

    for segment in segments:
        texto += segment.text

    return texto.strip()


# =========================
# ESPERAR POR "BIMO"
# =========================

def esperar_bimo():

    while True:

        # Escuta 1 segundo
        audio = gravar_segundos(1)

        # Converte para tensor
        tensor = torch.from_numpy(audio)

        # Verifica se existe fala
        timestamps = get_speech_timestamps(
            tensor,
            vad_model,
            sampling_rate=SAMPLE_RATE
        )

        # Se NÃO encontrou fala, ignora completamente
        if len(timestamps) == 0:
            continue

        # Só chega aqui se detectou voz
        print("Detectei voz, verificando...")

        texto = transcrever(audio)

        print("Ouvi:", texto)

        if "bimo" in texto.lower():
            return


# =========================
# GRAVAR ATÉ O SILÊNCIO
# =========================

def gravar_comando():

    print("\nBimo ativado!")
    print("Pode falar...")

    partes = []

    começou = False
    tempo_silencio = 0

    while True:

        # Grava 0,5 segundo
        audio = gravar_segundos(0.5)

        partes.append(audio)

        tensor = torch.from_numpy(audio)

        timestamps = get_speech_timestamps(
            tensor,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.4,
            min_speech_duration_ms=100
        )

        falando = len(timestamps) > 0

        if falando:

            começou = True
            tempo_silencio = 0

            print("Fala detectada...")

        elif começou:

            tempo_silencio += 0.5

            print(f"Silêncio: {tempo_silencio:.1f}s")

            if tempo_silencio >= SILENCE_TIME:
                break

    audio_completo = np.concatenate(partes)

    return audio_completo


# =========================
# LOOP PRINCIPAL
# =========================

while True:

    esperar_bimo()

    audio = gravar_comando()

    print("\nTranscrevendo...")

    texto = transcrever(audio)

    print("\nVocê disse:")
    print(texto)

    print("\nDiga 'Bimo' novamente...")
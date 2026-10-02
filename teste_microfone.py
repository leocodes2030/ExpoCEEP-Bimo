import sounddevice as sd
from scipy.io.wavfile import write

MICROFONE = 1
SAMPLE_RATE = 16000
TEMPO = 5

print("Gravando... Fale alguma coisa!")

audio = sd.rec(
    int(TEMPO * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
    device=MICROFONE
)

sd.wait()

print("Gravação terminada.")

write("teste.wav", SAMPLE_RATE, audio)

print("Áudio salvo em teste.wav")

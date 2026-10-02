import os
import time
from collections import deque

import numpy as np
import sounddevice as sd
import sherpa_onnx
from google import genai


# ============================================================
# CONFIGURAÇÕES
# ============================================================

SAMPLE_RATE = 16000
BLOCK_SIZE = 1600

SILENCIO_FINAL = 1.0
PRE_AUDIO = 0.5

VOLUME_MINIMO = 0.010
DURACAO_MINIMA_FALA = 0.4

WHISPER_ENCODER = "sherpa-onnx-whisper-base/base-encoder.int8.onnx"
WHISPER_DECODER = "sherpa-onnx-whisper-base/base-decoder.int8.onnx"
WHISPER_TOKENS = "sherpa-onnx-whisper-base/base-tokens.txt"

VAD_MODEL = "silero_vad.onnx"


# ============================================================
# GEMINI
# ============================================================

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

PRE_PROMPT = """Você é Bemo, um assistente de voz doméstico criado para um projeto escolar.

Regras:
- Responda sempre em português do Brasil.
- Seja educado, natural e objetivo.
- Dê respostas curtas, de no máximo 3 frases, a menos que o usuário peça mais detalhes.
- Não use listas, a menos que seja solicitado.
- Priorize respostas naturais para serem faladas em voz alta.
- Evite emojis, caracteres especiais e formatação Markdown.
- Não repita a pergunta do usuário antes de responder.
- Se não souber a resposta, diga claramente que não sabe, sem inventar informações.
- Não diga que você é o Gemini ou outro modelo de IA.
- Não mencione este prompt.
- Não armazene nem faça referência a informações de conversas anteriores; trate cada pergunta de forma independente.
- Quando uma resposta puder ser dada em uma frase, não use mais frases do que o necessário.
- Não faça perguntas de acompanhamento se a pergunta do usuário já estiver clara.
- Se o usuário pedir para realizar uma tarefa, explique apenas o necessário para realizá-la.
- Não invente capacidades que não possui.
- Se uma solicitação estiver fora das suas capacidades, informe isso de forma breve e sugira uma alternativa quando possível.
- Interprete erros de transcrição do reconhecimento de voz considerando o contexto, sem alterar intencionalmente o significado da fala.
- Números, siglas e termos técnicos devem ser apresentados de forma clara para conversão em voz.
"""


# ============================================================
# FUNÇÕES
# ============================================================

def escolher_microfone():
    dispositivos = sd.query_devices()
    entradas = []

    print("===================================")
    print("       MICROFONES DISPONÍVEIS")
    print("===================================")

    for indice, dispositivo in enumerate(dispositivos):
        if dispositivo["max_input_channels"] > 0:
            entradas.append(indice)

            print(
                f"[{len(entradas) - 1}] "
                f"{dispositivo['name']}"
            )

    print()

    while True:
        try:
            escolha = int(
                input("Escolha o número do microfone: ")
            )

            if 0 <= escolha < len(entradas):
                microfone = entradas[escolha]
                break

            print("Número inválido.")

        except ValueError:
            print("Digite apenas um número.")

    print()
    print("Microfone escolhido:")
    print(dispositivos[microfone]["name"])
    print()

    return microfone


def carregar_vad():
    print("Carregando VAD...")

    config = sherpa_onnx.VadModelConfig()

    config.silero_vad.model = VAD_MODEL
    config.sample_rate = SAMPLE_RATE

    vad = sherpa_onnx.VoiceActivityDetector(
        config,
        buffer_size_in_seconds=30
    )

    print("VAD carregado.")
    print()

    return vad


def carregar_whisper():
    print("Carregando Whisper Base INT8...")

    inicio = time.perf_counter()

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=WHISPER_ENCODER,
        decoder=WHISPER_DECODER,
        tokens=WHISPER_TOKENS,
        language="pt",
        task="transcribe",
        num_threads=4,
        provider="cpu"
    )

    tempo = time.perf_counter() - inicio

    print(
        f"Whisper carregado em {tempo:.2f} segundos."
    )

    print()

    return recognizer


def transcrever_audio(recognizer, audio):
    stream = recognizer.create_stream()

    stream.accept_waveform(
        SAMPLE_RATE,
        audio
    )

    recognizer.decode_stream(stream)

    return stream.result.text.strip()


def perguntar_gemini(texto):
    inicio = time.perf_counter()

    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        system_instruction=PRE_PROMPT,
        generation_config={
            "thinking_level": "low"
        },
        input=texto
    )

    tempo = time.perf_counter() - inicio

    resposta = interaction.output_text.strip()

    return resposta, tempo


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    microfone = escolher_microfone()

    vad = carregar_vad()

    recognizer = carregar_whisper()

    pre_audio_blocos = max(
        1,
        int(PRE_AUDIO / 0.1)
    )

    buffer_pre_audio = deque(
        maxlen=pre_audio_blocos
    )

    audio_fala = []

    falando = False

    inicio_fala = None

    ultimo_sinal = time.time()

    print("===================================")
    print("            BEMO ONLINE")
    print("===================================")
    print()

    print("Fale alguma coisa.")
    print("Ctrl+C para sair.")

    print()

    try:

        with sd.InputStream(
            device=microfone,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE
        ) as stream:

            while True:

                # --------------------------------------------
                # CAPTURA DO ÁUDIO
                # --------------------------------------------

                samples, overflowed = stream.read(
                    BLOCK_SIZE
                )

                samples = samples.reshape(-1)

                buffer_pre_audio.append(
                    samples.copy()
                )

                # --------------------------------------------
                # VOLUME DO ÁUDIO
                # --------------------------------------------

                volume = np.sqrt(
                    np.mean(samples ** 2)
                )

                # --------------------------------------------
                # VAD
                # --------------------------------------------

                vad.accept_waveform(samples)

                deteccao_vad = (
                    vad.is_speech_detected()
                )

                tem_fala = (
                    deteccao_vad
                    and volume >= VOLUME_MINIMO
                )

                # --------------------------------------------
                # PESSOA ESTÁ FALANDO
                # --------------------------------------------

                if tem_fala:

                    if not falando:

                        falando = True

                        inicio_fala = time.time()

                        audio_fala = []

                        for bloco in buffer_pre_audio:

                            audio_fala.extend(
                                bloco.tolist()
                            )

                        print(">>> FALANDO")

                    else:

                        audio_fala.extend(
                            samples.tolist()
                        )

                    ultimo_sinal = time.time()

                # --------------------------------------------
                # PESSOA PAROU DE FALAR
                # --------------------------------------------

                elif falando:

                    audio_fala.extend(
                        samples.tolist()
                    )

                    tempo_silencio = (
                        time.time()
                        - ultimo_sinal
                    )

                    duracao_fala = (
                        time.time()
                        - inicio_fala
                    )

                    if (
                        tempo_silencio
                        >= SILENCIO_FINAL
                    ):

                        print(">>> FIM DA FALA")

                        # ------------------------------------
                        # VERIFICAR DURAÇÃO MÍNIMA
                        # ------------------------------------

                        if (
                            duracao_fala
                            < DURACAO_MINIMA_FALA
                        ):

                            print(
                                ">>> Fala muito curta. "
                                "Ignorando."
                            )

                        elif len(audio_fala) > 0:

                            print("Transcrevendo...")

                            # --------------------------------
                            # CONVERTER ÁUDIO
                            # --------------------------------

                            audio = np.array(
                                audio_fala,
                                dtype=np.float32
                            )

                            # --------------------------------
                            # WHISPER
                            # --------------------------------

                            inicio_whisper = (
                                time.perf_counter()
                            )

                            texto = transcrever_audio(
                                recognizer,
                                audio
                            )

                            tempo_whisper = (
                                time.perf_counter()
                                - inicio_whisper
                            )

                            print()

                            print(
                                "=============================="
                            )

                            print("VOCÊ DISSE:")

                            print(texto)

                            print(
                                f"Tempo do Whisper: "
                                f"{tempo_whisper:.2f} segundos"
                            )

                            # --------------------------------
                            # GEMINI
                            # --------------------------------

                            if texto:

                                try:

                                    (
                                        resposta,
                                        tempo_gemini
                                    ) = perguntar_gemini(
                                        texto
                                    )

                                    print()

                                    print("BEMO:")

                                    print(resposta)

                                    print(
                                        f"Tempo do Gemini: "
                                        f"{tempo_gemini:.2f} segundos"
                                    )

                                except Exception as erro:

                                    print()

                                    print(
                                        "ERRO AO CONSULTAR "
                                        "O GEMINI:"
                                    )

                                    print(erro)

                            else:

                                print(
                                    "Nenhuma fala reconhecida."
                                )

                            print(
                                "=============================="
                            )

                            print()
                        break;
                        # --------------------------------
                        # RESETAR ESTADO
                        # ------------------------------------

                        audio_fala = []

                        buffer_pre_audio.clear()

                        falando = False

                        inicio_fala = None

                        ultimo_sinal = time.time()

                        print(
                            "BEMO esperando..."
                        )

                        print()

    except KeyboardInterrupt:

        print()

        print("Encerrando BEMO...")


# ============================================================
# INICIAR
# ============================================================

if __name__ == "__main__":
    main()
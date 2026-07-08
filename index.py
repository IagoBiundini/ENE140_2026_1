import json
import os
import shutil
import subprocess
import tempfile
import wave
import asyncio
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from ultralytics import YOLO

erro_import_speech_recognition = None

try:
    import speech_recognition as sr
except Exception as erro:
    sr = None
    erro_import_speech_recognition = erro

# contadores para o comando /status
quantidade_textos = 0
quantidade_imagens = 0
quantidade_audios = 0
quantidade_erros = 0

modelo_yolo = None
MAX_DURACAO_MIDIA_SEGUNDOS = 120


def pegar_token():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Coloque o token na variavel TELEGRAM_BOT_TOKEN")
    return token


def carregar_yolo():
    global modelo_yolo
    if modelo_yolo is None:
        modelo_yolo = YOLO("yolov8n.pt")
    return modelo_yolo


def classificar_frase(texto):
    texto = texto.lower()

    if any(palavra in texto for palavra in ["oi", "ola", "olá", "bom dia", "boa tarde", "boa noite"]):
        return "saudacao"

    if any(palavra in texto for palavra in ["erro", "bug", "problema", "falha"]):
        return "relato de problema"

    if any(palavra in texto for palavra in ["ajuda", "duvida", "dúvida", "como", "explica"]):
        return "pedido de ajuda"

    return "mensagem comum"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensagem = (
        "Oi! Eu sou um bot simples feito em Python.\n\n"
        "Posso responder textos, analisar imagens com YOLO e tentar transcrever audios.\n"
        "Use /help para ver os comandos."
    )
    await update.message.reply_text(mensagem)


async def help_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensagem = (
        "Comandos:\n"
        "/start - inicia o bot\n"
        "/help - mostra esta ajuda\n"
        "/status - mostra quantas mensagens foram processadas\n\n"
        "Tambem da para enviar texto, foto ou audio de voz."
    )
    await update.message.reply_text(mensagem)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensagem = (
        "Status do bot:\n"
        f"Textos: {quantidade_textos}\n"
        f"Imagens: {quantidade_imagens}\n"
        f"Audios: {quantidade_audios}\n"
        f"Erros: {quantidade_erros}"
    )
    await update.message.reply_text(mensagem)


async def responder_texto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global quantidade_textos
    quantidade_textos += 1

    texto = update.message.text or ""
    categoria = classificar_frase(texto)

    if categoria == "saudacao":
        resposta = "Opa! Tudo certo? Envie uma foto ou audio para eu analisar."
    elif categoria == "pedido de ajuda":
        resposta = "Posso ajudar com texto, imagem e audio. Tente mandar uma foto ou uma mensagem de voz."
    elif categoria == "relato de problema":
        resposta = "Entendi que voce falou de um problema. Me mande mais detalhes para eu tentar ajudar."
    else:
        resposta = f"Recebi sua mensagem. Categoria: {categoria}."

    await update.message.reply_text(resposta)


def montar_resposta_yolo(caminho_imagem):
    modelo = carregar_yolo()
    resultados = modelo.predict(source=str(caminho_imagem), verbose=False)

    if not resultados or resultados[0].boxes is None or len(resultados[0].boxes) == 0:
        return "Não encontrei objetos na imagem."

    resultado = resultados[0]
    objetos = []

    for caixa in resultado.boxes[:10]:
        classe_id = int(caixa.cls[0])
        confianca = float(caixa.conf[0])
        nome = resultado.names.get(classe_id, "objeto")
        objetos.append(f"- {nome}: {confianca:.0%}")

    return "Objetos encontrados:\n" + "\n".join(objetos)


async def analisar_imagem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global quantidade_imagens, quantidade_erros
    quantidade_imagens += 1

    try:
        foto = update.message.photo[-1]
        arquivo = await context.bot.get_file(foto.file_id)

        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "foto.jpg"
            await arquivo.download_to_drive(caminho)
            resposta = montar_resposta_yolo(caminho)

        await update.message.reply_text(resposta)
    except Exception as erro:
        quantidade_erros += 1
        await update.message.reply_text(f"Erro ao analisar a imagem: {erro}")


def converter_ogg_para_wav(entrada, saida):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg não foi encontrado no computador")

    comando = [
        "ffmpeg",
        "-y",
        "-i",
        str(entrada),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(saida),
    ]

    processo = subprocess.run(comando, capture_output=True, text=True)
    if processo.returncode != 0:
        raise RuntimeError("não consegui converter o audio")


def transcrever_com_google(caminho_wav):
    if sr is None:
        if erro_import_speech_recognition:
            return None, f"SpeechRecognition indisponivel: {erro_import_speech_recognition}"
        return None, "SpeechRecognition não esta instalado."

    reconhecedor = sr.Recognizer()
    with sr.AudioFile(str(caminho_wav)) as fonte:
        audio = reconhecedor.record(fonte)

    try:
        texto = reconhecedor.recognize_google(audio, language="pt-BR")
        return texto, None
    except sr.UnknownValueError:
        return None, "Google Speech Recognition não conseguiu entender o audio."
    except sr.RequestError as erro:
        return None, f"Falha ao usar Google Speech Recognition: {erro}"


def transcrever_com_vosk(caminho_wav):
    try:
        from vosk import KaldiRecognizer, Model
    except ImportError:
        return None, "Vosk não esta instalado."

    modelo_path = os.getenv("VOSK_MODEL_PATH", "models/vosk-model-small-pt-0.3")
    if not Path(modelo_path).exists():
        return None, f"Modelo Vosk não encontrado em: {modelo_path}"

    modelo = Model(modelo_path)
    textos = []

    with wave.open(str(caminho_wav), "rb") as arquivo_wav:
        reconhecedor = KaldiRecognizer(modelo, arquivo_wav.getframerate())

        while True:
            dados = arquivo_wav.readframes(4000)
            if len(dados) == 0:
                break

            if reconhecedor.AcceptWaveform(dados):
                resultado = json.loads(reconhecedor.Result())
                if resultado.get("text"):
                    textos.append(resultado["text"])

        resultado_final = json.loads(reconhecedor.FinalResult())
        if resultado_final.get("text"):
            textos.append(resultado_final["text"])

    texto = " ".join(textos).strip()
    if texto:
        return texto, None

    return None, "Vosk não conseguiu entender o audio."


def transcrever_audio(caminho_wav):
    motivos = []

    texto, erro_google = transcrever_com_google(caminho_wav)
    if texto:
        return texto, "Google Speech Recognition"
    if erro_google:
        motivos.append(erro_google)

    texto, erro_vosk = transcrever_com_vosk(caminho_wav)
    if texto:
        return texto, "Vosk"
    if erro_vosk:
        motivos.append(erro_vosk)

    diagnostico = " | ".join(motivos) if motivos else "Sem diagnostico adicional."
    return "Não consegui transcrever.", diagnostico


async def analisar_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global quantidade_audios, quantidade_erros

    try:
        voice = update.message.voice
        duracao = voice.duration or 0

        if duracao > MAX_DURACAO_MIDIA_SEGUNDOS:
            await update.message.reply_text(
                f"Audio muito longo. Limite: {MAX_DURACAO_MIDIA_SEGUNDOS} segundos."
            )
            return

        quantidade_audios += 1
        arquivo = await context.bot.get_file(voice.file_id)

        with tempfile.TemporaryDirectory() as pasta:
            caminho_ogg = Path(pasta) / "audio.ogg"
            caminho_wav = Path(pasta) / "audio.wav"

            await arquivo.download_to_drive(caminho_ogg)
            converter_ogg_para_wav(caminho_ogg, caminho_wav)
            texto, origem_ou_erro = transcrever_audio(caminho_wav)
            categoria = classificar_frase(texto) if texto != "Não consegui transcrever." else "não classificado"

        resposta = f"Transcricao: {texto}\nCategoria: {categoria}\nDetalhes: {origem_ou_erro}"
        await update.message.reply_text(resposta)
    except Exception as erro:
        quantidade_erros += 1
        await update.message.reply_text(f"Erro ao analisar o audio: {erro}")


async def analisar_audio_arquivo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    audio = update.message.audio
    duracao = (audio.duration or 0) if audio else 0

    if duracao > MAX_DURACAO_MIDIA_SEGUNDOS:
        await update.message.reply_text(
            f"Audio muito longo. Limite: {MAX_DURACAO_MIDIA_SEGUNDOS} segundos."
        )
        return

    await update.message.reply_text(
        "Audio recebido. Para transcrever, envie como mensagem de voz do Telegram."
    )


async def analisar_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    video = update.message.video
    duracao = (video.duration or 0) if video else 0

    if duracao > MAX_DURACAO_MIDIA_SEGUNDOS:
        await update.message.reply_text(
            f"Video muito longo. Limite: {MAX_DURACAO_MIDIA_SEGUNDOS} segundos."
        )
        return

    await update.message.reply_text(
        "Video recebido. No momento eu não faço analise de video."
    )


def main():
    token = pegar_token()

    # Python 3.14 não cria loop automaticamente no MainThread.
    asyncio.set_event_loop(asyncio.new_event_loop())

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_bot))
    app.add_handler(CommandHandler("status", status))

    app.add_handler(MessageHandler(filters.PHOTO, analisar_imagem))
    app.add_handler(MessageHandler(filters.VOICE, analisar_audio))
    app.add_handler(MessageHandler(filters.AUDIO, analisar_audio_arquivo))
    app.add_handler(MessageHandler(filters.VIDEO, analisar_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder_texto))

    print("Bot rodando. Aperte Ctrl+C para parar.")
    app.run_polling()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""botdotelegram.ipynb

Grupo 3: Isadora, Kauã e Yuri
"""

from datetime import datetime
from telegram.ext import Application, MessageHandler, filters
from telegram import Update

import os
import uuid
import cv2
import whisper
from ultralytics import YOLO
from deep_translator import GoogleTranslator

def log(msg: str):
    """Registra uma nova entrada no arquivo de log."""
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {msg}\n")


class BotTelegram:
    """
    Classe responsável pela configuração e gerenciamento do bot do Telegram.

    Inicializa a aplicação do bot, protege o token de acesso por meio de
    encapsulamento e encaminha as mídias recebidas para o processamento adequado.
    """
    def __init__(self, token: str):
        """Inicializa o bot e configura o interceptador de mensagens."""
        self.__token = token
        self.application = Application.builder().token(self.__token).build()
        self.application.add_handler(
            MessageHandler(filters.ALL & ~filters.COMMAND, self.retorno))
    def iniciando_bot(self):
        """Ativa o recebimento contínuo de mensagens."""
        log("Bot ativado")
        print("Bot em funcionamento")
        self.application.run_polling()
    async def retorno(self, update, context):
        """Encaminha o arquivo recebido para o processamento de acordo com seu tipo."""
        if update.message.photo:
            log("Arquivo de imagem recebido")
            await self.processar_imagem(update, context)
        elif update.message.voice or update.message.audio:
            log("Arquivo de áudio recebido")
            await self.processar_audio(update, context)
        elif update.message.text:
            await update.message.reply_text("Seu texto foi recebido, no entanto só retorno informações de arquivos de áudio e imagem")
        else:
            await update.message.reply_text("Não fui programado para processar esse tipo de arquivo, envie uma imagem ou um áudio")
    async def processar_imagem(self, update, context):
        """Define a interface para o processamento de imagens."""
        await update.message.reply_text("Processamento de imagem não implementado nesta classe.")
    async def processar_audio(self, update, context):
        """Define a interface para o processamento de áudio."""
        await update.message.reply_text("Processamento de áudio não implementado nesta classe.")
        
class BotImagem(BotTelegram):
    def __init__(self, token, caminho_modelo="yolov8n.pt", confianca_minima=0.4, **kwargs):
        """Inicializa o bot carregando o modelo YOLO e criando a pasta de arquivos temporários."""
        super().__init__(token, **kwargs)
        self.__modelo = YOLO(caminho_modelo)
        self.__confianca_minima = confianca_minima
        self._pasta_temp = "temp_imagens"
        os.makedirs(self._pasta_temp, exist_ok=True)

    # encapsulamento sem decorator: getter e setter tradicionais
    def get_confianca_minima(self):
        """Retorna o limiar mínimo de confiança usado nas detecções."""
        return self.__confianca_minima

    def set_confianca_minima(self, valor):
        """Define o limiar mínimo de confiança (0 a 1), validando o intervalo."""
        if not 0.0 <= valor <= 1.0:
            raise ValueError("A confiança mínima deve estar entre 0 e 1.")
        self.__confianca_minima = valor

    # sobrescreve o gancho da classe base (polimorfismo)
    async def processar_imagem(self, update, context):
        """Baixa a foto recebida, roda a detecção YOLO, envia a imagem anotada com a legenda e remove os arquivos temporários."""
        foto = update.message.photo[-1]
        arquivo = await foto.get_file()
        caminho_entrada = os.path.join(self._pasta_temp, f"{uuid.uuid4().hex}.jpg")
        await arquivo.download_to_drive(caminho_entrada)
        resultados = self.__modelo.predict(caminho_entrada, conf=self.__confianca_minima, verbose=False)
        deteccoes = self.extrair_deteccoes(resultados)
        caminho_saida = self.gerar_imagem_anotada(resultados, caminho_entrada)
        if deteccoes:
            legenda = self.formatar_resultado(deteccoes)
        else:
            legenda = "Nenhum objeto identificado com confiança suficiente."
        with open(caminho_saida, "rb") as img:
            await update.message.reply_photo(photo=img, caption=legenda)
        log(f"Imagem processada: {legenda}")
        self.limpar_arquivos(caminho_entrada, caminho_saida)

    def extrair_deteccoes(self, resultados):
        """Converte os resultados do YOLO em uma lista de tuplas (nome_classe, confiança)."""
        deteccoes = []
        for r in resultados:
            for box in r.boxes:
                nome_classe = r.names[int(box.cls[0])]
                confianca = float(box.conf[0])
                deteccoes.append((nome_classe, confianca))
        return deteccoes

    def formatar_resultado(self, deteccoes):
        """Agrupa as detecções por classe e monta um texto com contagem e confiança média de cada objeto."""
        contagem = {}
        for nome, conf in deteccoes:
            contagem.setdefault(nome, []).append(conf)
        linhas = ["Objetos identificados:"]
        for nome, confs in sorted(contagem.items(), key=lambda kv: -max(kv[1])):
            media_conf = sum(confs) / len(confs)
            linhas.append(f"- {nome} (x{len(confs)}) - confianca media {media_conf:.0%}")
        return "\n".join(linhas)

    def gerar_imagem_anotada(self, resultados, caminho_entrada):
        """Desenha as caixas de detecção sobre a imagem original e salva o resultado em disco."""
        imagem_anotada = resultados[0].plot()
        caminho_saida = caminho_entrada.replace(".jpg", "_anotada.jpg")
        cv2.imwrite(caminho_saida, imagem_anotada)
        return caminho_saida

    def limpar_arquivos(self, *caminhos):
        """Remove os arquivos temporários informados, ignorando erros caso já não existam."""
        for c in caminhos:
            try:
                os.remove(c)
            except OSError:
                pass

class BotAudio(BotTelegram):
    """
    Classe responsável pela leitura de áudios recebidos pelo bot.

    Identifica se a mensagem recebida é áudio ou mensagem de voz, baixa o arquivo
    enviado pelo Telegram, transcreve usando Whisper, detecta o idioma e traduz
    para português quando o áudio estiver em inglês ou japonês.
    """
    # Recebe o token da classe mãe e carrega o modelo Whisper
    def __init__(self, token: str):
        super().__init__(token)
        self.__modelo4 = whisper.load_model('base')

    # Verifica se a mensagem recebida é áudio comum ou uma mensagem de voz
    def arquivo_audio4(self, update):
        if update.message.audio:
            return update.message.audio.file_id

        if update.message.voice:
            return update.message.voice.file_id

        return None

    # Baixa o áudio enviado pelo usuário e o salva localmente
    async def baixar_audio4(self, update, context, arquivo4='audio_grupo4.ogg'):
        file_id4 = self.arquivo_audio4(update)

        if file_id4 is None:
            return None

        arquivo_telegram4 = await context.bot.get_file(file_id4)
        await arquivo_telegram4.download_to_drive(arquivo4)

        return arquivo4

    # Usa o Whisper para transformar o áudio em texto e retorna o idioma detectado
    def transcrever_audio4(self, caminho_audio4):
        transcricao4 = self.__modelo4.transcribe(caminho_audio4)
        texto4 = transcricao4['text']
        idioma4 = transcricao4['language']

        return texto4, idioma4

    # Traduz o texto para português caso o idioma detectado seja inglês ou japonês
    def traduzir_audio4(self, texto4, idioma4):
        if idioma4 == 'pt':
            return None

        if idioma4 == 'en' or idioma4 == 'ja':
            traducao4 = GoogleTranslator(source=idioma4, target='pt').translate(texto4)
            return traducao4

        return None

    # Sobrescreve o método processar_audio da classe mãe BotTelegram
    async def processar_audio(self, update, context):
        caminho_audio4 = None

        try:
            await update.message.reply_text('O áudio foi recebido com sucesso, peço que aguarde alguns segundos para que eu possa processá-lo...')

            caminho_audio4 = await self.baixar_audio4(update, context)

            if caminho_audio4 is None:
                await update.message.reply_text('Não identifiquei nenhum arquivo de áudio ou de voz, tente novamente.')
                return

            texto4, idioma4 = self.transcrever_audio4(caminho_audio4)

            if texto4.strip() == '':
                await update.message.reply_text('Não consegui identificar fala no áudio enviado. Verifique se o áudio contém voz e tente novamente.')
                return

            traducao4 = self.traduzir_audio4(texto4, idioma4)

            resposta4 = (
                f'O áudio foi transcrito e detectei o idioma que foi utilizado, veja abaixo:'
                f'\n\nIdioma detectado: {idioma4}'
                f'\nTranscrição: {texto4}'
            )

            if traducao4 is not None:
                resposta4 += f'\n\nTradução para o português: {traducao4}'

            await update.message.reply_text(resposta4)

        # Caso tenha algum erro no download, transcrição ou tradução, avisa ao usuário que deu problema
        except Exception as erro4:
            await update.message.reply_text(
                f'Infelizmente ocorreu um erro ao tentar processar o áudio. '
                f'Verifique se o arquivo está correto, por favor, e tente novamente.\nErro: {erro4}'
            )

        # Remove o arquivo de áudio temporário depois do processo
        finally:
            if caminho_audio4 is not None and os.path.exists(caminho_audio4):
                os.remove(caminho_audio4)

class BotProjeto(BotImagem, BotAudio):
    pass

token = "8872031925:AAFNHyCbJO5huJ6txhPllmyp9Glebb5m-UI"
bot = BotProjeto(token)
bot.iniciando_bot()
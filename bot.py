import os
from collections import Counter

from telegram.ext import Application, MessageHandler, filters, ContextTypes
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from transformers import pipeline
from ultralytics import YOLO
from telegram import Update
import cv2

class BotTelegram():
    """
    Classe base para bots Telegram.

    Gerencia a inicialização do bot, a comunicação com a API do Telegram, o tratamento de mensagens e a classificação de texto.
    """

    def __init__(self, token: str):
        """
        Args:
            token (str): Token de acesso ao bot.
        """

        self.__token = token # Atributo privado para armazenar o token
        self.categorias = ["perguntas factuais simples", "pedido de ajuda ou suporte", "reclamação ou insatisfação", "conversa casual", "referências e opiniões", "recomendações"]
        self.classificador = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        self.app = Application.builder().token(token).build()
        self.app.add_handler(MessageHandler(filters.ALL, self.tratar_mensagem)) # Adiciona o handler para tratar todas as mensagens recebidas

    def run(self):
        """
        Inicia o bot.
        """
        self.app.run_polling() # Inicia o tratamento de mensagens

    def get_token(self) -> str:
        """
        Método getter para acessar o token.
        """

        return self.__token

    async def comando_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Responde ao comando '/start' com uma mensagem de boas-vindas."""
        texto = (
            "Bem-vindo! Eu processo texto (classificação), imagens (detecção de objetos com YOLO) "
            "e audios (transcrição de voz com classificação de conteúdo)."
            "Envie qualquer um dos tres tipos de mensagem para testar."
        )
        await self.responder(update, texto)

    async def tratar_mensagem(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Utiliza polimorfismo para reconhecer e tratar diferentes tipos de mensagens de forma eficiente.
        """

        try:
            os.makedirs(".temp", exist_ok=True) # Cria a pasta de .temp para as imagens e áudios se não existir
            mensagem = update.message
            await self.responder(update, "Processando...") # Informa ao usuário que a mensagem está sendo processada

            if mensagem.photo:
                imagem = mensagem.photo[-1] # Seleciona a melhor resolução da imagem
                arquivo = await imagem.get_file()
                dado = f".temp/imagem{os.path.splitext(arquivo.file_path)[1]}" # Obtem a extensão do arquivo
                await arquivo.download_to_drive(dado) # Baixa a imagem na pasta .temp
                bot = BotImagem(self.get_token())

            elif mensagem.voice or mensagem.audio:
                arquivo = await mensagem.voice.get_file() if mensagem.voice else await mensagem.audio.get_file()
                dado = f".temp/audio{os.path.splitext(arquivo.file_path)[1]}" # Obtem a extensão do arquivo
                await arquivo.download_to_drive(dado) # Baixa o áudio na pasta .temp
                bot = BotAudio(self.get_token())

            elif mensagem.text:
                if mensagem.text.strip() == "/start":
                    await self.comando_start(update, context)
                    return
                dado = mensagem.text
                bot = BotTexto(self.get_token())
                
            else: # Fallback para tipos de mensagens não suportados
                await self.responder(update, "Tipo de mensagem não suportado.")
                return
            
            await bot.processar_entrada(update, dado)

        except Exception as erro:
            await self.responder(update, f"Erro: {erro}")

        finally:
            # Limpa a pasta .temp após o processamento
            for arquivo in os.listdir(".temp"):
                caminho_arquivo = os.path.join(".temp", arquivo)
                if os.path.isfile(caminho_arquivo):
                    os.remove(caminho_arquivo) # Remove os arquivos da pasta .temp

    def classificar_texto(self, texto: str) -> str:
        """
        Classificação zero-shot para rotulação.

        Args:
            texto (str): Texto a ser classificado.
        """

        return self.classificador(texto, candidate_labels=self.categorias)["labels"][0] # Retorna a categoria com maior probabilidade

    async def responder(self, update: Update, resposta: str):
        """
        Envia uma resposta ao usuário.

        Args:
            update (Update): Objeto que contém informações sobre a atualização recebida.
            resposta (str): Mensagem de resposta a ser enviada.
        """

        await update.message.reply_text(resposta)

    def processar_entrada(self, update: Update, resposta: str):
        """
        Método a ser sobrescrito pelas classes filhas.
        """

        pass

class BotImagem(BotTelegram):
    """
    Bot para o processamento de imagens.

    Usa YOLO para detectar os objetos em uma imagem e responde o usuário
    com a lista de objetos encontrados, caso existam.
    """

    def __init__(self, token: str):
        super().__init__(token)

        self.modelo = YOLO("yolov8n.pt")

    async def processar_entrada(self,  update: Update, caminho: str):
        """
        Executa a detecção de objetos.

        Args:
            caminho (str): Caminho da imagem a ser processada.
            update (Update): Objeto que contém informações sobre a atualização recebida.
        """

        try:
            imagem = cv2.imread(caminho)

            if imagem is None:
                await self.responder(update, "Erro ao abrir imagem.")

            resultados = self.modelo(caminho)

            objetos_detectados = []
            for resultado in resultados:
                for caixa in resultado.boxes:
                    classe = int(caixa.cls[0])
                    nome = GoogleTranslator(source="en", target="pt").translate(self.modelo.names[classe]) # Traduz o nome do objeto para português
                    objetos_detectados.append(nome)

            if len(objetos_detectados) == 0: # Caso nenhum objeto seja detectado, envia uma mensagem informando ao usuário
                await self.responder(update, "Nenhum objeto detectado.")
            else:
                # Responde o usuário com a lista de objetos detectados e suas quantidades ordenadas por frequência
                contador = Counter(objetos_detectados)
                lista_objetos = [f"{quantidade}x {objeto}{'s' if quantidade > 1 else ''}" for objeto, quantidade in contador.most_common()]
                await self.responder(update, ("Objetos encontrados:\n"+"\n".join(lista_objetos)))

        except Exception as erro:
            await self.responder(update, f"Erro ao processar imagem: {erro}")

class BotAudio(BotTelegram):
    """
    Bot para o processamento de áudio.

    Responde o usuário com a transcrição de áudio em texto e a classificação do conteúdo transcrito.
    """

    def __init__(self, token: str):
        super().__init__(token)

        self.recognizer = WhisperModel("small", device="cpu")  # Inicializa o modelo Whisper para transcrição de áudio

    async def processar_entrada(self, update: Update, caminho: str):
        """
        Converte áudio em texto.

        Args:
            update (Update): Objeto que contém informações sobre a atualização recebida.
            caminho (str): Caminho do arquivo de áudio a ser processado.
        """

        try:
            segments, info = self.recognizer.transcribe(caminho,language="pt")
            texto = "".join(segment.text for segment in segments)

            categoria = self.classificar_texto(texto)

            # Responde o usuário com a transcrição do áudio e a categoria identificada
            await self.responder(update, f"Texto reconhecido:\n{texto.strip()}\nCategoria: {categoria}")

        except Exception as erro:
            await self.responder(update, f"Erro no reconhecimento: {erro}")

class BotTexto(BotTelegram):
    """
    Bot para o processamento de mensagens de texto.

    Responde o usuário com a classificação do conteúdo recebido.
    """

    def __init__(self, token: str):
        super().__init__(token)

    async def processar_entrada(self, update: Update, texto: str):
        """
        Processa mensagens textuais.

        Args:
            update (Update): Objeto que contém informações sobre a atualização recebida.
            texto (str): Texto a ser processado.
        """

        categoria = self.classificar_texto(texto)
        # Responde o usuário com a categoria identificada
        await self.responder(update, f"Categoria identificada: {categoria}")

if __name__ == "__main__":
    token = "8626741664:AAFom6agdgxkXQIElTLMKuLKc9-aKGmD_og"
    bot = BotTelegram(token=token)
    bot.run()

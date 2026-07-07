#Obs: É necessário instalar as bibliotecas do Telegram, reconhecimetno de voz e YOLO
#(python-telegram-bot ultralytics opencv-python SpeechRecognition)

#Bibliotecas do Telegram

from turtle import update
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from ultralytics import YOLO
import speech_recognition as sr

#Obs: Após testar o bot, percebeu-se que áudios gravados pelo usuário eram entregues em um formato que o trasncritor não conseguia ler, portanto, foi necessário instalar
#uma biblioteca adicional para converter o áudio para um formato que o trasncritor consiga ler (pydub)
import soundfile as sf

#Biblioteca usada para traduzir a resposta do YOLO para o português
from deep_translator import GoogleTranslator

#Biblioteca usada para contar a quantidade de ocorrências de cada objeto detectado na imagem
from collections import Counter

#Biblioteca para gerar a rosta em áudio do reconhecimento da imagem
from gtts import gTTS

class BotTelegram:
    #token é um dado sensível, portanto, deixamos ele como privado
    def __init__(self, token):
        self.__token = token
        print(f"Bot inicializado com token: {self.__token}")
        #Criação da aplicação do bot com o token fornecido
        self.app = Application.builder().token(self.__token).build()
        self._configurar_handlers()

    #Método necessário para acessar o token privado de forma segura
    def get_token(self):
        return self.__token
    
    #Esse método está aqui para as classes filhas já herdarem, mas serão implementadas de forma diferente em cada uma delas
    def processar_mensagem(self, update: Update, context: ContextTypes.DEFAULT_TYPE):        
        raise NotImplementedError("O método processar_mensagem deve ser implementado nas classes filhas.")
    
    #Isso faz o bot ficar rodando e checando se há novas mensagens no Telegram
    def executar(self):
        print( "Bot iniciado! Pressione Ctrl+C para desligar")
        self.app.run_polling()
   
    #Função para tratar o comando /start do bot
    #Obs: A função precisa ser assíncrona pois o bot não pode travar enquanto espera a resposta de um único usuário
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        #Aqui dentro depois vamos colocar a resposta para o usuário
        print("Usuário deu /start!")
        #await é necesário para suspender temporariamente a execução da função e deixá-la em segundo plano liberando o fluxo de controle
        #para processar novas tarefas enquanto aguarda a conclusão da operação assíncrona
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Olá! Bem-vindo ao nosso bot!")

    def _configurar_handlers(self):
        #Diz ao bot: "Quando o usuário mandar o comando /start, rode a função self.start"
        self.app.add_handler(CommandHandler("start", self.start))
        
        #Quando o bot receber uma foto, rodar a função self.tratar_foto
        self.app.add_handler(MessageHandler(filters.PHOTO, self.tratar_foto))
        
        #Quando o bot receber um áudio, chamar a função self.tratar_audio
        self.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.tratar_audio))

        #Caso a mensagem não seja um comando, nem foto, nem áudio, chamama-se a função self.tratar_tipo_invalido:
        self.app.add_handler(MessageHandler(~filters.COMMAND & ~filters.PHOTO & ~filters.VOICE & ~filters.AUDIO, self.tratar_tipo_invalido))

    #Quando chega uma foto, criamos o objeto da classe filha especialista em imagem
    async def tratar_foto(self, update: Update, context: ContextTypes.DEFAULT_TYPE):        
        processador = BotImagem(self.get_token())
        #Chamamos o método dela (que depois vai ter o YOLO)
        await processador.processar_mensagem(update, context)

    #Quando chega um áudio, criamos o objeto da classe filha especialista em áudio
    async def tratar_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):        
        processador = BotAudio(self.get_token())
        # Chamamos o método dela (que depois vai ter o Speech-to-Text)
        await processador.processar_mensagem(update, context)

    #Proteção para avisar que o formato enviado não é aceito
    async def tratar_tipo_invalido(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        print("⚠️ Usuário enviou um formato de arquivo inválido.")
        await update.message.reply_text("❌ Opa! Eu só consigo processar mensagens de **imagem (foto)** ou de **áudio/voz**. Por favor, envie um desses formatos!")    

#Criação das classes filhas
class BotImagem(BotTelegram):
    async def processar_mensagem(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        #Polimorfismo para processar mensagens de imagem
        print(" IA Processando mensagem de imagem...")

        #Seleciona a versão com maior resolução da foto enviada pelo usuário
        foto = update.message.photo[-1]
        #Usa o ID da foto para pedir ao servidor do Telegram o link de acesso ao arquivo real.
        arquivo_telegram = await context.bot.get_file(foto.file_id)
        #Definir o nome do arquivo e baixar a imagem
        caminho_foto = "imagem_recebida.jpg"
        await arquivo_telegram.download_to_drive(caminho_foto)
        #Carregar o modelo do YOLO
        modelo = YOLO("yolov8n.pt")
        #Fazer a detecção de objetos na imagem
        # Forçamos o YOLO a salvar direto na pasta 'runs/predict' sem duplicar caminhos
        resultados = modelo(caminho_foto, save=True, project="runs", name="predict", exist_ok=True)
        #Ler o que o YOLO encontrou e responder para o usuário
        await update.message.reply_text("Imagem analisada com sucesso!")

        """"O YOLO retorna uma lista de resultados, onde cada resultado contém informações sobre os objetos detectados na imagem.
        é necessário criar um loop para percorrer cada resultado e extrair somente as informações relevantes para o usuário."""

        #Criar uma lista para armazenar as informações dos objetos detectados
        objetos_detectados = []
        for r in resultados:
            for c in r.boxes.cls:
                #Para cada resultado detectado, converte-se o índice da classe pelo nome do objeto correspondente
                nome_ingles = modelo.names[int(c)]
                #Traduze-se o nome do objeto de inglês para português
                nome_traduzido = GoogleTranslator(source='en', target='pt').translate(nome_ingles)
                objetos_detectados.append(nome_traduzido)
        #Cria-se uma mensagem de resposta para o usuário
        if objetos_detectados:
            #O Counter conta quantas vezes cada objeto aparece na lista
            contagem = Counter(objetos_detectados)

            # Monta uma lista formatada, ex: ["2x carro", "1x pessoa"]
            itens_formatados = [f"{quantidade}x {item}" for item, quantidade in contagem.items()]

            # Junta tudo separado por vírgula
            texto_resposta = f"Objetos detectados: {', '.join(itens_formatados)}"
        else:
            texto_resposta = "Nenhum objeto conhecido foi detectado."
        #Enviar a mensagem de volta pelo telegram
        await update.message.reply_text(texto_resposta)
        tts = gTTS(text=texto_resposta, lang='pt', slow=False)
        tts.save("resposta_bot.mp3")

        # Envia o arquivo de áudio de volta para o usuário no Telegram
        with open("resposta_bot.mp3", 'rb') as audio_resposta:
            await update.message.reply_voice(voice=audio_resposta)

        # O YOLO sempre salva o resultado na pasta 'runs/detect/predict/' com o mesmo nome do arquivo original
        caminho_resultado = "runs/detect/runs/predict/imagem_recebida.jpg"

        try:
            # Abre o arquivo de imagem gerado pelo YOLO em modo de leitura binária ('rb')
            with open(caminho_resultado, 'rb') as foto_anexo:
                await update.message.reply_photo(photo=foto_anexo, caption="Segue os resultados da análise de imagem do YOLO!")
        except Exception as e:
            print(f"Erro ao tentar enviar a imagem do YOLO: {e}")


#Criação da outras classe filha
class BotAudio(BotTelegram):
    async def processar_mensagem(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        #Polimorfismo para processar mensagens de áudio
        print(" IA Processando mensagem de áudio...")
    
        """O telegram faz uma diferenciação entre as mensagens de áudio (gravadas pelo usuário) e os áudios (enviados como arquivo de música).
        Por isso, faze-se necessário tratamentos diferentes para cada formato, ou seja, primeiro verifica-se qual o tipo de mensagem recebida."""

        #Áudio gravado pelo usuário
        if update.message.voice: 
            audio_data = update.message.voice
        #Se foi um áudio enviado como arquivo
        else:
            audio_data = update.message.audio
        caminho_audio = "audio_original"
        #Tal como feito para imagens, usa-se o ID do áudio para pedir ao servidor do Telegram o link de acesso ao arquivo real.
        arquivo_telegram = await context.bot.get_file(audio_data.file_id)
        #Baixar o áudio para o disco local
        await arquivo_telegram.download_to_drive(caminho_audio)
        #Caso ocorra tudo certo nessa etapa, avisa o usuário que o áuido foi recebido e iniciará a transcrição
        await update.message.reply_text("Áudio recebido! Iniciando transcrição de voz...")

        #Caso o áudio seja do tipo OGG, é necessário convertê-lo para WAV, pois o reconhecedor de voz não consegue ler o formato OGG.
        caminho_wav = "audio_pronto.wav"

        try:
            #O soundfile abre o arquivo original lendo os bytes internos dele
            dados, taxa_amostragem = sf.read(caminho_audio)
            sf.write(caminho_wav, dados, taxa_amostragem)
        except Exception as e:
            print(f"Erro na conversão nativa: {e}")
            # Se o soundfile falhar ele apenas repassa o arquivo para tentar a sorte no reconhecedor 
            caminho_wav = caminho_audio

        #Como o serviçõ de transcrição depende de uma conexão em núvem, optou-se pelo google que é gratuito e confiável.
        #Inicializa-se o reconhecedor de voz da biblioteca
        reconhecedor = sr.Recognizer()
        
        #Abre-se o arquivo de áudio baixado
        with sr.AudioFile(caminho_wav) as fonte:
            #O reconhecedor lê os dados do arquivo de som
            dados_audio = reconhecedor.record(fonte)
        
        #Como depende de uma conexão com a internet, é necessário tratar possíveis erros que podem ocorrer durante a transcrição do áudio.
        #para evitar o travamento do bot.
        try:
            #Os dados são enviados para a IA do Google transcrever em Português do Brasil
            texto_transcrito = reconhecedor.recognize_google(dados_audio, language="pt-BR")

            palavras = texto_transcrito.split()
            total_palavras = len(palavras)
            
            #Conta as palavras mais faladas (ignorando conectivos como 'e', 'de', 'o')
            palavras_filtradas = [p for p in palavras if len(p) > 3]
            from collections import Counter
            mais_comuns = Counter(palavras_filtradas).most_common(2)
            
            palavras_chave = ", ".join([f"'{item[0]}'" for item in mais_comuns]) if mais_comuns else "Nenhuma"

            resposta_final = (
                f"🗣️ Transcrição: \"{texto_transcrito}\"\n\n"
                f"📊 **Métricas da sua fala:**\n"
                f"• Total de palavras: {total_palavras}\n"
                f"• Palavras-chave mais repetidas: {palavras_chave}"
            )

        #Resposta caso o áudio estiver muito baixo, com chiado ou se ninguém falar nada    
        except sr.UnknownValueError:            
            resposta_final = "❌ Desculpe, mensagem de áudio muito baixa ou com ruído."
        #Resposta caso haja alguma falha de conexão com o serviço de transcrição
        except sr.RequestError:            
            resposta_final = "❌ Erro ao conectar com o serviço de reconhecimento de voz."
            
        #Envia o texto final (ou o erro) de volta para o usuário no Telegram
        await update.message.reply_text(resposta_final)

if __name__ == "__main__":
    # Substitua pelo Token que o BotFather te deu
    TOKEN = "8907361407:AAFJquDllJ-s-HwB6kPyYzl_Ex7AaugpBVg" 
    
    # Cria o objeto do bot principal
    bot = BotTelegram(TOKEN)
    
    # Inicia o monitoramento de mensagens
    bot.executar()

    
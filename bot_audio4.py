import os
import whisper
from deep_translator import GoogleTranslator

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
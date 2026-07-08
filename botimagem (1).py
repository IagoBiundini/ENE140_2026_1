import os
import uuid
import cv2
from ultralytics import YOLO
from botdotelegram import BotTelegram, log


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
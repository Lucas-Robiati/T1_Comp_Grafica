"""
=============================================================================
 SISTEMA DE ANÁLISE DE VÍDEO - VISÃO COMPUTACIONAL CLÁSSICA
=============================================================================
 Autor: Gerado para análise de movimento e rastreamento geométrico
 Descrição: Detecta movimento por subtração de fundo (MOG2) e rastreia
             objetos via Centroid Tracker. Sem IA / Deep Learning.
 Dependências: opencv-python, scipy, tkinter (nativo Python), Pillow
=============================================================================
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import queue
import time
import os
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.spatial import distance as dist
from collections import OrderedDict


# =============================================================================
#  CENTROID TRACKER — Rastreamento clássico por centroide geométrico
# =============================================================================

class CentroidTracker:
    """
    Rastreador de objetos baseado em distância euclidiana entre centroides.

    Lógica:
      1. Cada novo contorno vira um "objeto" com ID único.
      2. A cada frame, tentamos associar os novos centroides aos objetos
         existentes usando a menor distância euclidiana.
      3. Se um objeto ficar ausente por mais de `max_desaparecido` frames,
         ele é removido (esquecido), e seu ID nunca mais será reutilizado.
    """

    def __init__(self, max_desaparecido: int = 30, max_distancia: float = 80.0):
        """
        Parâmetros
        ----------
        max_desaparecido : int
            Quantos frames consecutivos sem detecção antes de descartar o objeto.
        max_distancia : float
            Distância máxima em pixels para considerar dois centroides o mesmo objeto.
        """
        self.proximo_id = 1                        # Contador global de IDs (começa em 1)
        self.objetos = OrderedDict()               # id → centroide (cx, cy)
        self.bboxes = OrderedDict()                # id → (x, y, w, h)
        self.desaparecidos = OrderedDict()         # id → contagem de frames ausentes
        self.max_desaparecido = max_desaparecido
        self.max_distancia = max_distancia
        self.ids_salvos = set()                    # IDs cujo recorte já foi salvo
        self.confirmados = set()                   # IDs confirmados pelo CSRT
        self.frames_visto = OrderedDict()          # id → frames consecutivos com CSRT ok

    def registrar(self, centroide: tuple, bbox: tuple) -> int:
        """Cria e registra um novo objeto, retorna o ID atribuído."""
        novo_id = self.proximo_id
        self.objetos[novo_id] = centroide
        self.bboxes[novo_id] = bbox
        self.desaparecidos[novo_id] = 0
        self.proximo_id += 1
        return novo_id

    def desregistrar(self, obj_id: int):
        """Remove um objeto que ficou ausente demais."""
        del self.objetos[obj_id]
        del self.bboxes[obj_id]
        del self.desaparecidos[obj_id]

    def atualizar(self, rects: list) -> OrderedDict:
        """
        Recebe lista de (x, y, w, h) dos contornos detectados no frame atual
        e retorna dicionário {id: centroide} com todos os objetos ativos.

        Parâmetros
        ----------
        rects : list[tuple]
            Lista de bounding boxes detectados: [(x, y, w, h), ...]

        Retorna
        -------
        OrderedDict {id: (cx, cy)}
        """
        # --- Nenhuma detecção neste frame ---
        if len(rects) == 0:
            # Incrementa ausência de todos os objetos ativos
            for obj_id in list(self.desaparecidos.keys()):
                self.desaparecidos[obj_id] += 1
                if self.desaparecidos[obj_id] > self.max_desaparecido:
                    self.desregistrar(obj_id)
            return self.objetos

        # Calcula centroides das detecções atuais
        centroides_input = []
        for (x, y, w, h) in rects:
            cx = x + w // 2
            cy = y + h // 2
            centroides_input.append((cx, cy))

        # --- Nenhum objeto rastreado ainda: registra todos ---
        if len(self.objetos) == 0:
            for i, c in enumerate(centroides_input):
                self.registrar(c, rects[i])

        else:
            # --- Associação por distância mínima (Hungarian simplificado) ---
            ids_existentes = list(self.objetos.keys())
            centroides_existentes = list(self.objetos.values())

            # Matriz de distâncias: linhas = existentes, colunas = novos
            D = dist.cdist(np.array(centroides_existentes),
                           np.array(centroides_input))

            # Ordena pares (distância, linha, coluna) do menor para maior
            linhas = D.min(axis=1).argsort()
            colunas = D.argmin(axis=1)[linhas]

            linhas_usadas = set()
            colunas_usadas = set()

            for (linha, coluna) in zip(linhas, colunas):
                if linha in linhas_usadas or coluna in colunas_usadas:
                    continue
                if D[linha, coluna] > self.max_distancia:
                    continue  # Muito distante: não são o mesmo objeto

                obj_id = ids_existentes[linha]
                self.objetos[obj_id] = centroides_input[coluna]
                self.bboxes[obj_id] = rects[coluna]
                self.desaparecidos[obj_id] = 0

                linhas_usadas.add(linha)
                colunas_usadas.add(coluna)

            # Linhas não associadas → aumenta ausência
            linhas_nao_usadas = set(range(len(ids_existentes))) - linhas_usadas
            for linha in linhas_nao_usadas:
                obj_id = ids_existentes[linha]
                self.desaparecidos[obj_id] += 1
                if self.desaparecidos[obj_id] > self.max_desaparecido:
                    self.desregistrar(obj_id)

            # Colunas não associadas → novos objetos
            colunas_nao_usadas = set(range(len(centroides_input))) - colunas_usadas
            for coluna in colunas_nao_usadas:
                self.registrar(centroides_input[coluna], rects[coluna])

        return self.objetos

    def esta_confirmado(self, obj_id: int) -> bool:
        """Retorna True se o objeto já foi confirmado pelo CSRT."""
        return obj_id in self.confirmados

    def confirmar(self, obj_id: int):
        """Marca um objeto como confirmado (validado pelo CSRT)."""
        self.confirmados.add(obj_id)

    def resetar(self):
        """Limpa todo o estado do tracker (para novo vídeo)."""
        self.proximo_id = 1
        self.objetos.clear()
        self.bboxes.clear()
        self.desaparecidos.clear()
        self.ids_salvos.clear()
        self.confirmados.clear()
        self.frames_visto.clear()


# =============================================================================
#  PROCESSADOR DE VÍDEO — Roda em thread separada
# =============================================================================

class ProcessadorVideo:
    """
    Encapsula toda a lógica de captura e análise de frames.
    Roda em thread secundária para não travar a GUI.

    Pipeline híbrido de 3 estágios:
      1. MOG2 detecta candidatos por subtração de fundo
      2. CentroidTracker associa IDs por distância euclidiana
      3. TrackerCSRT valida cada novo objeto — só confirma após N frames
    """

    # Pastas de saída
    PASTA_FRAMES   = "saida/frames"
    PASTA_OBJETOS  = "saida/objetos"

    # Limites do sistema CSRT
    MAX_CSRT_TRACKERS = 10   # Máximo de trackers CSRT simultâneos

    def __init__(self, fila_gui: queue.Queue):
        """
        Parâmetros
        ----------
        fila_gui : queue.Queue
            Fila para enviar imagens/status à thread principal da GUI.
        """
        self.fila_gui = fila_gui

        # Estado de controle
        self.rodando = False
        self.pausado = False
        self.caminho_video = None

        # Parâmetros ajustáveis pela GUI (com valores padrão)
        self.limiar_binarizacao = 25      # Threshold para máscara de movimento
        self.area_minima = 500            # Área mínima de contorno (px²)
        self.max_desaparecido = 30        # Frames de tolerância no tracker
        self.frames_confirmacao = 5       # Frames mínimos p/ CSRT confirmar objeto

        # Tracker de objetos (centroide)
        self.tracker = CentroidTracker(max_desaparecido=self.max_desaparecido)

        # Trackers CSRT ativos: {obj_id: cv2.legacy.TrackerCSRT}
        self.csrt_trackers = {}

        # Contador sequencial de frames salvos
        self.contador_frames = 0

    # ------------------------------------------------------------------
    #  Gerenciamento de pastas
    # ------------------------------------------------------------------

    def preparar_pastas(self):
        """
        Limpa e recria as pastas de saída.
        Chamado sempre que um novo vídeo é processado (regra de inicialização).
        """
        for pasta in [self.PASTA_FRAMES, self.PASTA_OBJETOS]:
            if os.path.exists(pasta):
                shutil.rmtree(pasta)   # Remove todo o conteúdo anterior
            os.makedirs(pasta)         # Recria vazia

    # ------------------------------------------------------------------
    #  Gerenciamento de CSRT trackers
    # ------------------------------------------------------------------

    def _criar_csrt(self, frame, bbox, obj_id):
        """
        Inicializa um TrackerCSRT para validar um novo objeto.

        Se o limite de trackers simultâneos for atingido, remove o
        tracker não-confirmado mais antigo para liberar espaço.
        """
        # Limita quantidade de CSRTs simultâneos
        if len(self.csrt_trackers) >= self.MAX_CSRT_TRACKERS:
            # Remove o mais antigo que não está confirmado
            for old_id in list(self.csrt_trackers.keys()):
                if not self.tracker.esta_confirmado(old_id):
                    del self.csrt_trackers[old_id]
                    if old_id in self.tracker.frames_visto:
                        del self.tracker.frames_visto[old_id]
                    break

        csrt = cv2.legacy.TrackerCSRT_create()
        (x, y, w, h) = bbox
        # Garante bbox dentro dos limites do frame
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w > 0 and h > 0:
            csrt.init(frame, (x, y, w, h))
            self.csrt_trackers[obj_id] = csrt
            self.tracker.frames_visto[obj_id] = 0

    def _atualizar_csrts(self, frame):
        """
        Atualiza todos os trackers CSRT ativos.

        Para cada tracker:
          - Se update() retorna success=True E o centroide converge com
            o CentroidTracker → incrementa contador de frames vistos
          - Se atingir frames_confirmacao → marca como confirmado
          - Se update() falha → remove o tracker (falso positivo)
        """
        ids_para_remover = []

        for obj_id, csrt in list(self.csrt_trackers.items()):
            # Objeto já confirmado: não precisa mais do CSRT
            if self.tracker.esta_confirmado(obj_id):
                ids_para_remover.append(obj_id)
                continue

            # Objeto foi desregistrado do centroide: limpa CSRT
            if obj_id not in self.tracker.objetos:
                ids_para_remover.append(obj_id)
                continue

            success, bbox_csrt = csrt.update(frame)

            if success:
                # Calcula centroide do CSRT
                (cx_csrt, cy_csrt) = (
                    int(bbox_csrt[0] + bbox_csrt[2] / 2),
                    int(bbox_csrt[1] + bbox_csrt[3] / 2)
                )
                # Centroide do CentroidTracker
                centroide_ct = self.tracker.objetos.get(obj_id)

                if centroide_ct is not None:
                    # Verifica convergência: distância entre centroides
                    dx = abs(cx_csrt - centroide_ct[0])
                    dy = abs(cy_csrt - centroide_ct[1])
                    dist_centroides = (dx**2 + dy**2) ** 0.5

                    # Se convergem (< max_distancia), incrementa
                    if dist_centroides < self.tracker.max_distancia:
                        self.tracker.frames_visto[obj_id] = \
                            self.tracker.frames_visto.get(obj_id, 0) + 1

                        # Atingiu frames mínimos → CONFIRMADO
                        if self.tracker.frames_visto[obj_id] >= self.frames_confirmacao:
                            self.tracker.confirmar(obj_id)
                            ids_para_remover.append(obj_id)
                    else:
                        # Centroides divergem: reseta contagem
                        self.tracker.frames_visto[obj_id] = 0
            else:
                # CSRT falhou → provável falso positivo
                ids_para_remover.append(obj_id)
                # Remove do CentroidTracker também
                if obj_id in self.tracker.objetos:
                    self.tracker.desregistrar(obj_id)

        # Limpa trackers finalizados
        for obj_id in ids_para_remover:
            if obj_id in self.csrt_trackers:
                del self.csrt_trackers[obj_id]

    # ------------------------------------------------------------------
    #  Loop principal de processamento
    # ------------------------------------------------------------------

    def iniciar(self, caminho: str):
        """Inicia o processamento em thread separada."""
        self.caminho_video = caminho
        self.rodando = True
        self.pausado = False
        t = threading.Thread(target=self._loop_processamento, daemon=True)
        t.start()

    def pausar(self):
        """Alterna entre pausado e rodando."""
        self.pausado = not self.pausado

    def parar(self):
        """Para o processamento."""
        self.rodando = False

    def _loop_processamento(self):
        """
        Loop principal — roda na thread secundária.
        Pipeline híbrido: MOG2 → CentroidTracker → CSRT (confirmação).
        """
        # --- Limpeza inicial ---
        self.preparar_pastas()
        self.tracker.resetar()
        self.csrt_trackers.clear()
        self.contador_frames = 0

        # --- Abre o vídeo ---
        cap = cv2.VideoCapture(self.caminho_video)
        if not cap.isOpened():
            self.fila_gui.put(("erro", "Não foi possível abrir o arquivo de vídeo."))
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30

        # --- Subtrator de fundo MOG2 (clássico, sem IA) ---
        subtrator = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )

        # Kernel morfológico para limpeza de ruídos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # IDs conhecidos antes de atualizar (para detectar novos)
        ids_anteriores = set()

        frame_idx = 0

        while self.rodando:
            if self.pausado:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            progresso = int((frame_idx / max(total_frames, 1)) * 100)

            # ── ESTÁGIO 1: Pré-processamento + MOG2 ───────────────────
            frame_exibicao = cv2.resize(frame, (800, 450))
            frame_cinza = cv2.cvtColor(frame_exibicao, cv2.COLOR_BGR2GRAY)
            frame_blur  = cv2.GaussianBlur(frame_cinza, (21, 21), 0)

            mascara = subtrator.apply(frame_blur)
            _, mascara = cv2.threshold(
                mascara, self.limiar_binarizacao, 255, cv2.THRESH_BINARY
            )
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
            mascara = cv2.dilate(mascara, kernel, iterations=2)

            contornos, _ = cv2.findContours(
                mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            rects_validos = []
            for c in contornos:
                if cv2.contourArea(c) < self.area_minima:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                rects_validos.append((x, y, w, h))

            # Salva frame se há movimento detectado
            if len(rects_validos) > 0:
                self.contador_frames += 1
                nome_frame = f"frame_{self.contador_frames}.jpg"
                caminho_frame = os.path.join(self.PASTA_FRAMES, nome_frame)
                cv2.imwrite(caminho_frame, frame_exibicao)

            # ── ESTÁGIO 2: CentroidTracker associa IDs ─────────────────
            ids_anteriores = set(self.tracker.objetos.keys())
            objetos = self.tracker.atualizar(rects_validos)
            ids_atuais = set(objetos.keys())

            # Detecta IDs novos (recém-registrados neste frame)
            ids_novos = ids_atuais - ids_anteriores

            # Inicializa CSRT para cada novo objeto
            for obj_id in ids_novos:
                if obj_id in self.tracker.bboxes:
                    self._criar_csrt(frame_exibicao, self.tracker.bboxes[obj_id], obj_id)

            # ── ESTÁGIO 3: CSRT valida objetos pendentes ───────────────
            self._atualizar_csrts(frame_exibicao)

            # Salva recorte SOMENTE de objetos confirmados pelo CSRT
            for obj_id, centroide in objetos.items():
                if obj_id not in self.tracker.ids_salvos \
                        and self.tracker.esta_confirmado(obj_id):
                    if obj_id in self.tracker.bboxes:
                        (x, y, w, h) = self.tracker.bboxes[obj_id]
                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(frame_exibicao.shape[1], x + w)
                        y2 = min(frame_exibicao.shape[0], y + h)
                        recorte = frame_exibicao[y1:y2, x1:x2]
                        if recorte.size > 0:
                            nome_obj = f"objeto_id_{obj_id}.jpg"
                            caminho_obj = os.path.join(self.PASTA_OBJETOS, nome_obj)
                            cv2.imwrite(caminho_obj, recorte)
                            self.tracker.ids_salvos.add(obj_id)

            # ── Anotação visual no frame ───────────────────────────────
            frame_anotado = frame_exibicao.copy()

            # Conta objetos confirmados para estatísticas
            n_confirmados = len(self.tracker.confirmados & ids_atuais)

            for obj_id, centroide in objetos.items():
                if obj_id in self.tracker.bboxes:
                    (x, y, w, h) = self.tracker.bboxes[obj_id]

                    # Cor depende do status: verde = confirmado, amarelo = pendente
                    if self.tracker.esta_confirmado(obj_id):
                        cor_bbox = (0, 255, 0)       # Verde — confirmado
                        cor_texto = (0, 255, 255)     # Ciano
                        status = ""
                    else:
                        cor_bbox = (0, 200, 255)      # Amarelo/laranja — pendente
                        cor_texto = (0, 200, 255)
                        frames_v = self.tracker.frames_visto.get(obj_id, 0)
                        status = f" ({frames_v}/{self.frames_confirmacao})"

                    cv2.rectangle(frame_anotado, (x, y), (x+w, y+h), cor_bbox, 2)
                    cv2.circle(frame_anotado, centroide, 4, (0, 0, 255), -1)
                    cv2.putText(
                        frame_anotado,
                        f"ID {obj_id}{status}",
                        (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        cor_texto,
                        2
                    )

            # Máscara de movimento em mini-preview
            mascara_rgb = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
            h_mini = frame_anotado.shape[0] // 4
            w_mini = frame_anotado.shape[1] // 4
            mini = cv2.resize(mascara_rgb, (w_mini, h_mini))
            frame_anotado[0:h_mini, frame_anotado.shape[1]-w_mini:] = mini

            # Informações de status no frame
            cv2.putText(
                frame_anotado,
                f"Frame: {frame_idx}/{total_frames}  |  "
                f"Ativos: {len(objetos)}  |  "
                f"Confirmados: {n_confirmados}  |  "
                f"Salvos: {self.contador_frames}",
                (8, frame_anotado.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (200, 200, 200),
                1
            )

            # ── Envia frame para a GUI ─────────────────────────────────
            frame_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
            self.fila_gui.put(("frame", frame_rgb, progresso,
                               len(objetos), self.contador_frames,
                               n_confirmados))

            time.sleep(1.0 / fps_video)

        cap.release()
        self.rodando = False
        n_confirmados_total = len(self.tracker.confirmados)
        self.fila_gui.put(("concluido",
                            self.contador_frames,
                            n_confirmados_total))


# =============================================================================
#  INTERFACE GRÁFICA — Thread principal
# =============================================================================

class AppAnaliseVideo:
    """
    Interface gráfica principal construída em Tkinter.
    Exibe o vídeo anotado, controles e painéis de estatísticas.
    """

    COR_FUNDO      = "#1e1e2e"
    COR_PAINEL     = "#2a2a3e"
    COR_ACENTO     = "#7c3aed"
    COR_SUCESSO    = "#10b981"
    COR_ALERTA     = "#f59e0b"
    COR_TEXTO      = "#e2e8f0"
    COR_SUBTEXTO   = "#94a3b8"
    COR_BOTAO_OK   = "#4f46e5"
    COR_BOTAO_STOP = "#dc2626"
    COR_BOTAO_PAU  = "#d97706"

    def __init__(self, raiz: tk.Tk):
        self.raiz = raiz
        self.raiz.title("Análise de Vídeo — Visão Computacional Clássica")
        self.raiz.configure(bg=self.COR_FUNDO)
        self.raiz.resizable(True, True)

        # Fila de comunicação entre threads
        self.fila = queue.Queue(maxsize=5)

        # Instância do processador
        self.processador = ProcessadorVideo(self.fila)

        # Variáveis de estado da GUI
        self.caminho_video = tk.StringVar(value="Nenhum arquivo selecionado")
        self.var_limiar      = tk.IntVar(value=25)
        self.var_area        = tk.IntVar(value=500)
        self.var_persist     = tk.IntVar(value=30)
        self.var_confirmacao = tk.IntVar(value=5)
        self.var_progresso   = tk.DoubleVar(value=0.0)
        self.ultimo_frame_rgb = None

        self._construir_ui()
        self._poll_fila()   # Inicia verificação periódica da fila

    # ------------------------------------------------------------------
    #  Construção da Interface
    # ------------------------------------------------------------------

    def _construir_ui(self):
        """Monta todos os widgets da janela principal."""
        # ── Título ────────────────────────────────────────────────────
        topo = tk.Frame(self.raiz, bg=self.COR_FUNDO)
        topo.pack(fill="x", padx=15, pady=(12, 4))

        tk.Label(
            topo,
            text="🎬  Analisador de Vídeo — Visão Clássica",
            font=("Segoe UI", 16, "bold"),
            bg=self.COR_FUNDO,
            fg=self.COR_TEXTO
        ).pack(side="left")

        tk.Label(
            topo,
            text="sem IA · OpenCV · MOG2 · Centroid + CSRT",
            font=("Segoe UI", 9),
            bg=self.COR_FUNDO,
            fg=self.COR_SUBTEXTO
        ).pack(side="left", padx=12)

        # ── Layout principal: vídeo + painel direito ───────────────────
        corpo = tk.Frame(self.raiz, bg=self.COR_FUNDO)
        corpo.pack(fill="both", expand=True, padx=15, pady=6)

        # Canvas do vídeo
        self.canvas = tk.Canvas(
            corpo,
            width=800, height=450,
            bg="#0f0f1a",
            highlightthickness=1,
            highlightbackground=self.COR_ACENTO
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._ao_redimensionar_canvas)

        # Painel direito (com Scrollbar para evitar corte de widgets)
        container_painel = tk.Frame(corpo, bg=self.COR_PAINEL, width=280)
        container_painel.pack(side="right", fill="y", padx=(10, 0))
        container_painel.pack_propagate(False)

        canvas_scroll = tk.Canvas(container_painel, bg=self.COR_PAINEL, highlightthickness=0)
        canvas_scroll.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(container_painel, orient="vertical", command=canvas_scroll.yview)
        scrollbar.pack(side="right", fill="y")

        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        # Frame interno que realmente conterá os widgets
        painel = tk.Frame(canvas_scroll, bg=self.COR_PAINEL)
        canvas_window = canvas_scroll.create_window((0, 0), window=painel, anchor="nw")

        def _on_frame_configure(event):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))

        def _on_canvas_configure(event):
            canvas_scroll.itemconfig(canvas_window, width=event.width)

        painel.bind("<Configure>", _on_frame_configure)
        canvas_scroll.bind("<Configure>", _on_canvas_configure)

        # Habilita scroll com a roda do mouse
        def _on_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_up(event):
            canvas_scroll.yview_scroll(-1, "units")

        def _on_mousewheel_down(event):
            canvas_scroll.yview_scroll(1, "units")

        self._painel_arquivo(painel)
        self._painel_parametros(painel)
        self._painel_controles(painel)
        self._painel_estatisticas(painel)

        # Registra a rolagem de mouse de forma recursiva em todos os filhos
        def _registrar_scroll_recursivo(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel_up)
            widget.bind("<Button-5>", _on_mousewheel_down)
            for filho in widget.winfo_children():
                _registrar_scroll_recursivo(filho)

        _registrar_scroll_recursivo(canvas_scroll)
        _registrar_scroll_recursivo(painel)

        # ── Barra de progresso ─────────────────────────────────────────
        rodape = tk.Frame(self.raiz, bg=self.COR_FUNDO)
        rodape.pack(fill="x", padx=15, pady=(4, 10))

        tk.Label(
            rodape, text="Progresso:",
            bg=self.COR_FUNDO, fg=self.COR_SUBTEXTO,
            font=("Segoe UI", 8)
        ).pack(side="left")

        self.barra_prog = ttk.Progressbar(
            rodape,
            variable=self.var_progresso,
            maximum=100,
            length=650,
            mode="determinate"
        )
        self.barra_prog.pack(side="left", padx=8, fill="x", expand=True)

        self.lbl_pct = tk.Label(
            rodape, text="0%",
            bg=self.COR_FUNDO, fg=self.COR_TEXTO,
            font=("Segoe UI", 8, "bold"), width=5
        )
        self.lbl_pct.pack(side="left")

    def _painel_arquivo(self, pai):
        """Seção de seleção de arquivo."""
        frame = self._secao(pai, "📁  Arquivo de Vídeo")

        tk.Label(
            frame,
            textvariable=self.caminho_video,
            bg=self.COR_PAINEL,
            fg=self.COR_SUBTEXTO,
            font=("Segoe UI", 8),
            wraplength=230,
            justify="left"
        ).pack(anchor="w", pady=(0, 6))

        tk.Button(
            frame,
            text="Selecionar Vídeo",
            command=self._selecionar_video,
            bg=self.COR_BOTAO_OK, fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2",
            padx=10, pady=5
        ).pack(fill="x")

    def _painel_parametros(self, pai):
        """Seção de sliders de parâmetros."""
        frame = self._secao(pai, "⚙️  Parâmetros de Detecção")

        # Slider: Limiar de binarização
        self._slider(
            frame,
            label="Limiar de Binarização",
            variavel=self.var_limiar,
            minval=5, maxval=100,
            callback=self._atualizar_limiar,
            dica="Menor = mais sensível ao movimento"
        )

        # Slider: Área mínima
        self._slider(
            frame,
            label="Área Mínima (px²)",
            variavel=self.var_area,
            minval=50, maxval=5000,
            callback=self._atualizar_area,
            dica="Filtra ruído por tamanho do contorno"
        )

        # Slider: Persistência do tracker
        self._slider(
            frame,
            label="Persistência (frames)",
            variavel=self.var_persist,
            minval=5, maxval=120,
            callback=self._atualizar_persist,
            dica="Frames sem detecção antes de esquecer objeto"
        )

        # Slider: Frames para confirmação CSRT
        self._slider(
            frame,
            label="Confirmação CSRT (frames)",
            variavel=self.var_confirmacao,
            minval=2, maxval=30,
            callback=self._atualizar_confirmacao,
            dica="Frames consecutivos p/ confirmar objeto"
        )

    def _painel_controles(self, pai):
        """Botões de controle: Iniciar / Pausar / Parar."""
        frame = self._secao(pai, "▶  Controles")

        self.btn_iniciar = tk.Button(
            frame,
            text="▶  Iniciar Processamento",
            command=self._iniciar,
            bg=self.COR_SUCESSO, fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2",
            padx=10, pady=6
        )
        self.btn_iniciar.pack(fill="x", pady=(0, 4))

        self.btn_pausar = tk.Button(
            frame,
            text="⏸  Pausar",
            command=self._pausar,
            bg=self.COR_BOTAO_PAU, fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2",
            padx=10, pady=6,
            state="disabled"
        )
        self.btn_pausar.pack(fill="x", pady=(0, 4))

        self.btn_parar = tk.Button(
            frame,
            text="⏹  Parar",
            command=self._parar,
            bg=self.COR_BOTAO_STOP, fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2",
            padx=10, pady=6,
            state="disabled"
        )
        self.btn_parar.pack(fill="x")

    def _painel_estatisticas(self, pai):
        """Painel de estatísticas em tempo real."""
        frame = self._secao(pai, "📊  Estatísticas em Tempo Real")

        self.stat_frames   = self._stat_linha(frame, "Frames com movimento:", "0")
        self.stat_objetos  = self._stat_linha(frame, "Confirmados (CSRT):",   "0")
        self.stat_ativos   = self._stat_linha(frame, "Objetos ativos agora:", "0")
        self.stat_status   = self._stat_linha(frame, "Status:",               "Aguardando")

    # ------------------------------------------------------------------
    #  Helpers de layout
    # ------------------------------------------------------------------

    def _secao(self, pai, titulo: str) -> tk.Frame:
        """Cria uma seção com título dentro do painel."""
        wrapper = tk.Frame(pai, bg=self.COR_PAINEL)
        wrapper.pack(fill="x", padx=10, pady=(10, 2))

        tk.Label(
            wrapper,
            text=titulo,
            bg=self.COR_PAINEL,
            fg=self.COR_ACENTO,
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", pady=(0, 4))

        sep = tk.Frame(wrapper, bg=self.COR_ACENTO, height=1)
        sep.pack(fill="x", pady=(0, 6))

        return wrapper

    def _slider(self, pai, label, variavel, minval, maxval, callback, dica=""):
        """Cria um slider com rótulo e exibição de valor de forma compacta."""
        row = tk.Frame(pai, bg=self.COR_PAINEL)
        row.pack(fill="x", pady=4)

        # Linha do topo: Rótulo à esquerda, Valor em destaque à direita
        topo_linha = tk.Frame(row, bg=self.COR_PAINEL)
        topo_linha.pack(fill="x")

        tk.Label(
            topo_linha, text=label,
            bg=self.COR_PAINEL, fg=self.COR_TEXTO,
            font=("Segoe UI", 8, "bold")
        ).pack(side="left", anchor="w")

        lbl_val = tk.Label(
            topo_linha, textvariable=variavel,
            bg=self.COR_PAINEL, fg=self.COR_ACENTO,
            font=("Segoe UI", 8, "bold")
        ).pack(side="right", anchor="e")

        if dica:
            tk.Label(
                row, text=dica,
                bg=self.COR_PAINEL, fg=self.COR_SUBTEXTO,
                font=("Segoe UI", 7)
            ).pack(anchor="w", pady=(0, 2))

        sl = tk.Scale(
            row,
            variable=variavel,
            from_=minval, to=maxval,
            orient="horizontal",
            bg=self.COR_PAINEL, fg=self.COR_TEXTO,
            troughcolor="#3b3b5c",
            highlightthickness=0,
            showvalue=False,
            command=callback
        )
        sl.pack(fill="x", expand=True)

    def _stat_linha(self, pai, rotulo: str, valor_inicial: str) -> tk.Label:
        """Cria uma linha de estatística com rótulo e valor dinâmico."""
        row = tk.Frame(pai, bg=self.COR_PAINEL)
        row.pack(fill="x", pady=2)

        tk.Label(
            row, text=rotulo,
            bg=self.COR_PAINEL, fg=self.COR_SUBTEXTO,
            font=("Segoe UI", 8)
        ).pack(side="left")

        lbl = tk.Label(
            row, text=valor_inicial,
            bg=self.COR_PAINEL, fg=self.COR_SUCESSO,
            font=("Segoe UI", 8, "bold")
        )
        lbl.pack(side="right")
        return lbl

    # ------------------------------------------------------------------
    #  Callbacks dos controles
    # ------------------------------------------------------------------

    def _selecionar_video(self):
        """
        Abre o gerenciador de arquivos nativo do sistema via zenity/kdialog.
        Fallback para tkinter.filedialog caso não esteja disponível.
        """
        caminho = None

        # Tenta zenity (GNOME/GTK)
        try:
            resultado = subprocess.run(
                [
                    "zenity", "--file-selection",
                    "--title=Selecionar Vídeo",
                    "--file-filter=Vídeos (mp4 avi mkv mov) | *.mp4 *.avi *.mkv *.mov *.wmv *.flv",
                    "--file-filter=Todos os arquivos | *"
                ],
                capture_output=True, text=True, timeout=120
            )
            if resultado.returncode == 0 and resultado.stdout.strip():
                caminho = resultado.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Tenta kdialog (KDE) se zenity falhou
        if caminho is None:
            try:
                resultado = subprocess.run(
                    [
                        "kdialog", "--getopenfilename",
                        os.path.expanduser("~"),
                        "Vídeos (*.mp4 *.avi *.mkv *.mov *.wmv *.flv)"
                    ],
                    capture_output=True, text=True, timeout=120
                )
                if resultado.returncode == 0 and resultado.stdout.strip():
                    caminho = resultado.stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # Fallback: tkinter filedialog
        if caminho is None:
            caminho = filedialog.askopenfilename(
                title="Selecionar Vídeo",
                filetypes=[
                    ("Arquivos de Vídeo", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"),
                    ("Todos os arquivos", "*.*")
                ]
            )

        if caminho:
            self.caminho_video.set(os.path.basename(caminho))
            self._caminho_completo = caminho

    def _iniciar(self):
        """Inicia o processamento do vídeo selecionado."""
        if not hasattr(self, "_caminho_completo") or not self._caminho_completo:
            messagebox.showwarning("Atenção", "Selecione um arquivo de vídeo primeiro.")
            return

        if self.processador.rodando:
            messagebox.showinfo("Info", "Já há um processamento em andamento.")
            return

        # Aplica parâmetros atuais
        self.processador.limiar_binarizacao = self.var_limiar.get()
        self.processador.area_minima        = self.var_area.get()
        self.processador.max_desaparecido   = self.var_persist.get()
        self.processador.tracker.max_desaparecido = self.var_persist.get()
        self.processador.frames_confirmacao = self.var_confirmacao.get()

        # Atualiza botões
        self.btn_iniciar.config(state="disabled")
        self.btn_pausar.config(state="normal", text="⏸  Pausar")
        self.btn_parar.config(state="normal")
        self.stat_status.config(text="Processando...", fg=self.COR_ALERTA)
        self.var_progresso.set(0)
        self.canvas.delete("placeholder")

        self.processador.iniciar(self._caminho_completo)

    def _pausar(self):
        """Alterna pausa/retomada."""
        if not self.processador.rodando:
            return
        self.processador.pausar()
        if self.processador.pausado:
            self.btn_pausar.config(text="▶  Retomar")
            self.stat_status.config(text="Pausado", fg=self.COR_ALERTA)
        else:
            self.btn_pausar.config(text="⏸  Pausar")
            self.stat_status.config(text="Processando...", fg=self.COR_ALERTA)

    def _parar(self):
        """Para o processamento."""
        self.processador.parar()
        self.btn_iniciar.config(state="normal")
        self.btn_pausar.config(state="disabled", text="⏸  Pausar")
        self.btn_parar.config(state="disabled")
        self.stat_status.config(text="Interrompido", fg=self.COR_BOTAO_STOP)

    def _atualizar_limiar(self, val):
        self.processador.limiar_binarizacao = int(val)

    def _atualizar_area(self, val):
        self.processador.area_minima = int(val)

    def _atualizar_persist(self, val):
        v = int(val)
        self.processador.max_desaparecido = v
        self.processador.tracker.max_desaparecido = v

    def _atualizar_confirmacao(self, val):
        self.processador.frames_confirmacao = int(val)

    # ------------------------------------------------------------------
    #  Poll da fila (atualização da GUI na thread principal)
    # ------------------------------------------------------------------

    def _poll_fila(self):
        """
        Verifica periodicamente se há mensagens da thread de processamento.
        Deve ser chamado somente na thread principal do Tkinter.
        """
        try:
            while True:
                msg = self.fila.get_nowait()
                tipo = msg[0]

                if tipo == "frame":
                    _, frame_rgb, progresso, n_ativos, n_frames, n_total = msg
                    self._exibir_frame(frame_rgb)
                    self.var_progresso.set(progresso)
                    self.lbl_pct.config(text=f"{progresso}%")
                    self.stat_frames.config(text=str(n_frames))
                    self.stat_objetos.config(text=str(n_total))
                    self.stat_ativos.config(text=str(n_ativos))

                elif tipo == "concluido":
                    _, n_frames, n_total = msg
                    self.btn_iniciar.config(state="normal")
                    self.btn_pausar.config(state="disabled", text="⏸  Pausar")
                    self.btn_parar.config(state="disabled")
                    self.var_progresso.set(100)
                    self.lbl_pct.config(text="100%")
                    self.stat_status.config(text="Concluído ✓", fg=self.COR_SUCESSO)
                    messagebox.showinfo(
                        "Processamento Concluído",
                        f"Análise finalizada!\n\n"
                        f"• Frames com movimento salvos: {n_frames}\n"
                        f"  → Pasta: saida/frames/\n\n"
                        f"• Objetos confirmados (CSRT): {n_total}\n"
                        f"  → Pasta: saida/objetos/\n\n"
                        f"ℹ Objetos que duraram < {self.var_confirmacao.get()} frames\n"
                        f"  foram descartados como falsos positivos."
                    )

                elif tipo == "erro":
                    messagebox.showerror("Erro", msg[1])
                    self.btn_iniciar.config(state="normal")
                    self.btn_pausar.config(state="disabled")
                    self.btn_parar.config(state="disabled")

        except queue.Empty:
            pass

        # Reagenda a cada 30ms (~33 fps para a GUI)
        self.raiz.after(30, self._poll_fila)

    def _exibir_frame(self, frame_rgb: np.ndarray):
        """Converte um array NumPy RGB em imagem Tkinter e exibe no canvas, mantendo a proporção."""
        self.ultimo_frame_rgb = frame_rgb
        img_pil = Image.fromarray(frame_rgb)

        # Ajusta ao tamanho atual do canvas mantendo proporção (Aspect Ratio)
        cw = self.canvas.winfo_width()  or 800
        ch = self.canvas.winfo_height() or 450

        largura_original, altura_original = img_pil.size
        proporcao_img = largura_original / altura_original
        proporcao_canvas = cw / ch

        if proporcao_canvas > proporcao_img:
            # Canvas é mais largo que a imagem (barra preta nas laterais)
            nova_altura = ch
            nova_largura = int(ch * proporcao_img)
        else:
            # Canvas é mais alto que a imagem (barra preta no topo/baixo)
            nova_largura = cw
            nova_altura = int(cw / proporcao_img)

        # Redimensiona mantendo qualidade e proporção (valores mínimos de 1 para evitar crash de tamanho 0)
        img_pil = img_pil.resize((max(nova_largura, 1), max(nova_altura, 1)), Image.LANCZOS)

        self._img_tk = ImageTk.PhotoImage(img_pil)
        
        # Centraliza a imagem no Canvas
        x_pos = (cw - nova_largura) // 2
        y_pos = (ch - nova_altura) // 2

        self.canvas.delete("all")
        self.canvas.create_image(x_pos, y_pos, anchor="nw", image=self._img_tk)

    def _ao_redimensionar_canvas(self, event):
        """Manipula o redimensionamento do canvas para atualizar o frame ou centralizar o placeholder."""
        if hasattr(self, "ultimo_frame_rgb") and self.ultimo_frame_rgb is not None:
            self._exibir_frame(self.ultimo_frame_rgb)
        else:
            cw = event.width
            ch = event.height
            self.canvas.delete("all")
            self.canvas.create_text(
                cw // 2, ch // 2,
                text="Selecione um arquivo de vídeo para começar",
                fill=self.COR_SUBTEXTO,
                font=("Segoe UI", 13),
                tags="placeholder"
            )


# =============================================================================
#  PONTO DE ENTRADA
# =============================================================================

def main():
    raiz = tk.Tk()
    raiz.geometry("1150x640")
    raiz.minsize(950, 550)

    # Estilo da barra de progresso via ttk
    estilo = ttk.Style()
    estilo.theme_use("clam")
    estilo.configure(
        "TProgressbar",
        troughcolor="#2a2a3e",
        background="#7c3aed",
        thickness=14
    )

    app = AppAnaliseVideo(raiz)
    raiz.protocol("WM_DELETE_WINDOW", lambda: (app.processador.parar(), raiz.destroy()))
    raiz.mainloop()


if __name__ == "__main__":
    main()

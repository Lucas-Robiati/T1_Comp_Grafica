# 🎬 Analisador de Vídeo — Visão Computacional Clássica

---

## Instalação

```bash
pip install -r requirements.txt
```

> **tkinter** é nativo do Python. Se não disponível:
> - Ubuntu/Debian: `sudo apt-get install python3-tk`
> - macOS (Homebrew): `brew install python-tk`

---

## Execução

```bash
python analise_video.py
```

---

## Estrutura de Saídas

```
saida/
├── frames/
│   ├── frame_1.jpg      ← frame com movimento detectado
│   ├── frame_2.jpg
│   └── ...
└── objetos/
    ├── objeto_id_1.jpg  ← recorte do 1º objeto ao aparecer
    ├── objeto_id_2.jpg
    └── ...
```

> ⚠️ As pastas são **apagadas e recriadas** a cada novo processamento.

---

## Arquitetura Técnica

### Pipeline de Detecção (por frame)

```
Frame bruto
    ↓ Redimensionamento (800×450)
    ↓ Escala de cinza + Gaussian Blur
    ↓ MOG2 Background Subtractor
    ↓ Threshold binário (limiar configurável)
    ↓ Morfologia: OPEN → CLOSE → Dilate
    ↓ findContours → filtra por área mínima
    ↓
    ├─ Filtro 1: salva frame em saida/frames/
    └─ Filtro 2: Centroid Tracker → salva recortes em saida/objetos/
```

### Centroid Tracker (Rastreamento Clássico)

| Conceito | Implementação |
|----------|---------------|
| Identificação | ID inteiro único e sequencial (nunca reutilizado) |
| Associação | Distância euclidiana mínima via `scipy.spatial.distance.cdist` |
| Persistência | Contador de frames ausentes por objeto |
| Descarte | Objeto removido após `max_desaparecido` frames sem detecção |
| Salvamento | Recorte salvo **uma única vez** por ID, ao primeiro aparecimento |

---

## Parâmetros da GUI

| Parâmetro | Padrão | Efeito |
|-----------|--------|--------|
| **Limiar de Binarização** | 25 | Menor = mais sensível ao movimento |
| **Área Mínima (px²)** | 500 | Filtra ruído pequeno (pixels isolados) |
| **Persistência (frames)** | 30 | Quantos frames o objeto pode sumir sem ser esquecido |

---

## Tecnologias Utilizadas

- `cv2.createBackgroundSubtractorMOG2` — subtração de fundo estatística (Mistura de Gaussianas)
- `cv2.morphologyEx` — limpeza morfológica (OPEN, CLOSE, DILATE)
- `cv2.findContours` — extração de contornos geométricos
- `scipy.spatial.distance.cdist` — matriz de distâncias euclidianas
- `tkinter` + `threading` + `queue` — GUI não bloqueante
- `Pillow (PIL)` — conversão BGR→RGB para exibição no canvas Tkinter

Melhor versão do código:
"""
=============================================================================
 SISTEMA DE ANÁLISE DE VÍDEO - MÓDULO DE PROCESSAMENTO DE IMAGEM
=============================================================================
 Descrição: Contém a lógica de processamento de vídeo clássico usando OpenCV,
             detecção por MOG2 e rastreamento híbrido CentroidTracker + CSRT.
=============================================================================
"""

import os
import shutil
import time
import threading
import queue
import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

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

        # Baixo desenpenho, melhor para vídeos internos com muito ruido.
        #subtrator = cv2.createBackgroundSubtractorKNN(
        #    history=500,
        #    dist2Threshold=400,
        #    detectShadows=True
        #)
        
        subtrator = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=float(self.limiar_binarizacao * 10.0),
            detectShadows=False
        )

        # Kernel morfológico para limpeza de ruídos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Buffer para média temporal de frames (reduz tremulação/ruído rápido)
        buffer_frames = []

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

            # ── ESTÁGIO 1: Pré-processamento + KNN ───────────────────
            frame_exibicao = cv2.resize(frame, (800, 450))
            frame_cinza = cv2.cvtColor(frame_exibicao, cv2.COLOR_BGR2GRAY)
            
            # Adiciona o frame atual ao buffer de média temporal
            buffer_frames.append(frame_cinza)
            if len(buffer_frames) > 3:
                buffer_frames.pop(0)

            # Calcula a média temporal dos frames no buffer
            frame_cinza_medio = np.mean(buffer_frames, axis=0).astype(np.uint8)
            
            frame_blur  = cv2.GaussianBlur(frame_cinza_medio, (21, 21), 0)

            # Ajusta dinamicamente o limiar de variância do MOG2
            subtrator.setVarThreshold(float(self.limiar_binarizacao * 10.0))

            mascara = subtrator.apply(frame_blur)
            
            # Limiariza a 250 para eliminar sombras (valor 127) e manter apenas movimento real (255)
            _, mascara = cv2.threshold(
                mascara, 250, 255, cv2.THRESH_BINARY
            )
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
            mascara = cv2.dilate(mascara, kernel, iterations=2)

            # Calcula porcentagem de pixels em movimento para filtrar tremulações generalizadas de câmera
            pixels_ativos = cv2.countNonZero(mascara)
            proporcao_movimento = (pixels_ativos / mascara.size) * 100

            contornos, _ = cv2.findContours(
                mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            rects_validos = []
            # Se a movimentação global for menor que 12% da tela, processa os contornos individuais
            if proporcao_movimento < 12.0:
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
'''
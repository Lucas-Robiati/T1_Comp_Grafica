"""
 Pipeline: MOG2 + CLAHE + NMS + Kalman(Húngaro) + CSRT(validador)
"""

import os
import shutil
import time
import threading
import queue
from collections import OrderedDict, deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


#  NMS — Non-Maximum Suppression para bboxes

def nms_bboxes(bboxes, iou_thresh=0.45):
    """Remove bboxes sobrepostas, mantendo a de maior área."""
    if len(bboxes) == 0:
        return []

    bboxes = list(bboxes)
    areas = [w * h for (x, y, w, h) in bboxes]
    indices = sorted(range(len(bboxes)), key=lambda i: areas[i], reverse=True)

    keep = []
    while indices:
        i = indices.pop(0)
        keep.append(i)
        remaining = []
        for j in indices:
            if _iou_pair(bboxes[i], bboxes[j]) < iou_thresh:
                remaining.append(j)
        indices = remaining

    return [bboxes[i] for i in keep]


def _iou_pair(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


#  KALMAN TRACKER — Rastreamento com Kalman + Húngaro + anti-duplicata

class KalmanTracker:

    def __init__(self, max_desaparecido=30, iou_minimo=0.20, min_dist_novo=50):
        self.proximo_id = 1
        self.filtros = OrderedDict()
        self.bboxes = OrderedDict()
        self.desaparecidos = OrderedDict()
        self.frames_visto = OrderedDict()
        self.confirmados = set()
        self.ids_salvos = set()
        self.max_desaparecido = max_desaparecido    # Tempo de sobrevida de um objeto antes esquecê-lo
        self.iou_minimo = iou_minimo                # Remove bboxes sobrepostas, mantendo a de maior area
        self.min_dist_novo = min_dist_novo          # Distância mínima para registrar novo obj

    def _criar_filtro(self, cx, cy):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1],
             [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        kf.statePost = np.array(
            [[float(cx)], [float(cy)], [0.0], [0.0]], dtype=np.float32)
        return kf

    @staticmethod
    def _centroide(bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def _dist_centroide(self, c1, c2):
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** 0.5

    def _perto_de_existente(self, cx, cy):
        #Retorna True se (cx,cy) está muito perto de um objeto já ativo
        for fid in self.filtros:
            if fid in self.bboxes:
                ec = self._centroide(self.bboxes[fid])
                if self._dist_centroide((cx, cy), ec) < self.min_dist_novo:
                    return True
        return False

    def registrar(self, cx, cy, bbox):
        fid = self.proximo_id
        self.filtros[fid] = self._criar_filtro(cx, cy)
        self.bboxes[fid] = bbox
        self.desaparecidos[fid] = 0
        self.frames_visto[fid] = 0
        self.proximo_id += 1
        return fid

    def _remover(self, fid):
        for d in (self.filtros, self.bboxes, self.desaparecidos, self.frames_visto):
            d.pop(fid, None)

    def predizer_bboxes(self):
        preditos = {}
        for fid, kf in self.filtros.items():
            pred = kf.predict()
            cx_p, cy_p = int(pred[0, 0]), int(pred[1, 0])
            if fid in self.bboxes:
                _, _, w, h = self.bboxes[fid]
                preditos[fid] = (cx_p - w // 2, cy_p - h // 2, w, h)
        return preditos

    def atualizar(self, rects):

        preditos = self.predizer_bboxes()

        if len(rects) == 0:
            for fid in list(self.desaparecidos):
                self.desaparecidos[fid] += 1
                if self.desaparecidos[fid] > self.max_desaparecido:
                    self._remover(fid)
            return self._resultado()

        if len(preditos) == 0:
            for r in rects:
                cx, cy = r[0] + r[2] // 2, r[1] + r[3] // 2
                self.registrar(cx, cy, r)
            return self._resultado()

        # Matriz de custo: 1 - IoU
        ids_exist = list(preditos.keys())
        n_e, n_d = len(ids_exist), len(rects)
        custo = np.ones((n_e, n_d), dtype=np.float32)
        for i, fid in enumerate(ids_exist):
            for j, r in enumerate(rects):
                custo[i, j] = 1.0 - _iou_pair(preditos[fid], r)

        linhas, colunas = linear_sum_assignment(custo)
        usados_i, usados_j = set(), set()

        for i, j in zip(linhas, colunas):
            iou_val = 1.0 - custo[i, j]
            if iou_val < self.iou_minimo:
                continue
            fid = ids_exist[i]
            r = rects[j]
            cx, cy = r[0] + r[2] // 2, r[1] + r[3] // 2
            self.filtros[fid].correct(
                np.array([[np.float32(cx)], [np.float32(cy)]]))
            self.bboxes[fid] = r
            self.desaparecidos[fid] = 0
            usados_i.add(i)
            usados_j.add(j)

        for i in set(range(n_e)) - usados_i:
            fid = ids_exist[i]
            self.desaparecidos[fid] += 1
            if self.desaparecidos[fid] > self.max_desaparecido:
                self._remover(fid)

        # Novos objetos — com cooldown de distância
        for j in set(range(n_d)) - usados_j:
            r = rects[j]
            cx, cy = r[0] + r[2] // 2, r[1] + r[3] // 2
            if not self._perto_de_existente(cx, cy):
                self.registrar(cx, cy, r)

        return self._resultado()

    def _resultado(self):
        return OrderedDict(
            (fid, self._centroide(self.bboxes[fid]))
            for fid in self.filtros if fid in self.bboxes)

    def esta_confirmado(self, fid):
        return fid in self.confirmados

    def confirmar(self, fid):
        self.confirmados.add(fid)

    def resetar(self):
        self.proximo_id = 1
        self.filtros.clear()
        self.bboxes.clear()
        self.desaparecidos.clear()
        self.frames_visto.clear()
        self.confirmados.clear()
        self.ids_salvos.clear()

#  PROCESSADOR DE VÍDEO

class ProcessadorVideo:

    PASTA_FRAMES = "saida/frames"
    PASTA_OBJETOS = "saida/objetos"
    MAX_CSRT_TRACKERS = 8
    MOG2_HISTORY = 500
    MOG2_VAR_THRESHOLD = 50.0     # Aumentado para ignorar ruído leve
    MOG2_BINARY_THRESHOLD = 254
    MIN_EXTENSAO = 0.45           # Aumentado: contorno deve preencher bem a bbox
    MIN_SOLIDEZ = 0.55            # Aumentado: filtra formas muito irregulares (ruído)
    MIN_LADO_BBOX = 30            # Ignora bboxes menores que 30px
    NMS_IOU_THRESH = 0.45
    N_FRAMES_NITIDEZ = 10
    WARMUP_FRAMES = 0             # Zerado para não perder frames iniciais

    def __init__(self, fila_gui):
        self.fila_gui = fila_gui
        self.rodando = False
        self.pausado = False
        self.caminho_video = None

        self.var_threshold = self.MOG2_VAR_THRESHOLD
        self.area_minima = 800
        self.max_desaparecido = 30
        self.frames_confirmacao = 6  # Exige mais frames para validar (anti-ruído rápido)

        self.tracker = KalmanTracker(max_desaparecido=self.max_desaparecido)
        self.csrt_trackers = {}
        self._buffer_nitidez = {}
        self._histogramas_salvos = {}  # id → histograma HSV normalizado
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.contador_frames = 0

    def preparar_pastas(self):
        for pasta in [self.PASTA_FRAMES, self.PASTA_OBJETOS]:
            if os.path.exists(pasta):
                shutil.rmtree(pasta)
            os.makedirs(pasta)

    def _preprocessar_frame(self, frame):
        frame_exibicao = cv2.resize(frame, (800, 450))
        frame_cinza = cv2.cvtColor(frame_exibicao, cv2.COLOR_BGR2GRAY)
        frame_cinza = self._clahe.apply(frame_cinza)
        return frame_exibicao, frame_cinza

    @staticmethod
    def _nitidez(recorte):
        cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY) \
                if len(recorte.shape) == 3 else recorte
        return float(cv2.Laplacian(cinza, cv2.CV_64F).var())

    def _calc_histograma(self, recorte):
        hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _eh_duplicata_aparencia(self, recorte, limiar=0.35):
        hist_novo = self._calc_histograma(recorte)
        for oid, hist_salvo in self._histogramas_salvos.items():
            dist = cv2.compareHist(hist_novo, hist_salvo, cv2.HISTCMP_BHATTACHARYYA)
            if dist < limiar:
                return True
        return False

    # --- CSRT ---

    def _criar_csrt(self, frame, bbox, obj_id):
        if len(self.csrt_trackers) >= self.MAX_CSRT_TRACKERS:
            for old_id in list(self.csrt_trackers.keys()):
                if not self.tracker.esta_confirmado(old_id):
                    self._remover_csrt(old_id)
                    break

        factory = getattr(cv2, "TrackerCSRT_create", None) \
                  or cv2.legacy.TrackerCSRT_create
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        if w > 0 and h > 0:
            csrt = factory()
            csrt.init(frame, (x, y, w, h))
            self.csrt_trackers[obj_id] = csrt
            self.tracker.frames_visto[obj_id] = 0
            self._buffer_nitidez[obj_id] = deque(maxlen=self.N_FRAMES_NITIDEZ)

    def _remover_csrt(self, obj_id):
        self.csrt_trackers.pop(obj_id, None)
        self._buffer_nitidez.pop(obj_id, None)

    def _atualizar_csrts(self, frame):
        """CSRT como validador puro — NÃO corrige o Kalman."""
        ids_remover = []

        for obj_id, csrt in list(self.csrt_trackers.items()):
            if self.tracker.esta_confirmado(obj_id):
                # Já confirmado: alimenta buffer de nitidez mais um pouco, depois remove
                if obj_id in self.tracker.bboxes:
                    x, y, w, h = self.tracker.bboxes[obj_id]
                    recorte = frame[max(0,y):min(frame.shape[0],y+h),
                                    max(0,x):min(frame.shape[1],x+w)]
                    if recorte.size > 0:
                        buf = self._buffer_nitidez.get(obj_id)
                        if buf is not None and len(buf) < self.N_FRAMES_NITIDEZ:
                            buf.append((self._nitidez(recorte), recorte.copy()))
                ids_remover.append(obj_id)
                continue

            if obj_id not in self.tracker.filtros:
                ids_remover.append(obj_id)
                continue

            success, bbox_csrt = csrt.update(frame)

            if not success:
                ids_remover.append(obj_id)
                if not self.tracker.esta_confirmado(obj_id):
                    self.tracker._remover(obj_id)
                continue

            x_c, y_c = max(0, int(bbox_csrt[0])), max(0, int(bbox_csrt[1]))
            w_c = min(int(bbox_csrt[2]), frame.shape[1] - x_c)
            h_c = min(int(bbox_csrt[3]), frame.shape[0] - y_c)
            if w_c <= 0 or h_c <= 0:
                ids_remover.append(obj_id)
                continue

            bbox_kt = self.tracker.bboxes.get(obj_id)
            if bbox_kt:
                iou = _iou_pair((x_c, y_c, w_c, h_c), bbox_kt)
                if iou >= 0.15:
                    self.tracker.frames_visto[obj_id] = \
                        self.tracker.frames_visto.get(obj_id, 0) + 1
                else:
                    self.tracker.frames_visto[obj_id] = max(
                        0, self.tracker.frames_visto.get(obj_id, 0) - 1)

            # Alimenta buffer de nitidez
            recorte = frame[y_c:y_c + h_c, x_c:x_c + w_c]
            if recorte.size > 0:
                self._buffer_nitidez.setdefault(
                    obj_id, deque(maxlen=self.N_FRAMES_NITIDEZ)
                ).append((self._nitidez(recorte), recorte.copy()))

            if self.tracker.frames_visto.get(obj_id, 0) >= self.frames_confirmacao:
                self.tracker.confirmar(obj_id)

        for obj_id in ids_remover:
            self._remover_csrt(obj_id)

    def _salvar_melhor_recorte(self, obj_id):
        buf = self._buffer_nitidez.get(obj_id)
        if not buf:
            return

        melhor_nit, melhor_rec = max(buf, key=lambda x: x[0])

        # Deduplicação por aparência
        if melhor_rec.size > 0 and self._eh_duplicata_aparencia(melhor_rec):
            self.tracker.ids_salvos.add(obj_id)
            return

        nome = f"objeto_id_{obj_id}.jpg"
        cv2.imwrite(os.path.join(self.PASTA_OBJETOS, nome), melhor_rec)
        self.tracker.ids_salvos.add(obj_id)

        if melhor_rec.size > 0:
            self._histogramas_salvos[obj_id] = self._calc_histograma(melhor_rec)

    # --- Controles ---

    def iniciar(self, caminho):
        self.caminho_video = caminho
        self.rodando = True
        self.pausado = False
        t = threading.Thread(target=self._loop_processamento, daemon=True)
        t.start()

    def pausar(self):
        self.pausado = not self.pausado

    def parar(self):
        self.rodando = False

    # --- Loop Principal ---

    def _loop_processamento(self):
        self.preparar_pastas()
        self.tracker.resetar()
        self.csrt_trackers.clear()
        self._buffer_nitidez.clear()
        self._histogramas_salvos.clear()
        self.contador_frames = 0

        cap = cv2.VideoCapture(self.caminho_video)
        if not cap.isOpened():
            self.fila_gui.put(("erro", "Não foi possível abrir o arquivo de vídeo."))
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

        subtrator = cv2.createBackgroundSubtractorMOG2(
            history=self.MOG2_HISTORY,
            varThreshold=self.var_threshold,
            detectShadows=True)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # Maior para matar ruído
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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

            # Pré-processamento
            frame_exibicao, frame_cinza = self._preprocessar_frame(frame)
            frame_blur = cv2.GaussianBlur(frame_cinza, (5, 5), 0)

            # MOG2
            mascara = subtrator.apply(frame_blur)
            mascara[mascara == 127] = 0  # Remove sombras
            _, mascara = cv2.threshold(
                mascara, self.MOG2_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

            # Zona de warmup — ignora detecções durante estabilização
            if frame_idx <= self.WARMUP_FRAMES:
                frame_rgb = cv2.cvtColor(frame_exibicao, cv2.COLOR_BGR2RGB)
                cv2.putText(frame_exibicao, f"Estabilizando MOG2... {frame_idx}/{self.WARMUP_FRAMES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                frame_rgb = cv2.cvtColor(frame_exibicao, cv2.COLOR_BGR2RGB)
                self.fila_gui.put(("frame", frame_rgb, progresso, 0, 0, 0))
                time.sleep(1.0 / fps_video)
                continue

            mascara = cv2.medianBlur(mascara, 5)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_open)
            mascara = cv2.dilate(mascara, kernel_dilate, iterations=1)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_close)

            # Contornos
            contornos, _ = cv2.findContours(
                mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rects_brutos = []
            for c in contornos:
                area_c = cv2.contourArea(c)
                if area_c < self.area_minima:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if w <= 0 or h <= 0 or min(w, h) < self.MIN_LADO_BBOX:
                    continue
                razao = float(w) / float(h)
                if not (0.15 < razao < 6.5):
                    continue
                area_bbox = float(w * h)
                if area_bbox <= 0:
                    continue
                extensao = area_c / area_bbox
                if extensao < self.MIN_EXTENSAO:
                    continue
                hull = cv2.convexHull(c)
                area_hull = cv2.contourArea(hull)
                solidez = area_c / area_hull if area_hull > 0 else 0.0
                if solidez < self.MIN_SOLIDEZ:
                    continue
                rects_brutos.append((x, y, w, h))

            # NMS — elimina bboxes sobrepostas
            rects_validos = nms_bboxes(rects_brutos, self.NMS_IOU_THRESH)

            # Salva frame só se há objetos confirmados ativos
            tem_confirmado_ativo = bool(
                self.tracker.confirmados & set(self.tracker.filtros.keys()))
            if rects_validos and tem_confirmado_ativo:
                self.contador_frames += 1
                cv2.imwrite(
                    os.path.join(self.PASTA_FRAMES,
                                 f"frame_{self.contador_frames}.jpg"),
                    frame_exibicao)

            ids_anteriores = set(self.tracker.filtros.keys())
            objetos = self.tracker.atualizar(rects_validos)
            ids_novos = set(objetos.keys()) - ids_anteriores

            # CSRT para novos objetos
            for obj_id in ids_novos:
                if obj_id in self.tracker.bboxes:
                    self._criar_csrt(
                        frame_exibicao, self.tracker.bboxes[obj_id], obj_id)

            # Atualiza CSRTs (validação pura)
            self._atualizar_csrts(frame_exibicao)

            # Salva melhor recorte de objetos confirmados
            for obj_id in list(objetos.keys()):
                if obj_id not in self.tracker.ids_salvos \
                        and self.tracker.esta_confirmado(obj_id):
                    self._salvar_melhor_recorte(obj_id)

            # Anotação visual
            frame_anotado = frame_exibicao.copy()
            ids_atuais = set(objetos.keys())
            n_confirmados = len(self.tracker.confirmados & ids_atuais)

            for obj_id, centroide in objetos.items():
                if obj_id not in self.tracker.bboxes:
                    continue
                x, y, w, h = self.tracker.bboxes[obj_id]

                if self.tracker.esta_confirmado(obj_id):
                    cor_bbox, cor_texto, status = (0, 255, 0), (0, 255, 255), ""
                else:
                    fv = self.tracker.frames_visto.get(obj_id, 0)
                    cor_bbox = (0, 200, 255)
                    cor_texto = (0, 200, 255)
                    status = f" ({fv}/{self.frames_confirmacao})"

                cv2.rectangle(frame_anotado, (x, y), (x+w, y+h), cor_bbox, 2)
                cv2.circle(frame_anotado, centroide, 4, (0, 0, 255), -1)
                cv2.putText(frame_anotado, f"ID {obj_id}{status}",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, cor_texto, 2)

                if obj_id in self.tracker.filtros:
                    st = self.tracker.filtros[obj_id].statePost
                    vel = (float(st[2, 0])**2 + float(st[3, 0])**2) ** 0.5
                    cv2.putText(frame_anotado, f"{vel:.1f}px/f",
                                (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.40, (180, 180, 180), 1)

            # Mini-preview da máscara
            mascara_rgb = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
            h_m = frame_anotado.shape[0] // 4
            w_m = frame_anotado.shape[1] // 4
            mini = cv2.resize(mascara_rgb, (w_m, h_m))
            frame_anotado[0:h_m, frame_anotado.shape[1] - w_m:] = mini

            cv2.putText(
                frame_anotado,
                f"Frame: {frame_idx}/{total_frames}  |  "
                f"Ativos: {len(objetos)}  |  "
                f"Confirmados: {n_confirmados}  |  "
                f"Salvos: {self.contador_frames}",
                (8, frame_anotado.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            frame_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
            self.fila_gui.put((
                "frame", frame_rgb, progresso,
                len(objetos), self.contador_frames, n_confirmados))

            time.sleep(1.0 / fps_video)

        cap.release()
        self.rodando = False
        n_conf_total = len(self.tracker.confirmados)
        self.fila_gui.put(("concluido", self.contador_frames, n_conf_total))

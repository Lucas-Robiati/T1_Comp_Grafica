"""
tracker.py - Rastreamento de objetos ao longo do vídeo via IoU.

Implementa um tracker simples e robusto baseado em Intersection over Union (IoU)
para associar detecções entre frames e manter identidades únicas dos objetos.
Projetado para funcionar bem com frame skipping em vídeos longos.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Calcula o Intersection over Union (IoU) entre dois bounding boxes.

    Args:
        box_a: [x1, y1, x2, y2] do primeiro box.
        box_b: [x1, y1, x2, y2] do segundo box.

    Returns:
        Valor IoU entre 0.0 e 1.0.
    """
    # Coordenadas da interseção
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # Área da interseção
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    # Áreas dos boxes individuais
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    # União
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


@dataclass
class TrackedObject:
    """Representa um objeto rastreado ao longo de múltiplos frames."""
    track_id: int                           # ID único do objeto
    class_name: str                         # Classe do objeto (ex: "person")
    class_id: int                           # ID da classe COCO
    first_seen: float = 0.0                 # Timestamp de primeiro aparecimento (s)
    last_seen: float = 0.0                  # Timestamp de último aparecimento (s)
    bboxes: List[List[float]] = field(default_factory=list)   # Histórico de bboxes
    timestamps: List[float] = field(default_factory=list)     # Timestamps de cada bbox
    confidences: List[float] = field(default_factory=list)    # Confiança de cada detecção
    frames_since_seen: int = 0              # Contagem de frames sem detecção

    @property
    def last_bbox(self) -> Optional[List[float]]:
        """Retorna o último bounding box registrado."""
        return self.bboxes[-1] if self.bboxes else None

    @property
    def avg_confidence(self) -> float:
        """Retorna a confiança média das detecções."""
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)

    @property
    def centers(self) -> List[Tuple[float, float]]:
        """Retorna a lista de centros (cx, cy) do histórico de bboxes."""
        result = []
        for bbox in self.bboxes:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            result.append((cx, cy))
        return result

    def to_dict(self) -> dict:
        """Serializa o objeto rastreado para JSON."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "first_seen_seconds": round(self.first_seen, 2),
            "last_seen_seconds": round(self.last_seen, 2),
            "duration_seconds": round(self.last_seen - self.first_seen, 2),
            "avg_confidence": round(self.avg_confidence, 4),
            "num_detections": len(self.bboxes),
            "bboxes": [
                {
                    "timestamp": round(ts, 2),
                    "x1": round(b[0], 1),
                    "y1": round(b[1], 1),
                    "x2": round(b[2], 1),
                    "y2": round(b[3], 1),
                }
                for ts, b in zip(self.timestamps, self.bboxes)
            ],
        }


class IoUTracker:
    """
    Tracker de objetos baseado em IoU (Intersection over Union).

    Associa detecções de cada frame a tracks existentes usando IoU.
    Cria novos tracks para detecções não-associadas. Remove tracks
    que não recebem detecções por 'max_age' atualizações consecutivas.

    Projetado para funcionar com frame skipping: usa a classe da detecção
    como critério adicional de associação para compensar maiores
    deslocamentos entre frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        """
        Args:
            iou_threshold: IoU mínimo para associar detecção a track existente.
            max_age: Número máximo de updates sem detecção antes de perder o track.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id: int = 1
        self.active_tracks: Dict[int, TrackedObject] = {}
        self.finished_tracks: Dict[int, TrackedObject] = {}

        logger.info(
            f"IoUTracker inicializado (iou_threshold={iou_threshold}, "
            f"max_age={max_age})"
        )

    def update(self, detections: list, timestamp: float) -> List[TrackedObject]:
        """
        Atualiza o tracker com novas detecções de um frame.

        Args:
            detections: Lista de Detection do frame atual.
            timestamp: Tempo em segundos do frame no vídeo.

        Returns:
            Lista de TrackedObjects ativos após a atualização.
        """
        if not detections:
            # Incrementa idade de todos os tracks ativos
            tracks_to_remove = []
            for track_id, track in self.active_tracks.items():
                track.frames_since_seen += 1
                if track.frames_since_seen > self.max_age:
                    tracks_to_remove.append(track_id)

            for track_id in tracks_to_remove:
                self.finished_tracks[track_id] = self.active_tracks.pop(track_id)
                logger.debug(f"Track {track_id} finalizado (max_age atingido)")

            return list(self.active_tracks.values())

        # Cria matriz de IoU entre detecções e tracks ativos
        active_list = list(self.active_tracks.values())
        matched_det_indices = set()
        matched_track_ids = set()

        if active_list:
            # Calcula IoU matrix
            iou_matrix = np.zeros((len(detections), len(active_list)))
            for d_idx, det in enumerate(detections):
                for t_idx, track in enumerate(active_list):
                    if track.last_bbox is not None:
                        # Só associa se for da mesma classe
                        if det.class_name == track.class_name:
                            iou_matrix[d_idx, t_idx] = compute_iou(
                                det.bbox, track.last_bbox
                            )

            # Greedy matching: associa pares com maior IoU primeiro
            while True:
                if iou_matrix.size == 0:
                    break

                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break

                d_idx, t_idx = np.unravel_index(
                    iou_matrix.argmax(), iou_matrix.shape
                )

                det = detections[d_idx]
                track = active_list[t_idx]

                # Atualiza o track existente com a nova detecção
                track.bboxes.append(det.bbox)
                track.timestamps.append(timestamp)
                track.confidences.append(det.confidence)
                track.last_seen = timestamp
                track.frames_since_seen = 0

                matched_det_indices.add(d_idx)
                matched_track_ids.add(track.track_id)

                # Invalida linhas e colunas já usadas
                iou_matrix[d_idx, :] = -1
                iou_matrix[:, t_idx] = -1

        # Cria novos tracks para detecções não-associadas
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                new_track = TrackedObject(
                    track_id=self.next_id,
                    class_name=det.class_name,
                    class_id=det.class_id,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    bboxes=[det.bbox],
                    timestamps=[timestamp],
                    confidences=[det.confidence],
                    frames_since_seen=0,
                )
                self.active_tracks[self.next_id] = new_track
                logger.debug(
                    f"Novo track #{self.next_id}: {det.class_name} "
                    f"(conf={det.confidence:.2f}) em t={timestamp:.1f}s"
                )
                self.next_id += 1

        # Incrementa idade e remove tracks não-detectados
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if track_id not in matched_track_ids:
                track.frames_since_seen += 1
                if track.frames_since_seen > self.max_age:
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            self.finished_tracks[track_id] = self.active_tracks.pop(track_id)
            logger.debug(f"Track {track_id} finalizado (max_age atingido)")

        return list(self.active_tracks.values())

    def get_all_tracks(self) -> Dict[int, TrackedObject]:
        """
        Retorna todos os tracks (ativos + finalizados).

        Returns:
            Dicionário {track_id: TrackedObject} com todos os objetos rastreados.
        """
        all_tracks = {}
        all_tracks.update(self.finished_tracks)
        all_tracks.update(self.active_tracks)
        return all_tracks

    def get_summary(self) -> dict:
        """
        Retorna um resumo estatístico do rastreamento.

        Returns:
            Dicionário com total de objetos, contagem por classe, e top classes.
        """
        all_tracks = self.get_all_tracks()
        class_counts: Dict[str, int] = {}

        for track in all_tracks.values():
            class_counts[track.class_name] = (
                class_counts.get(track.class_name, 0) + 1
            )

        # Top classes ordenadas por frequência
        sorted_classes = sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "total_unique_objects": len(all_tracks),
            "class_counts": class_counts,
            "top_classes": sorted_classes[:3],
        }

    def reset(self):
        """Limpa todos os tracks e reinicia o contador de IDs."""
        self.active_tracks.clear()
        self.finished_tracks.clear()
        self.next_id = 1
        logger.info("Tracker resetado.")

"""
detector.py - Wrapper para detecção de objetos com YOLOv8.

Encapsula o modelo YOLOv8 da Ultralytics para fornecer uma interface
limpa de detecção de objetos em frames de vídeo.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Representa uma detecção individual de objeto em um frame."""
    bbox: List[float]       # [x1, y1, x2, y2] coordenadas do bounding box
    class_id: int           # ID da classe no dataset COCO
    class_name: str         # Nome da classe (ex: "person", "car")
    confidence: float       # Confiança da detecção (0.0 a 1.0)

    @property
    def center(self) -> tuple:
        """Retorna o centro (cx, cy) do bounding box."""
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        return (cx, cy)

    @property
    def area(self) -> float:
        """Retorna a área do bounding box."""
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return max(0, w) * max(0, h)


class ObjectDetector:
    """
    Detector de objetos usando YOLOv8 da Ultralytics.

    O modelo é baixado automaticamente na primeira execução.
    Suporta inferência em CPU e GPU (CUDA).
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.4
    ):
        """
        Inicializa o detector com o modelo YOLOv8.

        Args:
            model_name: Nome do modelo (ex: yolov8n.pt, yolov8s.pt).
            confidence_threshold: Confiança mínima para aceitar detecções.

        Raises:
            RuntimeError: Se o modelo não puder ser carregado.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None

        try:
            from ultralytics import YOLO
            logger.info(f"Carregando modelo '{model_name}'...")
            self.model = YOLO(model_name)
            logger.info(
                f"Modelo '{model_name}' carregado com sucesso. "
                f"Classes disponíveis: {len(self.model.names)}"
            )
        except ImportError:
            logger.error(
                "Biblioteca 'ultralytics' não encontrada. "
                "Instale com: pip install ultralytics"
            )
            raise RuntimeError("ultralytics não está instalado.")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo '{model_name}': {e}")
            raise RuntimeError(f"Falha ao carregar modelo: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detecta objetos em um frame de vídeo.

        Args:
            frame: Frame do vídeo como array NumPy (BGR, formato OpenCV).

        Returns:
            Lista de Detection com os objetos encontrados acima do threshold.
        """
        if self.model is None:
            logger.error("Modelo não está carregado.")
            return []

        try:
            # Inferência com verbose=False para não poluir o console
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False
            )

            detections: List[Detection] = []

            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes
                for i in range(len(boxes)):
                    # Coordenadas do bounding box (x1, y1, x2, y2)
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_name = self.model.names.get(class_id, f"class_{class_id}")

                    detections.append(Detection(
                        bbox=bbox,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence
                    ))

            logger.debug(
                f"Detectados {len(detections)} objetos no frame "
                f"(threshold={self.confidence_threshold})"
            )
            return detections

        except Exception as e:
            logger.error(f"Erro durante detecção: {e}")
            return []

    def get_class_names(self) -> dict:
        """Retorna o mapeamento de class_id -> class_name do modelo."""
        if self.model is None:
            return {}
        return dict(self.model.names)

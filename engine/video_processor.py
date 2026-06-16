"""
video_processor.py - Orquestrador principal do pipeline de processamento.

Coordena a leitura do vídeo (OpenCV), detecção (YOLOv8), rastreamento (IoU),
análise de vibes, e geração de todos os arquivos de saída.
"""

import json
import logging
import os
import time
import threading
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from engine.detector import ObjectDetector
from engine.tracker import IoUTracker
from plugins.vibe_prospecting import VibeProspector

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processador principal de vídeo: leitura -> detecção -> rastreamento -> vibes -> saída.
    Roda em thread separada da GUI com callbacks de progresso/log e cancelamento.
    """

    def __init__(self, config, cancel_event=None, progress_callback=None, log_callback=None):
        self.config = config
        self.cancel_event = cancel_event or threading.Event()
        self.progress_callback = progress_callback
        self.log_callback = log_callback

        proc = config.get("processing", {})
        self.frame_interval_seconds = proc.get("frame_interval_seconds", 2)
        self.confidence_threshold = proc.get("confidence_threshold", 0.4)
        self.model_name = proc.get("model_name", "yolov8n.pt")
        self.iou_threshold = proc.get("iou_threshold", 0.3)
        self.max_age = proc.get("max_age", 10)
        self.output_dir = config.get("output", {}).get("directory", "output")

    def _log(self, message):
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)

    def _update_progress(self, value):
        if self.progress_callback:
            self.progress_callback(value)

    def process(self, video_path):
        start_time = time.time()
        result = {"status": "error", "total_unique_objects": 0, "top_classes": [],
                  "class_counts": {}, "output_files": [], "error_message": ""}
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")

            self._log(f"Abrindo vídeo: {os.path.basename(video_path)}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self._log(f"Vídeo: {width}x{height}, {fps:.1f} FPS, {total_frames} frames, "
                       f"duração: {duration_seconds / 60:.1f} min")

            frame_interval = max(1, int(fps * self.frame_interval_seconds))
            frames_to_process = total_frames // frame_interval
            self._log(f"Processando 1 frame a cada {self.frame_interval_seconds}s "
                       f"(intervalo={frame_interval}, ~{frames_to_process} frames)")

            self._log("Carregando modelo de detecção...")
            detector = ObjectDetector(self.model_name, self.confidence_threshold)
            tracker = IoUTracker(self.iou_threshold, self.max_age)
            vibe_prospector = VibeProspector()
            track_color_samples = {}

            self._log("Iniciando processamento...")
            frame_count = 0
            processed_count = 0

            while True:
                if self.cancel_event.is_set():
                    self._log("Processamento cancelado pelo usuário.")
                    result["status"] = "cancelled"
                    cap.release()
                    return result

                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                if (frame_count - 1) % frame_interval != 0:
                    continue

                processed_count += 1
                timestamp = frame_count / fps if fps > 0 else 0

                detections = detector.detect(frame)
                active_tracks = tracker.update(detections, timestamp)

                # Capturar crops para análise de cor
                for det in detections:
                    for track in active_tracks:
                        if track.last_bbox is not None and track.last_bbox == det.bbox:
                            x1, y1 = max(0, int(det.bbox[0])), max(0, int(det.bbox[1]))
                            x2 = min(frame.shape[1], int(det.bbox[2]))
                            y2 = min(frame.shape[0], int(det.bbox[3]))
                            if x2 > x1 and y2 > y1:
                                track_color_samples[track.track_id] = frame[y1:y2, x1:x2]
                            break

                progress = (frame_count / total_frames) * 100
                self._update_progress(progress)

                if processed_count % 10 == 0 or processed_count == 1:
                    total_tracks = len(tracker.get_all_tracks())
                    self._log(f"  Frame {frame_count}/{total_frames} "
                               f"(t={timestamp:.1f}s) | Det: {len(detections)} | "
                               f"Obj únicos: {total_tracks}")

            cap.release()
            self._update_progress(100.0)

            elapsed = time.time() - start_time
            all_tracks = tracker.get_all_tracks()
            summary = tracker.get_summary()

            self._log(f"Processamento concluído em {elapsed:.1f}s")
            self._log(f"  Frames processados: {processed_count}/{total_frames}")
            self._log(f"  Objetos únicos: {summary['total_unique_objects']}")
            self._log(f"  Top classes: {summary['top_classes']}")

            os.makedirs(self.output_dir, exist_ok=True)

            # Gerar object_map.json
            object_map_path = os.path.join(self.output_dir, "object_map.json")
            object_map = {
                "video_info": {"path": video_path, "fps": fps,
                               "total_frames": total_frames,
                               "duration_seconds": round(duration_seconds, 2),
                               "resolution": f"{width}x{height}"},
                "processing_info": {"frame_interval_seconds": self.frame_interval_seconds,
                                    "frames_processed": processed_count,
                                    "model": self.model_name,
                                    "confidence_threshold": self.confidence_threshold,
                                    "processing_time_seconds": round(elapsed, 2)},
                "summary": {"total_unique_objects": summary["total_unique_objects"],
                             "class_counts": summary["class_counts"],
                             "top_3_classes": [{"class": c, "count": n}
                                               for c, n in summary["top_classes"]]},
                "objects": [t.to_dict() for t in sorted(all_tracks.values(),
                            key=lambda t: t.track_id)],
            }
            with open(object_map_path, "w", encoding="utf-8") as f:
                json.dump(object_map, f, ensure_ascii=False, indent=2)
            self._log(f"Mapeamento salvo: {object_map_path}")
            result["output_files"].append(object_map_path)

            # Análise de Vibes
            self._log("Executando análise de vibes...")
            vibe_files = vibe_prospector.analyze_all_tracks(
                all_tracks=all_tracks, color_samples=track_color_samples,
                output_dir=self.output_dir, fps=fps, frame_interval=frame_interval)
            result["output_files"].extend(vibe_files)
            for vf in vibe_files:
                self._log(f"Relatório: {vf}")

            result["status"] = "success"
            result["total_unique_objects"] = summary["total_unique_objects"]
            result["top_classes"] = summary["top_classes"]
            result["class_counts"] = summary["class_counts"]
            self._log(f"Todos os arquivos gerados em: {self.output_dir}/")
            return result

        except FileNotFoundError as e:
            result["error_message"] = str(e)
            self._log(f"ERRO: {e}")
            return result
        except RuntimeError as e:
            result["error_message"] = str(e)
            self._log(f"ERRO: {e}")
            return result
        except Exception as e:
            logger.error(f"Erro inesperado: {e}", exc_info=True)
            result["error_message"] = str(e)
            self._log(f"ERRO inesperado: {e}")
            return result

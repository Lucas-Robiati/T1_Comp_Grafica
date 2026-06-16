"""
vibe_prospecting.py - Plugin de análise de "vibes" dos objetos detectados.

Atribui uma vibe a cada objeto com base em:
- Cor predominante (HSV): energética, calma, neutra
- Movimento (velocidade média): agitada, moderada, estática
Gera relatórios CSV e TXT com estatísticas e insights de marketing.
"""

import csv
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VibeProspector:
    """Analisa vibes visuais e de movimento dos objetos rastreados."""

    # Faixas de Hue (OpenCV usa 0-179 para HSV)
    # Quentes: 0-15 ou 165-179 (mapeando 0-30° e 330-360° do círculo)
    # Frias: 90-135 (mapeando 180-270°)
    WARM_HUE_RANGES = [(0, 15), (165, 179)]
    COLD_HUE_RANGE = (90, 135)

    # Thresholds de velocidade (pixels/segundo)
    SPEED_AGITATED = 50.0
    SPEED_MODERATE = 10.0

    def analyze_color(self, crop: np.ndarray) -> str:
        """
        Analisa a cor predominante de um crop (região do objeto).

        Args:
            crop: Imagem BGR recortada do bounding box.

        Returns:
            "energética", "calma" ou "neutra".
        """
        if crop is None or crop.size == 0:
            return "neutra"

        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mean_hue = float(np.mean(hsv[:, :, 0]))

            # Verificar se é cor quente
            for low, high in self.WARM_HUE_RANGES:
                if low <= mean_hue <= high:
                    return "energética"

            # Verificar se é cor fria
            if self.COLD_HUE_RANGE[0] <= mean_hue <= self.COLD_HUE_RANGE[1]:
                return "calma"

            return "neutra"

        except Exception as e:
            logger.debug(f"Erro ao analisar cor: {e}")
            return "neutra"

    def analyze_movement(
        self,
        positions: List[Tuple[float, float]],
        timestamps: List[float],
    ) -> Tuple[str, float]:
        """
        Calcula a velocidade média do objeto e classifica o movimento.

        Args:
            positions: Lista de centros (cx, cy) ao longo do tempo.
            timestamps: Lista de timestamps correspondentes.

        Returns:
            Tupla (classificação, velocidade_media) onde classificação é
            "agitada", "moderada" ou "estática".
        """
        if len(positions) < 2 or len(timestamps) < 2:
            return "estática", 0.0

        total_speed = 0.0
        count = 0

        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            dist = math.sqrt(dx * dx + dy * dy)
            dt = timestamps[i] - timestamps[i - 1]

            if dt > 0:
                speed = dist / dt
                total_speed += speed
                count += 1

        avg_speed = total_speed / count if count > 0 else 0.0

        if avg_speed > self.SPEED_AGITATED:
            return "agitada", avg_speed
        elif avg_speed > self.SPEED_MODERATE:
            return "moderada", avg_speed
        else:
            return "estática", avg_speed

    def compute_vibe(self, color_vibe: str, movement_vibe: str) -> str:
        """Combina vibes de cor e movimento em uma vibe final."""
        return f"{color_vibe}/{movement_vibe}"

    def _generate_insights(self, vibe_counts: Dict[str, int]) -> List[str]:
        """Gera insights de marketing baseados nas vibes predominantes."""
        insights = []
        total = sum(vibe_counts.values())
        if total == 0:
            return ["Nenhum objeto detectado para análise."]

        # Ordenar por frequência
        sorted_vibes = sorted(vibe_counts.items(), key=lambda x: x[1], reverse=True)
        top_vibe = sorted_vibes[0][0] if sorted_vibes else ""
        top_pct = (sorted_vibes[0][1] / total * 100) if sorted_vibes else 0

        # Contar dimensões
        energetic = sum(v for k, v in vibe_counts.items() if "energética" in k)
        calm = sum(v for k, v in vibe_counts.items() if "calma" in k)
        agitated = sum(v for k, v in vibe_counts.items() if "agitada" in k)
        static = sum(v for k, v in vibe_counts.items() if "estática" in k)

        insights.append(f"Vibe predominante: '{top_vibe}' ({top_pct:.1f}% dos objetos)")

        if energetic > calm:
            insights.append(
                "Ambiente com predominância de cores quentes e energéticas – "
                "ideal para ações de esportes, alta performance e promoções vibrantes."
            )
        elif calm > energetic:
            insights.append(
                "Ambiente com predominância de tons frios e calmos – "
                "ideal para branding de bem-estar, tecnologia e produtos premium."
            )
        else:
            insights.append(
                "Equilíbrio entre cores quentes e frias – ambiente versátil "
                "para campanhas diversificadas."
            )

        if agitated > static:
            insights.append(
                "Alta movimentação detectada – conteúdo dinâmico ideal para "
                "vídeos de ação, esportes e experiências imersivas."
            )
        elif static > agitated:
            insights.append(
                "Objetos predominantemente estáticos – ideal para catálogos "
                "visuais, cenários e ambientação de marca."
            )

        return insights

    def analyze_all_tracks(
        self,
        all_tracks: dict,
        color_samples: Dict[int, np.ndarray],
        output_dir: str,
        fps: float = 30.0,
        frame_interval: int = 1,
    ) -> List[str]:
        """
        Analisa todos os tracks e gera relatórios de vibes.

        Returns:
            Lista de caminhos dos arquivos gerados.
        """
        output_files = []
        vibe_data = []
        vibe_counts: Dict[str, int] = {}

        for track_id, track in all_tracks.items():
            # Análise de cor
            crop = color_samples.get(track_id)
            color_vibe = self.analyze_color(crop)

            # Análise de movimento
            centers = track.centers
            movement_vibe, avg_speed = self.analyze_movement(centers, track.timestamps)

            # Vibe combinada
            combined = self.compute_vibe(color_vibe, movement_vibe)
            vibe_counts[combined] = vibe_counts.get(combined, 0) + 1

            vibe_data.append({
                "track_id": track_id,
                "class_name": track.class_name,
                "color_vibe": color_vibe,
                "movement_vibe": movement_vibe,
                "combined_vibe": combined,
                "avg_speed_px_s": round(avg_speed, 2),
                "num_detections": len(track.bboxes),
                "duration_s": round(track.last_seen - track.first_seen, 2),
            })

        # Gerar CSV
        csv_path = os.path.join(output_dir, "vibe_analysis.csv")
        try:
            fieldnames = ["track_id", "class_name", "color_vibe", "movement_vibe",
                          "combined_vibe", "avg_speed_px_s", "num_detections", "duration_s"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(vibe_data)
            output_files.append(csv_path)
            logger.info(f"CSV de vibes salvo: {csv_path}")
        except Exception as e:
            logger.error(f"Erro ao gerar CSV: {e}")

        # Gerar relatório TXT
        report_path = os.path.join(output_dir, "vibe_prospecting_report.txt")
        try:
            insights = self._generate_insights(vibe_counts)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("  RELATÓRIO DE PROSPECÇÃO DE VIBES\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Total de objetos analisados: {len(vibe_data)}\n\n")
                f.write("--- Distribuição de Vibes ---\n")
                for vibe, count in sorted(vibe_counts.items(),
                                           key=lambda x: x[1], reverse=True):
                    pct = count / len(vibe_data) * 100 if vibe_data else 0
                    f.write(f"  {vibe:30s}  {count:4d} objetos ({pct:5.1f}%)\n")
                f.write(f"\n--- Insights para Marketing/Vendas ---\n")
                for insight in insights:
                    f.write(f"  • {insight}\n")
                f.write("\n" + "=" * 60 + "\n")
            output_files.append(report_path)
            logger.info(f"Relatório de vibes salvo: {report_path}")
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")

        return output_files

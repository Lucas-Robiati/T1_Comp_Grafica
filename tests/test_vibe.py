"""
test_vibe.py - Testes unitários para o módulo de Vibe Prospecting.

Testa classificação de cor (HSV), velocidade de movimento e vibes combinadas.
"""

import numpy as np
import pytest
from plugins.vibe_prospecting import VibeProspector


class TestColorAnalysis:
    """Testes para análise de cor (HSV)."""

    def setup_method(self):
        self.prospector = VibeProspector()

    def test_warm_color_vibe(self):
        """Cores com Hue quente (0-15 no OpenCV) devem ser 'energética'."""
        # Criar imagem vermelha (Hue ~0 no OpenCV HSV)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :] = [0, 0, 255]  # BGR vermelho puro
        vibe = self.prospector.analyze_color(img)
        assert vibe == "energética"

    def test_cold_color_vibe(self):
        """Cores com Hue frio (90-135 no OpenCV) devem ser 'calma'."""
        # Criar imagem azul (Hue ~120 no OpenCV HSV)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :] = [255, 0, 0]  # BGR azul puro
        vibe = self.prospector.analyze_color(img)
        assert vibe == "calma"

    def test_neutral_color_vibe(self):
        """Cores com Hue intermediário devem ser 'neutra'."""
        # Criar imagem verde (Hue ~60 no OpenCV HSV)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :] = [0, 255, 0]  # BGR verde puro
        vibe = self.prospector.analyze_color(img)
        assert vibe == "neutra"

    def test_empty_crop(self):
        """Crop vazio deve retornar 'neutra'."""
        img = np.zeros((0, 0, 3), dtype=np.uint8)
        vibe = self.prospector.analyze_color(img)
        assert vibe == "neutra"

    def test_none_crop(self):
        """Crop None deve retornar 'neutra'."""
        vibe = self.prospector.analyze_color(None)
        assert vibe == "neutra"


class TestMovementAnalysis:
    """Testes para análise de movimento."""

    def setup_method(self):
        self.prospector = VibeProspector()

    def test_static_movement(self):
        """Objeto quase parado deve ser 'estática'."""
        positions = [(100, 100), (101, 101), (102, 100)]
        timestamps = [0.0, 1.0, 2.0]
        vibe, speed = self.prospector.analyze_movement(positions, timestamps)
        assert vibe == "estática"
        assert speed < 10.0

    def test_moderate_movement(self):
        """Objeto com velocidade moderada (10-50 px/s)."""
        positions = [(100, 100), (130, 100), (160, 100)]
        timestamps = [0.0, 1.0, 2.0]
        vibe, speed = self.prospector.analyze_movement(positions, timestamps)
        assert vibe == "moderada"
        assert 10.0 <= speed <= 50.0

    def test_agitated_movement(self):
        """Objeto rápido (>50 px/s) deve ser 'agitada'."""
        positions = [(100, 100), (200, 100), (300, 100)]
        timestamps = [0.0, 1.0, 2.0]
        vibe, speed = self.prospector.analyze_movement(positions, timestamps)
        assert vibe == "agitada"
        assert speed > 50.0

    def test_single_position(self):
        """Com apenas uma posição, deve ser 'estática'."""
        positions = [(100, 100)]
        timestamps = [0.0]
        vibe, speed = self.prospector.analyze_movement(positions, timestamps)
        assert vibe == "estática"
        assert speed == 0.0

    def test_empty_positions(self):
        """Lista vazia deve retornar 'estática'."""
        vibe, speed = self.prospector.analyze_movement([], [])
        assert vibe == "estática"
        assert speed == 0.0


class TestCombinedVibe:
    """Testes para vibes combinadas."""

    def setup_method(self):
        self.prospector = VibeProspector()

    def test_combined_vibe_format(self):
        """Vibe combinada deve ter formato 'cor/movimento'."""
        vibe = self.prospector.compute_vibe("energética", "agitada")
        assert vibe == "energética/agitada"

    def test_combined_calm_static(self):
        vibe = self.prospector.compute_vibe("calma", "estática")
        assert vibe == "calma/estática"

    def test_combined_neutral_moderate(self):
        vibe = self.prospector.compute_vibe("neutra", "moderada")
        assert vibe == "neutra/moderada"

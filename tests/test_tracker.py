"""
test_tracker.py - Testes unitários para o tracker IoU.

Testa cálculo de IoU, criação de tracks, associação e perda de tracks.
"""

import pytest
from engine.tracker import compute_iou, IoUTracker, TrackedObject
from engine.detector import Detection


class TestComputeIoU:
    """Testes para a função compute_iou."""

    def test_identical_boxes(self):
        """Boxes idênticos devem ter IoU = 1.0."""
        box = [0, 0, 100, 100]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Boxes sem sobreposição devem ter IoU = 0.0."""
        box_a = [0, 0, 50, 50]
        box_b = [100, 100, 200, 200]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Boxes com sobreposição parcial devem ter IoU entre 0 e 1."""
        box_a = [0, 0, 100, 100]
        box_b = [50, 50, 150, 150]
        iou = compute_iou(box_a, box_b)
        # Interseção: 50x50 = 2500, União: 10000 + 10000 - 2500 = 17500
        expected = 2500 / 17500
        assert iou == pytest.approx(expected, abs=1e-4)

    def test_contained_box(self):
        """Box contido dentro de outro."""
        box_a = [0, 0, 100, 100]
        box_b = [25, 25, 75, 75]
        iou = compute_iou(box_a, box_b)
        # Interseção: 50x50 = 2500, União: 10000 + 2500 - 2500 = 10000
        expected = 2500 / 10000
        assert iou == pytest.approx(expected, abs=1e-4)

    def test_zero_area_box(self):
        """Box com área zero deve retornar IoU 0."""
        box_a = [0, 0, 0, 0]
        box_b = [0, 0, 100, 100]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_touching_edges(self):
        """Boxes tocando nas bordas sem sobreposição real."""
        box_a = [0, 0, 50, 50]
        box_b = [50, 0, 100, 50]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)


class TestIoUTracker:
    """Testes para o IoUTracker."""

    def test_new_track_creation(self):
        """Detecções sem tracks existentes devem criar novos tracks."""
        tracker = IoUTracker(iou_threshold=0.3, max_age=5)
        detections = [
            Detection(bbox=[10, 10, 50, 50], class_id=0,
                      class_name="person", confidence=0.9),
            Detection(bbox=[200, 200, 300, 300], class_id=2,
                      class_name="car", confidence=0.8),
        ]

        active = tracker.update(detections, timestamp=0.0)
        assert len(active) == 2

        all_tracks = tracker.get_all_tracks()
        assert len(all_tracks) == 2
        classes = {t.class_name for t in all_tracks.values()}
        assert classes == {"person", "car"}

    def test_track_association(self):
        """Detecção próxima deve ser associada ao track existente."""
        tracker = IoUTracker(iou_threshold=0.2, max_age=5)

        # Frame 1: criar track
        det1 = [Detection(bbox=[10, 10, 60, 60], class_id=0,
                          class_name="person", confidence=0.9)]
        tracker.update(det1, timestamp=0.0)

        # Frame 2: detecção deslocada mas com IoU suficiente
        det2 = [Detection(bbox=[15, 15, 65, 65], class_id=0,
                          class_name="person", confidence=0.85)]
        tracker.update(det2, timestamp=1.0)

        all_tracks = tracker.get_all_tracks()
        assert len(all_tracks) == 1  # Deve ser o mesmo track

        track = list(all_tracks.values())[0]
        assert len(track.bboxes) == 2
        assert track.first_seen == 0.0
        assert track.last_seen == 1.0

    def test_track_loss(self):
        """Track sem detecção por max_age frames deve ser finalizado."""
        tracker = IoUTracker(iou_threshold=0.3, max_age=2)

        # Frame 1: criar track
        det = [Detection(bbox=[10, 10, 50, 50], class_id=0,
                         class_name="person", confidence=0.9)]
        tracker.update(det, timestamp=0.0)
        assert len(tracker.active_tracks) == 1

        # Frames sem detecção (simula max_age+1 = 3 updates vazios)
        tracker.update([], timestamp=1.0)
        tracker.update([], timestamp=2.0)
        tracker.update([], timestamp=3.0)

        # Track deve ter sido movido para finished
        assert len(tracker.active_tracks) == 0
        assert len(tracker.finished_tracks) == 1

    def test_different_class_no_association(self):
        """Detecções de classes diferentes não devem ser associadas."""
        tracker = IoUTracker(iou_threshold=0.2, max_age=5)

        det1 = [Detection(bbox=[10, 10, 60, 60], class_id=0,
                          class_name="person", confidence=0.9)]
        tracker.update(det1, timestamp=0.0)

        # Mesma posição mas classe diferente
        det2 = [Detection(bbox=[10, 10, 60, 60], class_id=2,
                          class_name="car", confidence=0.85)]
        tracker.update(det2, timestamp=1.0)

        all_tracks = tracker.get_all_tracks()
        assert len(all_tracks) == 2  # Devem ser tracks separados

    def test_summary(self):
        """Resumo deve conter estatísticas corretas."""
        tracker = IoUTracker(iou_threshold=0.3, max_age=5)

        detections = [
            Detection(bbox=[10, 10, 50, 50], class_id=0,
                      class_name="person", confidence=0.9),
            Detection(bbox=[100, 100, 150, 150], class_id=0,
                      class_name="person", confidence=0.8),
            Detection(bbox=[200, 200, 300, 300], class_id=2,
                      class_name="car", confidence=0.7),
        ]
        tracker.update(detections, timestamp=0.0)

        summary = tracker.get_summary()
        assert summary["total_unique_objects"] == 3
        assert summary["class_counts"]["person"] == 2
        assert summary["class_counts"]["car"] == 1

    def test_reset(self):
        """Reset deve limpar todos os tracks."""
        tracker = IoUTracker()
        det = [Detection(bbox=[10, 10, 50, 50], class_id=0,
                         class_name="person", confidence=0.9)]
        tracker.update(det, timestamp=0.0)
        assert len(tracker.get_all_tracks()) == 1

        tracker.reset()
        assert len(tracker.get_all_tracks()) == 0
        assert tracker.next_id == 1

    def test_tracked_object_to_dict(self):
        """Serialização do TrackedObject para dicionário."""
        obj = TrackedObject(
            track_id=1, class_name="person", class_id=0,
            first_seen=0.0, last_seen=2.0,
            bboxes=[[10, 10, 50, 50], [15, 15, 55, 55]],
            timestamps=[0.0, 2.0],
            confidences=[0.9, 0.85],
        )
        d = obj.to_dict()
        assert d["track_id"] == 1
        assert d["class_name"] == "person"
        assert d["duration_seconds"] == 2.0
        assert len(d["bboxes"]) == 2
        assert d["avg_confidence"] == pytest.approx(0.875, abs=1e-3)

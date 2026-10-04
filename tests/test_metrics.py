import pytest
import numpy as np
import torch

from ai_model.metrics import (
    calculate_iou_per_class,
    calculate_mean_iou,
    calculate_dice_score,
    evaluate_segmentation_batch
)


class TestMetrics:
    def test_calculate_iou_perfect_match(self):
        """Birebir eslesen tahminlerde IoU 1.0 olmalidir."""
        targets = np.array([0, 0, 1, 1, 2, 2])
        preds = np.array([0, 0, 1, 1, 2, 2])
        ious = calculate_iou_per_class(preds, targets, num_classes=5)

        assert ious[0] == pytest.approx(1.0)
        assert ious[1] == pytest.approx(1.0)
        assert ious[2] == pytest.approx(1.0)
        assert np.isnan(ious[3])
        assert np.isnan(ious[4])

    def test_calculate_iou_partial_match(self):
        """Kismi kesisen tahminlerde dogru IoU hesaplanmalidir."""
        targets = np.array([1, 1, 1, 1])
        preds = np.array([1, 1, 2, 2])
        ious = calculate_iou_per_class(preds, targets, num_classes=3)

        # Sinif 1 icin: TP=2, FP=0, FN=2 -> IoU = 2 / (2+0+2) = 0.5
        assert ious[1] == pytest.approx(0.5)
        # Sinif 2 icin: TP=0, FP=2, FN=0 -> IoU = 0 / 2 = 0.0
        assert ious[2] == pytest.approx(0.0)

    def test_calculate_mean_iou(self):
        """mIoU ortalamasi dogru siniflar uzerinden alinmalidir."""
        targets = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 1, 2])
        # Sinif 0: TP=2, Union=2 -> IoU=1.0
        # Sinif 1: TP=1, FP=0, FN=1 -> IoU=0.5
        # Sinif 2: TP=0, FP=1, FN=0 -> IoU=0.0
        # Ortalama = (1.0 + 0.5 + 0.0) / 3 = 0.5

        miou = calculate_mean_iou(preds, targets, num_classes=3, ignore_background=False)
        assert miou == pytest.approx(0.5)

        # ignore_background=True durumunda sinif 0 haric:
        # Sinif 1 (0.5) + Sinif 2 (0.0) -> Ortalama = 0.25
        miou_no_bg = calculate_mean_iou(preds, targets, num_classes=3, ignore_background=True)
        assert miou_no_bg == pytest.approx(0.25)

    def test_calculate_mean_iou_empty(self):
        """Hicbir sinif bulunmadiginda 0.0 donmelidir."""
        targets = np.array([])
        preds = np.array([])
        miou = calculate_mean_iou(preds, targets, num_classes=5)
        assert miou == 0.0

    def test_calculate_dice_score(self):
        """Dice katsayisi (F1-score) hesaplama dogrulamasi."""
        targets = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 1, 1])
        dice = calculate_dice_score(preds, targets, num_classes=2)
        assert dice == pytest.approx(1.0, abs=1e-4)

        # Tamamen ayrik
        preds_wrong = np.array([1, 1, 0, 0])
        dice_wrong = calculate_dice_score(preds_wrong, targets, num_classes=2)
        assert dice_wrong == pytest.approx(0.0, abs=1e-4)

    def test_evaluate_segmentation_batch_tensor(self):
        """PyTorch tensörleri ile toplu metrik degerlendirme."""
        targets = torch.tensor([[0, 1, 2], [1, 2, 0]])
        preds = torch.tensor([[0, 1, 2], [1, 2, 0]])

        res = evaluate_segmentation_batch(preds, targets, num_classes=3)
        assert isinstance(res, dict)
        assert "accuracy" in res
        assert "mIoU" in res
        assert "dice" in res
        assert res["accuracy"] == pytest.approx(1.0)
        assert res["mIoU"] == pytest.approx(1.0)
        assert res["dice"] == pytest.approx(1.0, abs=1e-4)

    def test_evaluate_segmentation_batch_numpy(self):
        """NumPy dizileri ile toplu metrik degerlendirme."""
        targets = np.array([0, 1, 2, 0])
        preds = np.array([0, 1, 0, 0])

        res = evaluate_segmentation_batch(preds, targets, num_classes=3)
        assert res["accuracy"] == pytest.approx(0.75)
        assert 0.0 <= res["mIoU"] <= 1.0
        assert 0.0 <= res["dice"] <= 1.0

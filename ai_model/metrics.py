import torch
import numpy as np
from typing import Dict, Any, Optional
from ai_model.dataset import FDI_CLASSES, IDX_TO_LABEL


def calculate_iou_per_class(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 33
) -> np.ndarray:
    """
    Her sınıf için Kesişim / Birleşim (Intersection over Union - IoU) skorunu hesaplar.

    IoU = Intersection / Union = TP / (TP + FP + FN)

    Args:
        preds (np.ndarray): Model tahmin sınıf indeksleri (N,).
        targets (np.ndarray): Gerçek referans (ground truth) sınıf indeksleri (N,).
        num_classes (int): Toplam sınıf sayısı (varsayılan 33).

    Returns:
        np.ndarray: Her sınıfın IoU değeri (mevcut olmayan sınıflar için np.nan).
    """
    preds = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    ious = np.full(num_classes, np.nan, dtype=np.float64)

    for cls_idx in range(num_classes):
        pred_mask = (preds == cls_idx)
        target_mask = (targets == cls_idx)

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union > 0:
            ious[cls_idx] = float(intersection / union)

    return ious


def calculate_mean_iou(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 33,
    ignore_background: bool = False
) -> float:
    """
    Mevcut sınıflar üzerinden ortalama IoU (mIoU) hesaplar.
    """
    ious = calculate_iou_per_class(preds, targets, num_classes=num_classes)
    if ignore_background and len(ious) > 1:
        ious = ious[1:]  # Sınıf 0 (Diş eti) hariç tut

    valid_ious = ious[~np.isnan(ious)]
    if len(valid_ious) == 0:
        return 0.0
    return float(np.mean(valid_ious))


def calculate_dice_score(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 33,
    eps: float = 1e-7
) -> float:
    """
    Makro ortalama Sørensen-Dice Katsayısı (F1-Skoru) hesaplar.
    Dice = 2 * TP / (2 * TP + FP + FN)
    """
    preds = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    dices = []
    for cls_idx in range(num_classes):
        pred_mask = (preds == cls_idx)
        target_mask = (targets == cls_idx)

        tp = np.logical_and(pred_mask, target_mask).sum()
        fp = np.logical_and(pred_mask, ~target_mask).sum()
        fn = np.logical_and(~pred_mask, target_mask).sum()

        denom = 2 * tp + fp + fn
        if (target_mask.sum() + pred_mask.sum()) > 0:
            dice = (2.0 * tp + eps) / (denom + eps)
            dices.append(dice)

    if not dices:
        return 0.0
    return float(np.mean(dices))


def evaluate_segmentation_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 33
) -> Dict[str, Any]:
    """
    Tensörler üzerinden Accuracy, mIoU ve Dice metriklerini hesaplar.
    """
    if torch.is_tensor(preds):
        preds_np = preds.detach().cpu().numpy()
    else:
        preds_np = np.asarray(preds)

    if torch.is_tensor(targets):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.asarray(targets)

    # Düzleştir
    p_flat = preds_np.reshape(-1)
    t_flat = targets_np.reshape(-1)

    correct = (p_flat == t_flat).sum()
    total = len(t_flat)
    accuracy = float(correct / total) if total > 0 else 0.0

    miou = calculate_mean_iou(p_flat, t_flat, num_classes=num_classes)
    dice = calculate_dice_score(p_flat, t_flat, num_classes=num_classes)

    return {
        "accuracy": accuracy,
        "mIoU": miou,
        "dice": dice
    }

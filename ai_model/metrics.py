import torch
import numpy as np
from typing import Dict, Any


def calculate_iou_per_class(
    tahminler: np.ndarray = None,
    hedefler: np.ndarray = None,
    sinif_sayisi: int = 33,
    **ek_parametreler
) -> np.ndarray:
    """
    Her sinif icin Kesisim / Birlesim (IoU) skorunu hesaplar.

    IoU = Kesisim / Birlesim = DP / (DP + YP + YN)
    (DP: Dogru Pozitif, YP: Yanlis Pozitif, YN: Yanlis Negatif)

    Args:
        tahminler (np.ndarray): Model tahmin sinif indeksleri (N,).
        hedefler (np.ndarray): Gercek referans (ground truth) sinif indeksleri (N,).
        sinif_sayisi (int): Toplam sinif sayisi (varsayilan 33).

    Returns:
        np.ndarray: Her sinifin IoU degeri (mevcut olmayan siniflar icin np.nan).
    """
    tahminler = ek_parametreler.get("preds", tahminler)
    hedefler = ek_parametreler.get("targets", hedefler)
    sinif_sayisi = ek_parametreler.get("num_classes", sinif_sayisi)

    tahminler = np.asarray(tahminler).reshape(-1)
    hedefler = np.asarray(hedefler).reshape(-1)

    iou_degerleri = np.full(sinif_sayisi, np.nan, dtype=np.float64)

    for sinif_indeks in range(sinif_sayisi):
        tahmin_maskesi = (tahminler == sinif_indeks)
        hedef_maskesi = (hedefler == sinif_indeks)

        kesisim = np.logical_and(tahmin_maskesi, hedef_maskesi).sum()
        birlesim = np.logical_or(tahmin_maskesi, hedef_maskesi).sum()

        if birlesim > 0:
            iou_degerleri[sinif_indeks] = float(kesisim / birlesim)

    return iou_degerleri


def calculate_mean_iou(
    tahminler: np.ndarray = None,
    hedefler: np.ndarray = None,
    sinif_sayisi: int = 33,
    arka_plani_yoksay: bool = False,
    **ek_parametreler
) -> float:
    """
    Mevcut siniflar uzerinden ortalama IoU (mIoU) hesaplar.
    """
    tahminler = ek_parametreler.get("preds", tahminler)
    hedefler = ek_parametreler.get("targets", hedefler)
    sinif_sayisi = ek_parametreler.get("num_classes", sinif_sayisi)
    arka_plani_yoksay = ek_parametreler.get("ignore_background", arka_plani_yoksay)

    iou_degerleri = calculate_iou_per_class(tahminler, hedefler, sinif_sayisi=sinif_sayisi)
    if arka_plani_yoksay and len(iou_degerleri) > 1:
        iou_degerleri = iou_degerleri[1:]  # Sinif 0 (Dis eti) haric tut

    gecerli_iou = iou_degerleri[~np.isnan(iou_degerleri)]
    if len(gecerli_iou) == 0:
        return 0.0
    return float(np.mean(gecerli_iou))


def calculate_dice_score(
    tahminler: np.ndarray = None,
    hedefler: np.ndarray = None,
    sinif_sayisi: int = 33,
    epsilon: float = 1e-7,
    **ek_parametreler
) -> float:
    """
    Makro ortalama Sorensen-Dice Katsayisi (F1-Skoru) hesaplar.
    Dice = 2 * DP / (2 * DP + YP + YN)
    """
    tahminler = ek_parametreler.get("preds", tahminler)
    hedefler = ek_parametreler.get("targets", hedefler)
    sinif_sayisi = ek_parametreler.get("num_classes", sinif_sayisi)
    epsilon = ek_parametreler.get("epsilon", epsilon)

    tahminler = np.asarray(tahminler).reshape(-1)
    hedefler = np.asarray(hedefler).reshape(-1)

    dice_degerleri = []
    for sinif_indeks in range(sinif_sayisi):
        tahmin_maskesi = (tahminler == sinif_indeks)
        hedef_maskesi = (hedefler == sinif_indeks)

        dogru_pozitif = np.logical_and(tahmin_maskesi, hedef_maskesi).sum()
        yanlis_pozitif = np.logical_and(tahmin_maskesi, ~hedef_maskesi).sum()
        yanlis_negatif = np.logical_and(~tahmin_maskesi, hedef_maskesi).sum()

        payda = 2 * dogru_pozitif + yanlis_pozitif + yanlis_negatif
        if (hedef_maskesi.sum() + tahmin_maskesi.sum()) > 0:
            dice = (2.0 * dogru_pozitif + epsilon) / (payda + epsilon)
            dice_degerleri.append(dice)

    if not dice_degerleri:
        return 0.0
    return float(np.mean(dice_degerleri))


def evaluate_segmentation_batch(
    tahminler: torch.Tensor = None,
    hedefler: torch.Tensor = None,
    sinif_sayisi: int = 33,
    **ek_parametreler
) -> Dict[str, Any]:
    """
    Tensorler uzerinden Dogruluk (accuracy), mIoU ve Dice metriklerini hesaplar.
    """
    tahminler = ek_parametreler.get("preds", tahminler)
    hedefler = ek_parametreler.get("targets", hedefler)
    sinif_sayisi = ek_parametreler.get("num_classes", sinif_sayisi)
    if torch.is_tensor(tahminler):
        tahminler_np = tahminler.detach().cpu().numpy()
    else:
        tahminler_np = np.asarray(tahminler)

    if torch.is_tensor(hedefler):
        hedefler_np = hedefler.detach().cpu().numpy()
    else:
        hedefler_np = np.asarray(hedefler)

    # Duzlestir
    duz_tahmin = tahminler_np.reshape(-1)
    duz_hedef = hedefler_np.reshape(-1)

    dogru_sayisi = (duz_tahmin == duz_hedef).sum()
    toplam = len(duz_hedef)
    dogruluk = float(dogru_sayisi / toplam) if toplam > 0 else 0.0

    miou = calculate_mean_iou(duz_tahmin, duz_hedef, sinif_sayisi=sinif_sayisi)
    dice = calculate_dice_score(duz_tahmin, duz_hedef, sinif_sayisi=sinif_sayisi)

    return {
        "accuracy": dogruluk,
        "mIoU": miou,
        "dice": dice
    }

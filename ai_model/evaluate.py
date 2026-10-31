"""
Test seti degerlendirmesi: kose bazli (tam cozunurluk) segmentasyon metrikleri.

Uretilen ciktilar (cikti dizini altinda):
  test_metrics.json      -> genel metrikler + dis bazinda IoU tablosu
  per_tooth_iou.csv      -> rapora/Excel'e dogrudan alinabilir tablo
  per_tooth_iou.png      -> dis bazinda IoU grafigi
  confusion_matrix.npy   -> 33x33 karisiklik matrisi
  samples/*.ply          -> en iyi / en kotu vakalarin renkli tahmin + gercek mesh'leri

Kullanim:
  python ai_model/evaluate.py --checkpoint runs/dgcnn/best.pth --split runs/dgcnn/split.json
"""
import os
import sys
import json
import time
import numpy as np
import torch
import open3d as o3d

KOK_DIZIN = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK_DIZIN not in sys.path:
    sys.path.insert(0, KOK_DIZIN)

from ai_model.dataset import FDI_SINIFLARI, bolumu_yukle, mesh_oku, hasta_kimligi, mesh_geometri_oku, o3d_mesh_olustur
from ai_model.network import checkpoint_yukle
from ai_model.inference import koseleri_tahmin_et
from ui.viewer import get_fdi_color_palette
from utils.logger import get_logger
from utils.config import load_config, get_base_parser
from utils.validator import get_safe_path

kayitci = get_logger(__name__)


def karisiklik_matrisi(tahmin, hedef, sinif_sayisi=33):
    return np.bincount(hedef * sinif_sayisi + tahmin, minlength=sinif_sayisi ** 2).reshape(sinif_sayisi, sinif_sayisi)


def matristen_metrikler(km):
    """Karisiklik matrisinden IoU / Dice / dogruluk hesaplar."""
    dp = np.diag(km).astype(np.float64)
    yp = km.sum(axis=0) - dp
    yn = km.sum(axis=1) - dp
    birlesim = dp + yp + yn
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(birlesim > 0, dp / birlesim, np.nan)
        dice = np.where(birlesim > 0, 2 * dp / (2 * dp + yp + yn), np.nan)
    mevcut = km.sum(axis=1) > 0   # test setinde gercekten bulunan siniflar
    return {
        "accuracy": float(dp.sum() / max(km.sum(), 1)),
        "mIoU": float(np.nanmean(np.where(mevcut, iou, np.nan))),
        "mIoU_teeth": float(np.nanmean(np.where(mevcut, iou, np.nan)[1:])),
        "dice": float(np.nanmean(np.where(mevcut, dice, np.nan))),
        "iou_per_class": iou,
        "support": km.sum(axis=1),
    }


def _renkli_mesh_kaydet(obj_yolu, sinif_idx, cikti_yolu):
    palet = get_fdi_color_palette()
    v, f = mesh_geometri_oku(obj_yolu)
    mesh = o3d_mesh_olustur(v, f, palet[np.clip(sinif_idx, 0, len(palet) - 1)])
    o3d.io.write_triangle_mesh(cikti_yolu, mesh, write_vertex_colors=True)


def degerlendir(model, ornekler, nokta_sayisi, cihaz, cikti_dizini, tur_sayisi=3,
                sinif_sayisi=33, ornek_kaydet=3, maks_ornek=None):
    """Model'i verilen ornekler uzerinde tam cozunurlukte degerlendirir ve dosyalara yazar."""
    os.makedirs(cikti_dizini, exist_ok=True)
    km = np.zeros((sinif_sayisi, sinif_sayisi), dtype=np.int64)
    vaka_sonuclari = []
    dis_dogru, dis_toplam = 0, 0
    if maks_ornek:
        ornekler = ornekler[:maks_ornek]

    t0 = time.time()
    for i, (obj, js) in enumerate(ornekler):
        try:
            v, n, hedef, hasta = mesh_oku(obj, js)
        except Exception as hata:
            kayitci.warning(f"Atlandi ({obj}): {hata}")
            continue
        tahmin, _ = koseleri_tahmin_et(model, v, n, nokta_sayisi=nokta_sayisi, cihaz=cihaz, tur_sayisi=tur_sayisi)
        vaka_km = karisiklik_matrisi(tahmin, hedef, sinif_sayisi)
        km += vaka_km
        m = matristen_metrikler(vaka_km)

        # Dis tanima: gercekte var olan her dis icin IoU > 0.5 ise "dogru tanindi"
        for c in np.unique(hedef):
            if c == 0:
                continue
            dis_toplam += 1
            if m["iou_per_class"][c] > 0.5:
                dis_dogru += 1

        vaka_sonuclari.append({"obj": obj, "patient": hasta, "accuracy": m["accuracy"],
                               "mIoU": m["mIoU"], "mIoU_teeth": m["mIoU_teeth"], "_tahmin": tahmin, "_hedef": hedef})
        if (i + 1) % 10 == 0:
            kayitci.info(f"Degerlendirme: {i + 1}/{len(ornekler)} vaka ({time.time() - t0:.0f} sn)")

    if not vaka_sonuclari:
        raise RuntimeError("Degerlendirilecek gecerli vaka bulunamadi.")

    genel = matristen_metrikler(km)
    vaka_miou = np.array([v["mIoU_teeth"] for v in vaka_sonuclari])

    # En iyi / en kotu vakalari gorsel olarak kaydet
    ornek_dizin = os.path.join(cikti_dizini, "samples")
    os.makedirs(ornek_dizin, exist_ok=True)
    sirali = sorted(vaka_sonuclari, key=lambda v: v["mIoU_teeth"])
    kotu = sirali[:ornek_kaydet]
    iyi = [v for v in sirali[-ornek_kaydet:] if all(v is not k for k in kotu)]
    secilenler = [("worst", v) for v in kotu] + [("best", v) for v in iyi]
    for etiket, v in secilenler:
        ad = f"{etiket}_{os.path.splitext(os.path.basename(v['obj']))[0]}_miou{v['mIoU_teeth']:.2f}"
        try:
            _renkli_mesh_kaydet(v["obj"], v["_tahmin"], os.path.join(ornek_dizin, ad + "_pred.ply"))
            _renkli_mesh_kaydet(v["obj"], v["_hedef"], os.path.join(ornek_dizin, ad + "_gt.ply"))
        except Exception as hata:
            kayitci.warning(f"Ornek mesh kaydedilemedi: {hata}")

    tablo = []
    for c in range(sinif_sayisi):
        if genel["support"][c] == 0:
            continue
        tablo.append({"fdi": FDI_SINIFLARI[c], "name": "gingiva" if c == 0 else f"tooth_{FDI_SINIFLARI[c]}",
                      "iou": float(genel["iou_per_class"][c]), "vertices": int(genel["support"][c])})

    sonuc = {
        "num_cases": len(vaka_sonuclari),
        "num_patients": len({v["patient"] for v in vaka_sonuclari}),
        "accuracy": genel["accuracy"],
        "mIoU_all": genel["mIoU"],
        "mIoU_teeth": genel["mIoU_teeth"],
        "dice": genel["dice"],
        "case_mIoU_teeth_mean": float(np.nanmean(vaka_miou)),
        "case_mIoU_teeth_std": float(np.nanstd(vaka_miou)),
        "tooth_identification_rate_iou50": dis_dogru / max(dis_toplam, 1),
        "per_class": tablo,
        "cases": [{k: v[k] for k in ("obj", "patient", "accuracy", "mIoU", "mIoU_teeth")} for v in vaka_sonuclari],
    }

    with open(os.path.join(cikti_dizini, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, indent=2)
    np.save(os.path.join(cikti_dizini, "confusion_matrix.npy"), km)
    with open(os.path.join(cikti_dizini, "per_tooth_iou.csv"), "w", encoding="utf-8") as f:
        f.write("fdi,name,iou,vertices\n")
        for r in tablo:
            f.write(f"{r['fdi']},{r['name']},{r['iou']:.4f},{r['vertices']}\n")
    _iou_grafigi(tablo, os.path.join(cikti_dizini, "per_tooth_iou.png"), sonuc)

    kayitci.info(
        f"TEST SONUCU -> Vaka: {sonuc['num_cases']} | Dogruluk: %{sonuc['accuracy'] * 100:.2f} | "
        f"mIoU (disler): %{sonuc['mIoU_teeth'] * 100:.2f} | mIoU (tum): %{sonuc['mIoU_all'] * 100:.2f} | "
        f"Dice: %{sonuc['dice'] * 100:.2f} | Dis tanima (IoU>0.5): %{sonuc['tooth_identification_rate_iou50'] * 100:.1f}"
    )
    return sonuc


def _iou_grafigi(tablo, yol, sonuc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    etiketler = ["Dis eti" if r["fdi"] == 0 else str(r["fdi"]) for r in tablo]
    degerler = [r["iou"] for r in tablo]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(range(len(degerler)), degerler, color="#3b6ea5")
    ax.axhline(sonuc["mIoU_teeth"], color="#c0392b", linestyle="--", linewidth=1,
               label=f"mIoU (disler) = {sonuc['mIoU_teeth']:.3f}")
    ax.set_xticks(range(len(etiketler)))
    ax.set_xticklabels(etiketler, rotation=90)
    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU")
    ax.set_xlabel("FDI dis numarasi")
    ax.set_title(f"Dis bazinda IoU - test seti ({sonuc['num_cases']} vaka, {sonuc['num_patients']} hasta)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(yol, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = get_base_parser(description="Dis segmentasyon modeli test degerlendirmesi")
    p.add_argument("--checkpoint", required=True, help="Model agirligi (.pth)")
    p.add_argument("--split", default=None, help="split.json (egitimde uretilen)")
    p.add_argument("--subset", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--model", default="pointnet", help="Eski (salt state_dict) checkpoint'ler icin mimari")
    p.add_argument("--num-points", type=int, default=None)
    p.add_argument("--votes", type=int, default=3)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--out", default=None, help="Cikti dizini (varsayilan: checkpoint klasoru/eval_<subset>)")
    a = p.parse_args()

    cfg = load_config(a.config)
    veri_koku = a.data_root or cfg.ai_model.data_root
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, kayit = checkpoint_yukle(a.checkpoint, cihaz, varsayilan_model=a.model)
    nokta = a.num_points or int(kayit.get("num_points", cfg.ai_model.num_points))

    split_yolu = a.split or os.path.join(os.path.dirname(a.checkpoint), "split.json")
    ornekler = bolumu_yukle(split_yolu, veri_koku)[a.subset]
    cikti = a.out or os.path.join(os.path.dirname(a.checkpoint), f"eval_{a.subset}")
    degerlendir(model, ornekler, nokta, cihaz, cikti, tur_sayisi=a.votes, maks_ornek=a.max_cases)

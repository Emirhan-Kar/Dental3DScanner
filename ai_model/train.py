import os
import sys
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Proje kok dizinini sys.path'e ekle
KOK_DIZIN = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK_DIZIN not in sys.path:
    sys.path.insert(0, KOK_DIZIN)

from ai_model.dataset import (DentalMeshDataset, ornekleri_tara, hasta_bazli_bol, bolumu_kaydet,
                              bolumu_yukle, sinif_agirliklari_hesapla, hasta_kimligi)
from ai_model.network import model_olustur, DentalPointNetSeg
from ai_model.evaluate import karisiklik_matrisi, matristen_metrikler, degerlendir
from utils.logger import get_logger, setup_logger
from utils.config import load_config, get_base_parser

kayitci = get_logger(__name__)


def _isci_baslat(isci_id):
    """Her DataLoader iscisine farkli numpy tohumu ver (ayni veri artirmasini tekrarlamasin)."""
    np.random.seed((torch.initial_seed() + isci_id) % 2 ** 32)


def _gecmis_yaz(gecmis, cikti_dizini):
    with open(os.path.join(cikti_dizini, "history.json"), "w", encoding="utf-8") as f:
        json.dump(gecmis, f, indent=1)
    if gecmis:
        with open(os.path.join(cikti_dizini, "history.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(gecmis[0].keys()))
            w.writeheader()
            w.writerows(gecmis)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [g["epoch"] for g in gecmis]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.plot(ep, [g["train_loss"] for g in gecmis], label="Egitim")
        a1.plot(ep, [g["val_loss"] for g in gecmis], label="Dogrulama")
        a1.set_xlabel("Epoch"); a1.set_ylabel("Kayip"); a1.legend(); a1.set_title("Kayip")
        a2.plot(ep, [g["val_miou_teeth"] for g in gecmis], label="Dogrulama mIoU (disler)")
        a2.plot(ep, [g["val_acc"] for g in gecmis], label="Dogrulama dogruluk")
        a2.set_xlabel("Epoch"); a2.set_ylim(0, 1); a2.legend(); a2.set_title("Dogrulama metrikleri")
        for a in (a1, a2):
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(cikti_dizini, "learning_curves.png"), dpi=150)
        plt.close(fig)
    except Exception:
        pass


def train(
    konfig=None,
    sinif_sayisi=None,
    nokta_sayisi=None,
    batch_boyutu=None,
    epoch_sayisi=None,
    ogrenme_orani=None,
    checkpoint_yolu=None,
    veri_parcalari=None,
    lr_adim_araligi=None,
    lr_gamma=None,
    devam_yolu=None,
    model_adi=None,
    cikti_dizini=None,
    onbellek_dizini=None,
    isci_sayisi=None,
    dogrulama_orani=0.1,
    test_orani=0.1,
    tohum=42,
    sinif_agirligi=True,
    zamanlayici_tipi="cosine",
    test_sonunda=True,
    maks_test_vaka=None,
    veri_koku=None,
):
    """
    3D Dis Segmentasyon Agi Egitimi (PointNet / DGCNN).

    Onemli farklar (onceki surume gore):
      * HASTA bazli egitim/dogrulama/test bolmesi (split.json olarak kaydedilir).
      * Veri artirma SADECE egitim setinde; dogrulama sabit orneklemeli.
      * Sinif agirlikli CrossEntropy (dis eti baskinligina karsi).
      * Her epoch 'last.pth' (Colab kopmasina karsi devam), en iyi mIoU'da 'best.pth'.
      * history.csv / learning_curves.png her epoch guncellenir.
      * Egitim sonunda en iyi model test setinde TAM cozunurlukte degerlendirilir.

    Returns:
        bool veya dict: Basarisizsa False, basariliysa test metrikleri (ya da True).
    """
    if konfig is None:
        konfig = load_config()
    ai = konfig.ai_model

    sinif_sayisi = sinif_sayisi or ai.num_classes
    nokta_sayisi = nokta_sayisi or ai.num_points
    batch_boyutu = batch_boyutu or ai.batch_size
    epoch_sayisi = epoch_sayisi or ai.epochs
    ogrenme_orani = ogrenme_orani or ai.learning_rate
    veri_parcalari = veri_parcalari or ai.parts
    lr_adim_araligi = lr_adim_araligi or getattr(ai, "lr_step_size", 10)
    lr_gamma = lr_gamma or getattr(ai, "lr_gamma", 0.5)
    model_adi = (model_adi or getattr(ai, "model", "pointnet")).lower()
    veri_koku = veri_koku or ai.data_root
    cikti_dizini = cikti_dizini or os.path.join("runs", model_adi)
    isci_sayisi = isci_sayisi if isci_sayisi is not None else getattr(ai, "num_workers", 0)
    giris_kanallari = getattr(ai, "in_channels", 6)

    if min(sinif_sayisi, nokta_sayisi, batch_boyutu, epoch_sayisi) <= 0 or ogrenme_orani <= 0:
        raise ValueError(
            "Tum hiperparametreler (sinif_sayisi, nokta_sayisi, batch_boyutu, epoch_sayisi, ogrenme_orani) "
            "pozitif olmalidir."
        )

    os.makedirs(cikti_dizini, exist_ok=True)
    en_iyi_yol = checkpoint_yolu or os.path.join(cikti_dizini, "best.pth")
    son_yol = os.path.join(cikti_dizini, "last.pth")

    torch.manual_seed(tohum)
    np.random.seed(tohum)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = cihaz.type == "cuda" and torch.cuda.is_bf16_supported()
    if cihaz.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        kayitci.info(f"GPU: {torch.cuda.get_device_name(0)} | bf16 karisik hassasiyet: {bf16}")

    kayitci.info(
        f"Egitim -> Model: {model_adi}, Epoch: {epoch_sayisi}, Batch: {batch_boyutu}, LR: {ogrenme_orani}, "
        f"Nokta: {nokta_sayisi}, Parcalar: {veri_parcalari}, Cikti: {cikti_dizini}"
    )

    # 1. Hasta bazli bolme (varsa ayni bolme yeniden kullanilir -> devam/karsilastirma tutarli kalir)
    split_yolu = os.path.join(cikti_dizini, "split.json")
    if os.path.exists(split_yolu):
        bolum = bolumu_yukle(split_yolu, veri_koku)
        kayitci.info(f"Mevcut bolme kullaniliyor: {split_yolu}")
    else:
        if not os.path.exists(veri_koku):
            kayitci.error(f"Veri seti kok dizini bulunamadi: {veri_koku}")
            return False
        tum = ornekleri_tara(veri_koku, veri_parcalari)
        if not tum:
            kayitci.error(f"Belirtilen parcalar ({veri_parcalari}) altinda gecerli vaka bulunamadi!")
            return False
        bolum = hasta_bazli_bol(tum, dogrulama_orani, test_orani, tohum)
        bolumu_kaydet(bolum, split_yolu, veri_koku)
    if not bolum["val"]:
        bolum["val"] = bolum["train"][:1]

    for ad in ("train", "val", "test"):
        kayitci.info(f"  {ad:5s}: {len(bolum[ad])} vaka, {len({hasta_kimligi(o) for o, _ in bolum[ad]})} hasta")

    egitim_seti = DentalMeshDataset(ornek_listesi=bolum["train"], nokta_sayisi=nokta_sayisi,
                                    veri_artirma=True, onbellek_dizini=onbellek_dizini)
    dogrulama_seti = DentalMeshDataset(ornek_listesi=bolum["val"], nokta_sayisi=nokta_sayisi,
                                       veri_artirma=False, sabit_ornekleme=True, onbellek_dizini=onbellek_dizini)
    if len(egitim_seti) == 0:
        kayitci.error("Egitim seti bos!")
        return False

    yukleyici_ayar = dict(num_workers=isci_sayisi, pin_memory=cihaz.type == "cuda",
                          worker_init_fn=_isci_baslat, persistent_workers=isci_sayisi > 0)
    egitim_yukleyici = DataLoader(egitim_seti, batch_size=batch_boyutu, shuffle=True,
                                  drop_last=len(egitim_seti) > batch_boyutu, **yukleyici_ayar)
    dogrulama_yukleyici = DataLoader(dogrulama_seti, batch_size=batch_boyutu, shuffle=False, **yukleyici_ayar)

    # 2. Model / kayip / optimizasyon
    model = model_olustur(model_adi, giris_kanallari, sinif_sayisi).to(cihaz)
    kayitci.info(f"Parametre sayisi: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    agirliklar = None
    if sinif_agirligi:
        agirliklar = sinif_agirliklari_hesapla(bolum["train"], sinif_sayisi)
        kayitci.info(f"Sinif agirliklari (dis eti={agirliklar[0]:.2f}, dis ort.={agirliklar[1:].mean():.2f})")
        agirliklar = torch.tensor(agirliklar, device=cihaz)
    kayip_fonksiyonu = nn.CrossEntropyLoss(weight=agirliklar)

    optimizasyon = torch.optim.AdamW(model.parameters(), lr=ogrenme_orani, weight_decay=1e-4)
    if zamanlayici_tipi == "step":
        zamanlayici = torch.optim.lr_scheduler.StepLR(optimizasyon, step_size=lr_adim_araligi, gamma=lr_gamma)
    else:
        zamanlayici = torch.optim.lr_scheduler.CosineAnnealingLR(optimizasyon, T_max=epoch_sayisi,
                                                                 eta_min=ogrenme_orani * 0.01)

    baslangic_epoch, en_iyi_miou, gecmis = 1, -1.0, []

    def kayit_sozlugu(epoch, ek=None):
        d = {
            "epoch": epoch, "model_name": model_adi, "in_channels": giris_kanallari,
            "num_classes": sinif_sayisi, "num_points": nokta_sayisi,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizasyon.state_dict(),
            "scheduler_state_dict": zamanlayici.state_dict(),
            "best_val_miou": en_iyi_miou, "history": gecmis,
        }
        d.update(ek or {})
        return d

    # 3. Devam (resume)
    if devam_yolu:
        if not os.path.exists(devam_yolu):
            kayitci.error(f"Devam edilecek checkpoint bulunamadi: {devam_yolu}")
            return False
        try:
            kayit = torch.load(devam_yolu, map_location=cihaz, weights_only=False)
            if isinstance(kayit, dict) and "model_state_dict" in kayit:
                model.load_state_dict(kayit["model_state_dict"])
                if "optimizer_state_dict" in kayit:
                    optimizasyon.load_state_dict(kayit["optimizer_state_dict"])
                if "scheduler_state_dict" in kayit:
                    zamanlayici.load_state_dict(kayit["scheduler_state_dict"])
                baslangic_epoch = kayit.get("epoch", 0) + 1
                en_iyi_miou = kayit.get("best_val_miou", -1.0)
                gecmis = kayit.get("history", [])
            else:
                model.load_state_dict(kayit)
            kayitci.info(f"{devam_yolu} yuklendi, {baslangic_epoch}. epoch'tan devam ediliyor.")
        except Exception as hata:
            kayitci.error(f"Checkpoint yuklenemedi: {hata}")
            return False

    # 4. Egitim dongusu
    epoch = baslangic_epoch - 1
    try:
        for epoch in range(baslangic_epoch, epoch_sayisi + 1):
            t0 = time.time()
            model.train()
            top_kayip, batch_sayaci, dogru, toplam, hata_sayaci = 0.0, 0, 0, 0, 0
            cubuk = tqdm(egitim_yukleyici, desc=f"Epoch [{epoch}/{epoch_sayisi}]", leave=False)
            for batch in cubuk:
                try:
                    x = batch["points"].to(cihaz, non_blocking=True)
                    y = batch["labels"].to(cihaz, non_blocking=True)
                    optimizasyon.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=cihaz.type, dtype=torch.bfloat16, enabled=bf16):
                        skor = model(x)
                        kayip = kayip_fonksiyonu(skor.float(), y)
                    kayip.backward()
                    optimizasyon.step()

                    top_kayip += kayip.item()
                    batch_sayaci += 1
                    dogru += (skor.argmax(1) == y).sum().item()
                    toplam += y.numel()
                    cubuk.set_postfix({"kayip": f"{kayip.item():.3f}"})
                except torch.cuda.OutOfMemoryError:
                    kayitci.warning("GPU bellegi yetmedi (OOM)! batch_size veya num_points dusurun.")
                    torch.cuda.empty_cache()
                    hata_sayaci += 1
                except Exception as batch_hatasi:
                    kayitci.error(f"Batch islenirken hata: {batch_hatasi}")
                    hata_sayaci += 1
                if hata_sayaci > 20:
                    raise RuntimeError("Cok fazla batch hatasi; egitim durduruldu (yukaridaki loglara bakin).")
            zamanlayici.step()

            # Dogrulama
            model.eval()
            v_kayip, v_batch = 0.0, 0
            km = np.zeros((sinif_sayisi, sinif_sayisi), dtype=np.int64)
            with torch.no_grad():
                for batch in dogrulama_yukleyici:
                    x = batch["points"].to(cihaz, non_blocking=True)
                    y = batch["labels"].to(cihaz, non_blocking=True)
                    with torch.autocast(device_type=cihaz.type, dtype=torch.bfloat16, enabled=bf16):
                        skor = model(x).float()
                    v_kayip += kayip_fonksiyonu(skor, y).item()
                    v_batch += 1
                    km += karisiklik_matrisi(skor.argmax(1).cpu().numpy().ravel(),
                                             y.cpu().numpy().ravel(), sinif_sayisi)
            m = matristen_metrikler(km)

            kayit_satiri = {
                "epoch": epoch,
                "lr": optimizasyon.param_groups[0]["lr"],
                "train_loss": top_kayip / max(batch_sayaci, 1),
                "train_acc": dogru / max(toplam, 1),
                "val_loss": v_kayip / max(v_batch, 1),
                "val_acc": m["accuracy"],
                "val_miou": m["mIoU"],
                "val_miou_teeth": m["mIoU_teeth"],
                "val_dice": m["dice"],
                "seconds": round(time.time() - t0, 1),
            }
            gecmis.append(kayit_satiri)
            kayitci.info(
                f"[Epoch {epoch}/{epoch_sayisi}] kayip {kayit_satiri['train_loss']:.4f} / {kayit_satiri['val_loss']:.4f} | "
                f"dogrulama: dogruluk %{m['accuracy'] * 100:.2f}, mIoU(dis) %{m['mIoU_teeth'] * 100:.2f}, "
                f"Dice %{m['dice'] * 100:.2f} | {kayit_satiri['seconds']} sn"
            )

            if m["mIoU_teeth"] > en_iyi_miou:
                en_iyi_miou = m["mIoU_teeth"]
                torch.save(kayit_sozlugu(epoch), en_iyi_yol)
                kayitci.info(f"  -> En iyi model kaydedildi ({en_iyi_yol}), mIoU(dis) %{en_iyi_miou * 100:.2f}")
            torch.save(kayit_sozlugu(epoch), son_yol)
            _gecmis_yaz(gecmis, cikti_dizini)

    except KeyboardInterrupt:
        kayitci.warning("Egitim kullanici tarafindan durduruldu; 'last.pth' ile --resume yapabilirsiniz.")
        torch.save(kayit_sozlugu(epoch), son_yol)
        return False
    except Exception as hata:
        kayitci.error(f"Beklenmeyen egitim hatasi: {hata}")
        return False

    # 5. Test seti degerlendirmesi (en iyi model, tam cozunurluk)
    if test_sonunda and bolum["test"] and os.path.exists(en_iyi_yol):
        try:
            kayit = torch.load(en_iyi_yol, map_location=cihaz, weights_only=False)
            model.load_state_dict(kayit["model_state_dict"])
            sonuc = degerlendir(model, bolum["test"], nokta_sayisi, cihaz, os.path.join(cikti_dizini, "eval_test"),
                                maks_ornek=maks_test_vaka)
            sonuc["best_epoch"] = kayit.get("epoch")
            return sonuc
        except Exception as hata:
            kayitci.error(f"Test degerlendirmesi basarisiz: {hata}")
    return True


if __name__ == "__main__":
    p = get_base_parser(description="Dental 3D Dis Segmentasyon Modeli Egitimi")
    p.add_argument("--model", type=str, default=None, choices=["pointnet", "dgcnn"], help="Mimari")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-points", type=int, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--parts", nargs="+", default=None)
    p.add_argument("--out", type=str, default=None, help="Cikti dizini (varsayilan runs/<model>)")
    p.add_argument("--cache-dir", type=str, default=None, help=".npz onbellek dizini (hizli yukleme)")
    p.add_argument("--workers", type=int, default=None, help="DataLoader isci sayisi")
    p.add_argument("--resume", type=str, default=None, help="Devam edilecek checkpoint (genelde <out>/last.pth)")
    p.add_argument("--checkpoint", type=str, default=None, help="En iyi model yolu (varsayilan <out>/best.pth)")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--max-test-cases", type=int, default=None)
    a = p.parse_args()

    yuklu = load_config(a.config)
    setup_logger()
    sonuc = train(
        konfig=yuklu, model_adi=a.model, epoch_sayisi=a.epochs, batch_boyutu=a.batch_size,
        ogrenme_orani=a.lr, nokta_sayisi=a.num_points, veri_parcalari=a.parts, veri_koku=a.data_root,
        cikti_dizini=a.out, onbellek_dizini=a.cache_dir, isci_sayisi=a.workers, devam_yolu=a.resume,
        checkpoint_yolu=a.checkpoint, dogrulama_orani=a.val_ratio, test_orani=a.test_ratio, tohum=a.seed,
        sinif_agirligi=not a.no_class_weights, zamanlayici_tipi=a.scheduler,
        test_sonunda=not a.skip_test, maks_test_vaka=a.max_test_cases,
    )
    sys.exit(0 if sonuc else 1)

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Proje kok dizinini sys.path'e ekle
KOK_DIZIN = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK_DIZIN not in sys.path:
    sys.path.insert(0, KOK_DIZIN)

from ai_model.dataset import DentalMeshDataset
from ai_model.network import DentalPointNetSeg
from ai_model.metrics import evaluate_segmentation_batch
from utils.logger import get_logger
from utils.config import load_config, get_base_parser

kayitci = get_logger(__name__)


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
    devam_yolu=None
):
    """
    3D Dis Nokta Bulutu Segmentasyon Agi Egitim Fonksiyonu.
    Parametreler verilmezse `configs/config.yaml` dosyasindan otomatik yuklenir.

    Args:
        konfig (ConfigNode veya dict, optional): Konfigurasyon nesnesi.
        sinif_sayisi (int, optional): Sinif adedi.
        nokta_sayisi (int, optional): Her ornekten alinacak nokta sayisi.
        batch_boyutu (int, optional): Mini-batch boyutu.
        epoch_sayisi (int, optional): Toplam epoch sayisi.
        ogrenme_orani (float, optional): Ogrenme katsayisi.
        checkpoint_yolu (str, optional): Model agirliklarinin kaydedilecegi yol.
        veri_parcalari (list, optional): Egitime dahil edilecek data_part klasorleri listesi.
        lr_adim_araligi (int, optional): StepLR adim araligi.
        lr_gamma (float, optional): StepLR gamma katsayisi.
        devam_yolu (str, optional): Egitime devam edilecek checkpoint dosyasi.
    """
    # Konfigurasyon nesnesi yoksa varsayilani yukle
    if konfig is None:
        konfig = load_config()

    ai_konfig = konfig.ai_model

    # Parametreleri config ile harmanla (Fonksiyon argumani varsa onceliklidir)
    sinif_sayisi = sinif_sayisi if sinif_sayisi is not None else ai_konfig.num_classes
    nokta_sayisi = nokta_sayisi if nokta_sayisi is not None else ai_konfig.num_points
    batch_boyutu = batch_boyutu if batch_boyutu is not None else ai_konfig.batch_size
    epoch_sayisi = epoch_sayisi if epoch_sayisi is not None else ai_konfig.epochs
    ogrenme_orani = ogrenme_orani if ogrenme_orani is not None else ai_konfig.learning_rate
    checkpoint_yolu = checkpoint_yolu if checkpoint_yolu is not None else ai_konfig.checkpoint_path
    veri_parcalari = veri_parcalari if veri_parcalari is not None else ai_konfig.parts
    lr_adim_araligi = lr_adim_araligi if lr_adim_araligi is not None else getattr(ai_konfig, "lr_step_size", 10)
    lr_gamma = lr_gamma if lr_gamma is not None else getattr(ai_konfig, "lr_gamma", 0.5)

    # 1. Konfigurasyon Dogrulama
    if sinif_sayisi <= 0 or nokta_sayisi <= 0 or batch_boyutu <= 0 or epoch_sayisi <= 0 or ogrenme_orani <= 0:
        raise ValueError(
            "Tum hiperparametreler (sinif_sayisi, nokta_sayisi, batch_boyutu, epoch_sayisi, ogrenme_orani) "
            "pozitif olmalidir."
        )

    kayitci.info(
        f"Egitim baslatiliyor -> Epoch: {epoch_sayisi}, Batch Boyutu: {batch_boyutu}, "
        f"Ogrenme Orani: {ogrenme_orani}, Nokta: {nokta_sayisi}, Parcalar: {veri_parcalari}"
    )

    # Cihaz secimi (GPU varsa CUDA, yoksa CPU)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kayitci.info(f"Egitim Cihazi: {cihaz}")
    if cihaz.type == "cuda":
        try:
            kayitci.info(f"GPU: {torch.cuda.get_device_name(0)}")
            kayitci.info(f"GPU Toplam Bellek: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        except Exception as hata:
            kayitci.warning(f"GPU bilgisi alinirken uyari: {hata}")

    # 2. Veri Seti Yukleme ve Bolme
    try:
        tum_veri_seti = DentalMeshDataset(parcalar=veri_parcalari, nokta_sayisi=nokta_sayisi, veri_artirma=True)
    except Exception as hata:
        kayitci.error(f"Veri seti yuklenirken hata olustu: {hata}")
        return False

    toplam_vaka = len(tum_veri_seti)
    if toplam_vaka == 0:
        kayitci.error(f"Belirtilen parcalar ({veri_parcalari}) altinda gecerli vaka bulunamadi!")
        return False

    # Guvenli egitim / dogrulama bolmesi (En az 1 vaka egitim, 1 vaka dogrulama olmali)
    if toplam_vaka == 1:
        kayitci.warning("Yalnizca 1 vaka mevcut. Hem egitim hem dogrulama icin ayni veri kullanilacak.")
        egitim_seti = tum_veri_seti
        dogrulama_seti = tum_veri_seti
    else:
        egitim_boyutu = max(1, int(0.8 * toplam_vaka))
        dogrulama_boyutu = toplam_vaka - egitim_boyutu
        if dogrulama_boyutu == 0:
            dogrulama_boyutu = 1
            egitim_boyutu = toplam_vaka - 1
        egitim_seti, dogrulama_seti = random_split(tum_veri_seti, [egitim_boyutu, dogrulama_boyutu])

    son_eksik_batch_atilsin = len(egitim_seti) >= batch_boyutu
    egitim_yukleyici = DataLoader(
        egitim_seti, batch_size=batch_boyutu, shuffle=True, drop_last=son_eksik_batch_atilsin
    )
    dogrulama_yukleyici = DataLoader(dogrulama_seti, batch_size=batch_boyutu, shuffle=False)

    kayitci.info(f"Toplam Vaka: {toplam_vaka} | Egitim: {len(egitim_seti)} | Dogrulama: {len(dogrulama_seti)}")

    # 3. Model, Kayip (Loss) Fonksiyonu ve Optimizasyon Algoritmasi
    try:
        giris_kanallari = getattr(ai_konfig, "in_channels", 6)
        model = DentalPointNetSeg(giris_kanallari=giris_kanallari, sinif_sayisi=sinif_sayisi).to(cihaz)
    except Exception as hata:
        kayitci.error(f"Model olusturulamadi: {hata}")
        return False

    kayip_fonksiyonu = nn.CrossEntropyLoss()
    optimizasyon = torch.optim.Adam(model.parameters(), lr=ogrenme_orani)
    zamanlayici = torch.optim.lr_scheduler.StepLR(optimizasyon, step_size=lr_adim_araligi, gamma=lr_gamma)

    en_iyi_dogruluk = 0.0
    en_iyi_miou = 0.0
    baslangic_epoch = 1
    epoch = baslangic_epoch

    # Devam edilecek checkpoint varsa yukle
    # NOT: Checkpoint sozluk anahtarlari ("model_state_dict" vb.) eski kayitlarla uyum icin degistirilmedi.
    if devam_yolu:
        if not os.path.exists(devam_yolu):
            kayitci.error(f"Devam edilecek checkpoint bulunamadi: {devam_yolu}")
            return False
        try:
            kayitci.info(f"Checkpoint yukleniyor: {devam_yolu}")
            kayit = torch.load(devam_yolu, map_location=cihaz)
            if isinstance(kayit, dict) and "model_state_dict" in kayit:
                model.load_state_dict(kayit["model_state_dict"])
                if "optimizer_state_dict" in kayit:
                    optimizasyon.load_state_dict(kayit["optimizer_state_dict"])
                if "scheduler_state_dict" in kayit:
                    zamanlayici.load_state_dict(kayit["scheduler_state_dict"])
                baslangic_epoch = kayit.get("epoch", 0) + 1
                en_iyi_miou = kayit.get("best_val_miou", 0.0)
                en_iyi_dogruluk = kayit.get("best_val_acc", 0.0)
                kayitci.info(
                    f"Egitim {baslangic_epoch}. epoch'tan devam ettiriliyor "
                    f"(Onceki mIoU: %{en_iyi_miou * 100:.2f}, Dogruluk: %{en_iyi_dogruluk * 100:.2f})"
                )
            else:
                model.load_state_dict(kayit)
                kayitci.info("Salt model agirliklari (state_dict) yuklendi.")
        except Exception as kayit_hatasi:
            kayitci.error(f"Checkpoint yuklenemedi: {kayit_hatasi}")
            return False

    if baslangic_epoch > epoch_sayisi:
        kayitci.info(
            f"Mevcut epoch ({baslangic_epoch - 1}) zaten hedeflenen epoch sayisina ({epoch_sayisi}) ulasmis."
        )
        return True

    # 4. Egitim Dongusu
    try:
        for epoch in range(baslangic_epoch, epoch_sayisi + 1):
            model.train()
            toplam_egitim_kaybi = 0.0
            dogru_egitim_noktasi = 0
            toplam_egitim_noktasi = 0
            batch_sayaci = 0

            egitim_cubugu = tqdm(egitim_yukleyici, desc=f"Epoch [{epoch}/{epoch_sayisi}] - Egitim")
            for batch in egitim_cubugu:
                try:
                    noktalar = batch["points"].to(cihaz)     # Boyut: (B, 6, N)
                    etiketler = batch["labels"].to(cihaz)    # Boyut: (B, N)

                    optimizasyon.zero_grad()
                    skorlar = model(noktalar)                # Boyut: (B, SINIF_SAYISI, N)

                    kayip = kayip_fonksiyonu(skorlar, etiketler)
                    kayip.backward()
                    optimizasyon.step()

                    toplam_egitim_kaybi += kayip.item()
                    batch_sayaci += 1

                    tahminler = torch.argmax(skorlar, dim=1)
                    dogru_egitim_noktasi += (tahminler == etiketler).sum().item()
                    toplam_egitim_noktasi += etiketler.numel()

                    egitim_cubugu.set_postfix({"Kayip": f"{kayip.item():.4f}"})

                except torch.cuda.OutOfMemoryError:
                    kayitci.warning("GPU Bellegi Yetmedi (CUDA OutOfMemory)! Bellek temizleniyor...")
                    torch.cuda.empty_cache()
                    continue
                except Exception as batch_hatasi:
                    kayitci.error(f"Batch islenirken hata olustu: {batch_hatasi}")
                    continue

            zamanlayici.step()
            egitim_dogrulugu = (dogru_egitim_noktasi / toplam_egitim_noktasi) if toplam_egitim_noktasi > 0 else 0.0
            ort_egitim_kaybi = (toplam_egitim_kaybi / batch_sayaci) if batch_sayaci > 0 else 0.0

            # 5. Dogrulama (Validation) Dongusu
            model.eval()
            toplam_dogrulama_kaybi = 0.0
            dogru_dogrulama_noktasi = 0
            toplam_dogrulama_noktasi = 0
            dogrulama_batch_sayaci = 0
            tum_dogrulama_tahminleri = []
            tum_dogrulama_hedefleri = []

            with torch.no_grad():
                for batch in dogrulama_yukleyici:
                    try:
                        noktalar = batch["points"].to(cihaz)
                        etiketler = batch["labels"].to(cihaz)

                        skorlar = model(noktalar)
                        kayip = kayip_fonksiyonu(skorlar, etiketler)

                        toplam_dogrulama_kaybi += kayip.item()
                        dogrulama_batch_sayaci += 1

                        tahminler = torch.argmax(skorlar, dim=1)
                        dogru_dogrulama_noktasi += (tahminler == etiketler).sum().item()
                        toplam_dogrulama_noktasi += etiketler.numel()

                        tum_dogrulama_tahminleri.append(tahminler.detach().cpu())
                        tum_dogrulama_hedefleri.append(etiketler.detach().cpu())
                    except Exception as dogrulama_hatasi:
                        kayitci.error(f"Dogrulama batch hatasi: {dogrulama_hatasi}")
                        continue

            ort_dogrulama_kaybi = (
                (toplam_dogrulama_kaybi / dogrulama_batch_sayaci) if dogrulama_batch_sayaci > 0 else 0.0
            )

            if tum_dogrulama_tahminleri and tum_dogrulama_hedefleri:
                birlesik_tahminler = torch.cat(tum_dogrulama_tahminleri, dim=0)
                birlesik_hedefler = torch.cat(tum_dogrulama_hedefleri, dim=0)
                dogrulama_metrikleri = evaluate_segmentation_batch(
                    birlesik_tahminler, birlesik_hedefler, sinif_sayisi=sinif_sayisi
                )
                dogrulama_dogrulugu = dogrulama_metrikleri["accuracy"]
                dogrulama_miou = dogrulama_metrikleri["mIoU"]
                dogrulama_dice = dogrulama_metrikleri["dice"]
            else:
                dogrulama_dogrulugu = (
                    (dogru_dogrulama_noktasi / toplam_dogrulama_noktasi) if toplam_dogrulama_noktasi > 0 else 0.0
                )
                dogrulama_miou = 0.0
                dogrulama_dice = 0.0

            kayitci.info(
                f"[Epoch {epoch}/{epoch_sayisi}] "
                f"Egitim Kaybi: {ort_egitim_kaybi:.4f}, Dogruluk: %{egitim_dogrulugu * 100:.2f} | "
                f"Dogrulama Kaybi: {ort_dogrulama_kaybi:.4f}, Dogruluk: %{dogrulama_dogrulugu * 100:.2f}, "
                f"mIoU: %{dogrulama_miou * 100:.2f}, Dice: %{dogrulama_dice * 100:.2f}"
            )

            # En yuksek mIoU veya dogruluga ulasildiginda agirliklari guvenli kaydet
            daha_iyi_mi = (dogrulama_miou > en_iyi_miou) or (en_iyi_miou == 0.0 and dogrulama_dogrulugu > en_iyi_dogruluk)
            if daha_iyi_mi:
                en_iyi_miou = max(en_iyi_miou, dogrulama_miou)
                en_iyi_dogruluk = max(en_iyi_dogruluk, dogrulama_dogrulugu)
                try:
                    checkpoint_dizini = os.path.dirname(checkpoint_yolu)
                    if checkpoint_dizini:
                        os.makedirs(checkpoint_dizini, exist_ok=True)
                    checkpoint_verisi = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizasyon.state_dict(),
                        "scheduler_state_dict": zamanlayici.state_dict(),
                        "best_val_miou": en_iyi_miou,
                        "best_val_acc": en_iyi_dogruluk,
                        "num_classes": sinif_sayisi,
                        "num_points": nokta_sayisi,
                    }
                    torch.save(checkpoint_verisi, checkpoint_yolu)
                    kayitci.info(
                        f"En iyi model agirligi guncellendi -> {checkpoint_yolu} "
                        f"(Dogrulama mIoU: %{dogrulama_miou * 100:.2f}, "
                        f"Dogrulama Dogruluk: %{dogrulama_dogrulugu * 100:.2f})"
                    )
                except Exception as kaydetme_hatasi:
                    kayitci.error(f"Checkpoint kaydedilemedi: {kaydetme_hatasi}")

        return True

    except KeyboardInterrupt:
        kayitci.warning("Kullanici tarafindan egitim durduruldu (KeyboardInterrupt).")
        try:
            yarida_kalan_yol = checkpoint_yolu.replace(".pth", "_interrupted.pth")
            checkpoint_verisi = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizasyon.state_dict(),
                "scheduler_state_dict": zamanlayici.state_dict(),
                "best_val_miou": en_iyi_miou,
                "best_val_acc": en_iyi_dogruluk,
            }
            torch.save(checkpoint_verisi, yarida_kalan_yol)
            kayitci.info(f"Mevcut model durumu kaydedildi: {yarida_kalan_yol}")
        except Exception:
            pass
        return False
    except Exception as hata:
        kayitci.error(f"Beklenmeyen egitim hatasi: {hata}")
        return False


if __name__ == "__main__":
    ayristirici = get_base_parser(description="Dental 3D Nokta Bulutu Segmentasyon Modeli Egitimi")
    ayristirici.add_argument("--epochs", type=int, default=None, help="Toplam epoch sayisi")
    ayristirici.add_argument("--batch-size", type=int, default=None, help="Mini-batch boyutu")
    ayristirici.add_argument("--lr", type=float, default=None, help="Ogrenme orani (Learning rate)")
    ayristirici.add_argument("--num-points", type=int, default=None, help="Model girdi nokta sayisi")
    ayristirici.add_argument("--checkpoint", type=str, default=None, help="Model agirlik dosya yolu (.pth)")
    ayristirici.add_argument("--resume", type=str, default=None, help="Egitime devam edilecek checkpoint dosya yolu (.pth)")
    ayristirici.add_argument("--parts", nargs="+", default=None, help="Egitilecek veri parcalari (orn: data_part_1 data_part_2)")

    argumanlar = ayristirici.parse_args()

    # Konfigurasyonu yukle
    yuklu_konfig = load_config(argumanlar.config)

    # CLI argumanlari varsa onceliklendir
    train(
        konfig=yuklu_konfig,
        epoch_sayisi=argumanlar.epochs,
        batch_boyutu=argumanlar.batch_size,
        ogrenme_orani=argumanlar.lr,
        nokta_sayisi=argumanlar.num_points,
        checkpoint_yolu=argumanlar.checkpoint,
        veri_parcalari=argumanlar.parts,
        devam_yolu=argumanlar.resume
    )
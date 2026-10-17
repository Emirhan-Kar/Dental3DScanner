import os
import glob
import json
import logging
import random
import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d
from utils.logger import get_logger
from utils.validator import validate_dental_sample, get_safe_path

logger = get_logger(__name__)

# FDI Dis Standart Numaralari (11-48 arasi 32 dis + 0 Dis Eti)
FDI_SINIFLARI = [0] + [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48
]

ETIKETTEN_INDEKSE = {fdi: indeks for indeks, fdi in enumerate(FDI_SINIFLARI)}
INDEKSTEN_ETIKETE = {indeks: fdi for indeks, fdi in enumerate(FDI_SINIFLARI)}

# Geriye donuk uyumluluk takma adlari (backward-compatibility aliases)
FDI_CLASSES = FDI_SINIFLARI
LABEL_TO_IDX = ETIKETTEN_INDEKSE
IDX_TO_LABEL = INDEKSTEN_ETIKETE


class DentalMeshDataset(data.Dataset):
    def __init__(
        self,
        veri_koku="data/3DTeethData",
        parcalar=None,
        nokta_sayisi=8192,
        veri_artirma=False,
        gecersizleri_ele=False,
        guvenli_mod=True,
        **ek_parametreler
    ):
        """
        Teeth3DS Veri Yukleyici (PyTorch).

        Args:
            veri_koku (str): Veri seti kok dizini.
            parcalar (list): Yuklenecek parca klasorleri (or. ["data_part_1"]).
            nokta_sayisi (int): Her ornekten cekilecek nokta sayisi (pozitif tamsayi).
            veri_artirma (bool): Rastgele dondurme ve gurultu ekleme aktifligi.
            gecersizleri_ele (bool): Baslangicta tum vakalari dogrulayip bozuk olanlari ayiklar.
            guvenli_mod (bool): Okuma hatasi olan vakalarda egitimi cokertmek yerine guvenli vaka ikamesi yapar.

        Raises:
            TypeError: Arguman tipleri gecersiz ise.
            ValueError: nokta_sayisi <= 0 ise.
            FileNotFoundError: veri_koku mevcut degilse.
        """
        super().__init__()

        veri_koku = ek_parametreler.get("data_root", veri_koku)
        parcalar = ek_parametreler.get("parts", parcalar)
        nokta_sayisi = ek_parametreler.get("num_points", nokta_sayisi)
        veri_artirma = ek_parametreler.get("augment", ek_parametreler.get("augmentation", veri_artirma))
        gecersizleri_ele = ek_parametreler.get("filter_invalid", gecersizleri_ele)
        guvenli_mod = ek_parametreler.get("safe_mode", guvenli_mod)

        if not isinstance(veri_koku, str):
            raise TypeError(f"veri_koku string olmalidir, alinan: {type(veri_koku).__name__}")
        if not os.path.exists(veri_koku):
            raise FileNotFoundError(f"Veri seti kok dizini bulunamadi: {os.path.abspath(veri_koku)}")

        if not isinstance(nokta_sayisi, int) or nokta_sayisi <= 0:
            raise ValueError(f"nokta_sayisi pozitif bir tamsayi olmalidir, alinan: {nokta_sayisi}")

        if parcalar is None:
            parcalar = ["data_part_1"]
        elif not isinstance(parcalar, (list, tuple)):
            raise TypeError(f"parcalar liste veya demet (tuple) olmalidir, alinan: {type(parcalar).__name__}")

        self.veri_koku = veri_koku
        self.nokta_sayisi = nokta_sayisi
        self.veri_artirma = bool(veri_artirma)
        self.guvenli_mod = bool(guvenli_mod)
        self.ornekler = []

        # Takma ozellikler
        self.data_root = self.veri_koku
        self.num_points = self.nokta_sayisi
        self.augment = self.veri_artirma
        self.safe_mode = self.guvenli_mod

        # Parca klasorlerindeki tum .obj ve .json ciftlerini topla
        for parca in parcalar:
            parca_yolu = os.path.join(veri_koku, str(parca))
            if not os.path.exists(parca_yolu):
                logger.warning(f"Belirtilen veri seti alt parcasi bulunamadi, atlaniyor: {parca_yolu}")
                continue

            obj_dosyalari = glob.glob(os.path.join(parca_yolu, "*", "*", "*.obj"))
            for obj_dosya in obj_dosyalari:
                ana_dizin = os.path.dirname(obj_dosya)
                temel_ad = os.path.splitext(os.path.basename(obj_dosya))[0]
                json_dosya = os.path.join(ana_dizin, f"{temel_ad}.json")

                if os.path.exists(json_dosya):
                    self.ornekler.append((obj_dosya, json_dosya))
                else:
                    logger.debug(f"JSON etiketi bulunamadi, atlandi: {obj_dosya}")

        # Istege bagli on dogrulama ve bozuk vakalari ayiklama
        if gecersizleri_ele and self.ornekler:
            gecerli_ornekler = []
            for obj_dosya, json_dosya in self.ornekler:
                sonuc = validate_dental_sample(obj_dosya, json_dosya, min_kose=100)
                if sonuc.gecerli_mi:
                    gecerli_ornekler.append((obj_dosya, json_dosya))
                else:
                    logger.warning(f"Bozuk vaka ayiklandi ({obj_dosya}): {sonuc.hata_mesaji}")
            elenen_sayi = len(self.ornekler) - len(gecerli_ornekler)
            if elenen_sayi > 0:
                logger.info(f"On dogrulama ile {elenen_sayi} adet gecersiz vaka ayiklandi.")
            self.ornekler = gecerli_ornekler

        logger.info(f"[{', '.join(parcalar)}] icinde toplam {len(self.ornekler)} adet etiketli vaka bulundu.")

    @property
    def samples(self):
        return self.ornekler

    @samples.setter
    def samples(self, value):
        self.ornekler = value

    def __len__(self):
        return len(self.ornekler)

    def _normalize_points(self, noktalar):
        """Merkezleme ve birim kureye olcekleme yapar."""
        if len(noktalar) == 0:
            return noktalar
        merkez = np.mean(noktalar, axis=0)
        noktalar -= merkez
        uzakliklar = np.sqrt(np.sum(noktalar ** 2, axis=1))
        en_buyuk_uzaklik = np.max(uzakliklar) if len(uzakliklar) > 0 else 0.0
        if en_buyuk_uzaklik > 1e-8:
            noktalar /= en_buyuk_uzaklik
        return noktalar

    def _augment(self, noktalar):
        """Z ekseni etrafinda rastgele dondurme ve kucuk Gaussian gurultu ekler."""
        aci = np.random.uniform(0, 2 * np.pi)
        dondurme_matrisi = np.array([
            [np.cos(aci), -np.sin(aci), 0],
            [np.sin(aci),  np.cos(aci), 0],
            [0,            0,           1]
        ], dtype=noktalar.dtype)

        # Pozisyonlari dondur
        noktalar[:, :3] = np.dot(noktalar[:, :3], dondurme_matrisi)
        # Varsa normalleri de ayni matrisle dondur
        if noktalar.shape[1] >= 6:
            noktalar[:, 3:6] = np.dot(noktalar[:, 3:6], dondurme_matrisi)

        gurultu = np.random.normal(0.0, 0.01, size=noktalar[:, :3].shape).astype(noktalar.dtype)
        noktalar[:, :3] += gurultu
        return noktalar

    def _load_sample(self, indeks):
        """Tek bir ornegi yukler ve tensorlere donusturur."""
        obj_yolu, json_yolu = self.ornekler[indeks]

        # 1. Mesh'i Open3D ile oku
        mesh = o3d.io.read_triangle_mesh(get_safe_path(obj_yolu))
        koseler = np.asarray(mesh.vertices, dtype=np.float32)
        if len(koseler) == 0:
            raise ValueError(f"Mesh dosyasi bos veya kose noktasi icermiyor: {obj_yolu}")

        try:
            mesh.compute_vertex_normals()
            normaller = np.asarray(mesh.vertex_normals, dtype=np.float32)
            if len(normaller) != len(koseler):
                normaller = np.zeros_like(koseler, dtype=np.float32)
        except Exception:
            normaller = np.zeros_like(koseler, dtype=np.float32)

        # 2. JSON etiketlerini oku
        with open(json_yolu, "r", encoding="utf-8") as dosya:
            ust_veri = json.load(dosya)

        ham_etiketler = ust_veri.get("labels", [])
        if not isinstance(ham_etiketler, list):
            ham_etiketler = []

        if len(ham_etiketler) == len(koseler):
            kose_etiketleri = np.array(ham_etiketler, dtype=np.int64)
        else:
            kose_etiketleri = np.zeros(len(koseler), dtype=np.int64)
            yuzler = np.asarray(mesh.triangles)
            for yuz_indeks, yuz in enumerate(yuzler):
                if yuz_indeks < len(ham_etiketler):
                    etiket_degeri = ham_etiketler[yuz_indeks]
                    if isinstance(etiket_degeri, (int, float)):
                        kose_etiketleri[yuz] = int(etiket_degeri)

        # 3. Sabit sayida nokta ornekle
        kose_sayisi = len(koseler)
        if kose_sayisi >= self.nokta_sayisi:
            secim = np.random.choice(kose_sayisi, self.nokta_sayisi, replace=False)
        else:
            secim = np.random.choice(kose_sayisi, self.nokta_sayisi, replace=True)

        secilen_koseler = koseler[secim]
        secilen_normaller = normaller[secim]
        secilen_etiketler = kose_etiketleri[secim]

        # 4. Normalizasyon ve Veri Artirma
        secilen_koseler = self._normalize_points(secilen_koseler)
        oznitelikler = np.concatenate([secilen_koseler, secilen_normaller], axis=-1)

        # NaN/Inf temizligi
        if not np.all(np.isfinite(oznitelikler)):
            oznitelikler = np.nan_to_num(oznitelikler, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.veri_artirma:
            oznitelikler = self._augment(oznitelikler)

        eslenmis_etiketler = np.array(
            [ETIKETTEN_INDEKSE.get(int(etiket), 0) for etiket in secilen_etiketler], dtype=np.int64
        )

        return {
            "points": torch.tensor(oznitelikler, dtype=torch.float32).transpose(0, 1),
            "labels": torch.tensor(eslenmis_etiketler, dtype=torch.long),
            "patient_id": str(ust_veri.get("id_patient", "unknown"))
        }

    def __getitem__(self, indeks):
        if indeks < 0 or indeks >= len(self.ornekler):
            raise IndexError(f"Indeks sinir disi: {indeks} (Veri seti boyutu: {len(self.ornekler)})")

        if not self.guvenli_mod:
            return self._load_sample(indeks)

        deneme = 0
        guncel_indeks = indeks
        while deneme < 5:
            try:
                return self._load_sample(guncel_indeks)
            except Exception as hata:
                deneme += 1
                bozuk_obj = self.ornekler[guncel_indeks][0]
                logger.warning(
                    f"Bozuk veri ornegi ({bozuk_obj}): {hata}. "
                    f"Guvenli mod devrede, baska ornek deneniyor ({deneme}/5)..."
                )
                guncel_indeks = random.randint(0, len(self.ornekler) - 1)

        # 5 deneme de basarisiz olursa hata firlat
        raise RuntimeError("Ust uste 5 vaka yuklenemedi, veri seti butunlugu bozulmus olabilir.")
import os
import glob
import json
import hashlib
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

FDI_CLASSES = FDI_SINIFLARI
LABEL_TO_IDX = ETIKETTEN_INDEKSE
IDX_TO_LABEL = INDEKSTEN_ETIKETE


# ─────────────────────────────────────────────────────────────────────
# Yardimci fonksiyonlar: tarama, hasta bazli bolme, sinif agirliklari
# ─────────────────────────────────────────────────────────────────────

def ornekleri_tara(veri_koku, parcalar):
    """Parca klasorlerindeki tum (.obj, .json) ciftlerini bulur.

    Teeth3DS yapisi: <veri_koku>/<parca>/<upper|lower>/<HASTA_ID>/<HASTA_ID>_<cene>.obj
    """
    ornekler = []
    for parca in parcalar:
        parca_yolu = os.path.join(veri_koku, str(parca))
        if not os.path.exists(parca_yolu):
            logger.warning(f"Belirtilen veri seti alt parcasi bulunamadi, atlaniyor: {parca_yolu}")
            continue
        for obj_dosya in sorted(glob.glob(os.path.join(parca_yolu, "*", "*", "*.obj"))):
            json_dosya = os.path.splitext(obj_dosya)[0] + ".json"
            if os.path.exists(json_dosya):
                ornekler.append((obj_dosya, json_dosya))
            else:
                logger.debug(f"JSON etiketi bulunamadi, atlandi: {obj_dosya}")
    return ornekler


def hasta_kimligi(obj_yolu):
    """Ornegin ait oldugu hasta kimligi (Teeth3DS'te hasta klasor adi)."""
    return os.path.basename(os.path.dirname(obj_yolu))


def hasta_bazli_bol(ornekler, dogrulama_orani=0.1, test_orani=0.1, tohum=42):
    """Ornekleri HASTA bazinda egitim / dogrulama / test olarak boler.

    Ayni hastanin alt ve ust cenesi her zaman ayni sete duser; bu sayede
    test sonuclarina hasta sizintisi (data leakage) karismaz.

    Returns:
        dict: {"train": [...], "val": [...], "test": [...]} (obj, json) listeleri
    """
    if not (0.0 <= dogrulama_orani < 1.0 and 0.0 <= test_orani < 1.0 and dogrulama_orani + test_orani < 1.0):
        raise ValueError("dogrulama_orani + test_orani 1'den kucuk olmalidir.")

    hastalar = sorted({hasta_kimligi(o) for o, _ in ornekler})
    rng = random.Random(tohum)
    rng.shuffle(hastalar)

    n = len(hastalar)
    n_test = int(round(n * test_orani))
    n_val = int(round(n * dogrulama_orani))
    if n >= 3:
        n_test = max(n_test, 1) if test_orani > 0 else 0
        n_val = max(n_val, 1) if dogrulama_orani > 0 else 0

    test_h = set(hastalar[:n_test])
    val_h = set(hastalar[n_test:n_test + n_val])

    bolum = {"train": [], "val": [], "test": []}
    for o, j in ornekler:
        h = hasta_kimligi(o)
        anahtar = "test" if h in test_h else ("val" if h in val_h else "train")
        bolum[anahtar].append((o, j))
    return bolum


def bolumu_kaydet(bolum, yol, veri_koku=None):
    """Bolmeyi (goreli yollarla) JSON olarak kaydeder; rapora eklenebilir ve tekrar uretilebilir."""
    def goreli(p):
        return os.path.relpath(p, veri_koku).replace("\\", "/") if veri_koku else p.replace("\\", "/")
    cikti = {k: [[goreli(o), goreli(j)] for o, j in v] for k, v in bolum.items()}
    cikti["_hasta_sayilari"] = {k: len({hasta_kimligi(o) for o, _ in v}) for k, v in bolum.items()}
    os.makedirs(os.path.dirname(yol) or ".", exist_ok=True)
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(cikti, f, indent=1)


def bolumu_yukle(yol, veri_koku=None):
    with open(yol, "r", encoding="utf-8") as f:
        ham = json.load(f)
    def mutlak(p):
        return os.path.join(veri_koku, p) if veri_koku else p
    return {k: [(mutlak(o), mutlak(j)) for o, j in ham[k]] for k in ("train", "val", "test") if k in ham}


def sinif_agirliklari_hesapla(ornekler, sinif_sayisi=33, maks_ornek=400, tohum=0):
    """Sadece JSON etiketlerini okuyarak (mesh okumadan) sinif frekanslarini cikarir
    ve ENet tarzi agirlik uretir: w_c = 1 / ln(1.02 + p_c).

    Dis eti noktalari cogunlukta oldugu icin agirliksiz CrossEntropy modeli
    'her seye dis eti de' yonune iter; bu agirliklar kucuk dislerin (kesiciler,
    20 yas dislari) ogrenilmesini saglar.
    """
    sayac = np.zeros(sinif_sayisi, dtype=np.float64)
    secim = list(ornekler)
    random.Random(tohum).shuffle(secim)
    for _, json_yolu in secim[:maks_ornek]:
        try:
            with open(json_yolu, "r", encoding="utf-8") as f:
                etiketler = json.load(f).get("labels", [])
            idx = np.array([ETIKETTEN_INDEKSE.get(int(e), 0) for e in etiketler], dtype=np.int64)
            sayac += np.bincount(idx, minlength=sinif_sayisi)[:sinif_sayisi]
        except Exception as hata:
            logger.debug(f"Etiket okunamadi ({json_yolu}): {hata}")
    if sayac.sum() == 0:
        return np.ones(sinif_sayisi, dtype=np.float32)
    olasilik = sayac / sayac.sum()
    agirlik = 1.0 / np.log(1.02 + olasilik)
    agirlik[sayac == 0] = 0.0          # Hic gorulmeyen siniflar kayba katki yapmasin
    agirlik = agirlik / agirlik[sayac > 0].mean()
    return agirlik.astype(np.float32)


def obj_oku(obj_yolu):
    """OBJ dosyasini kose SIRASINI KORUYARAK okur.

    DIKKAT: open3d.io.read_triangle_mesh OBJ okurken koselerin sirasini degistirir.
    Teeth3DS etiketleri (json 'labels') OBJ'deki 'v' satirlarinin sirasina gore verildigi icin
    Open3D ile okunan koselere etiket eslemek etiketleri rastgele koselere dagitir.
    Bu fonksiyon dosyayi dogrudan ayristirarak sirayi korur.

    Returns:
        koseler (N, 3) float32, yuzler (M, 3) int64
    """
    koseler, yuzler = [], []
    with open(obj_yolu, "r", encoding="utf-8", errors="ignore") as f:
        for satir in f:
            if satir.startswith("v "):
                p = satir.split()
                koseler.append((float(p[1]), float(p[2]), float(p[3])))
            elif satir.startswith("f "):
                idx = [int(t.split("/")[0]) for t in satir.split()[1:]]
                for i in range(1, len(idx) - 1):          # cokgenleri ucgenlere bol (fan)
                    yuzler.append((idx[0], idx[i], idx[i + 1]))
    v = np.asarray(koseler, dtype=np.float32).reshape(-1, 3)
    y = np.asarray(yuzler, dtype=np.int64).reshape(-1, 3)
    if len(y):
        y = np.where(y > 0, y - 1, y + len(v))                 # 1 tabanli / negatif indeksler
    return v, y


def kose_normalleri(koseler, yuzler):
    """Yuz normallerinin alan agirlikli toplamiyla kose normalleri."""
    n = np.zeros_like(koseler, dtype=np.float64)
    if len(yuzler):
        a, b, c = koseler[yuzler[:, 0]], koseler[yuzler[:, 1]], koseler[yuzler[:, 2]]
        yn = np.cross(b - a, c - a)
        for i in range(3):
            np.add.at(n, yuzler[:, i], yn)
    uz = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.maximum(uz, 1e-12)).astype(np.float32)


def mesh_geometri_oku(yol):
    """Herhangi bir mesh dosyasini (koseler, yuzler) olarak, kose sirasi korunarak okur."""
    if yol.lower().endswith(".obj"):
        return obj_oku(yol)
    mesh = o3d.io.read_triangle_mesh(get_safe_path(yol))
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.triangles, dtype=np.int64)


def o3d_mesh_olustur(koseler, yuzler, renkler=None):
    """(koseler, yuzler) dizilerinden kose sirasi korunmus bir Open3D mesh'i kurar."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(koseler, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(yuzler, dtype=np.int32))
    if renkler is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(renkler, dtype=np.float64))
    mesh.compute_vertex_normals()
    return mesh


def mesh_oku(obj_yolu, json_yolu=None):
    """Mesh'i okuyup (koseler, normaller, etiket_indeksleri, hasta_id) dondurur (kose sirasi korunur)."""
    koseler, yuzler = mesh_geometri_oku(obj_yolu)
    if len(koseler) == 0:
        raise ValueError(f"Mesh dosyasi bos veya kose noktasi icermiyor: {obj_yolu}")
    normaller = kose_normalleri(koseler, yuzler)

    etiket_idx = np.zeros(len(koseler), dtype=np.int64)
    hasta = hasta_kimligi(obj_yolu)
    if json_yolu is not None:
        with open(json_yolu, "r", encoding="utf-8") as dosya:
            ust_veri = json.load(dosya)
        hasta = str(ust_veri.get("id_patient", hasta))
        ham = ust_veri.get("labels", [])
        if not isinstance(ham, list):
            ham = []
        if len(ham) == len(koseler):
            fdi = np.array(ham, dtype=np.int64)
        else:
            # Yuz (face) bazli etiket ise koselere tasi
            fdi = np.zeros(len(koseler), dtype=np.int64)
            for yi, yuz in enumerate(yuzler[:len(ham)]):
                if isinstance(ham[yi], (int, float)):
                    fdi[yuz] = int(ham[yi])
        etiket_idx = np.array([ETIKETTEN_INDEKSE.get(int(e), 0) for e in fdi], dtype=np.int64)

    return koseler, np.nan_to_num(normaller), etiket_idx, hasta


def noktalari_normalize_et(noktalar):
    """Merkezleme ve birim kureye olcekleme (yerinde degil, kopya dondurur)."""
    if len(noktalar) == 0:
        return noktalar
    noktalar = noktalar - noktalar.mean(axis=0)
    r = np.sqrt((noktalar ** 2).sum(axis=1)).max()
    if r > 1e-8:
        noktalar = noktalar / r
    return noktalar


class DentalMeshDataset(data.Dataset):
    def __init__(
        self,
        veri_koku="data/3DTeethData",
        parcalar=None,
        nokta_sayisi=8192,
        veri_artirma=False,
        gecersizleri_ele=False,
        guvenli_mod=True,
        ornek_listesi=None,
        onbellek_dizini=None,
        donme_acisi_derece=15.0,
        sabit_ornekleme=False,
        **ek_parametreler
    ):
        """
        Teeth3DS Veri Yukleyici (PyTorch).

        Args:
            veri_koku (str): Veri seti kok dizini.
            parcalar (list): Yuklenecek parca klasorleri (or. ["data_part_1"]).
            nokta_sayisi (int): Her ornekten cekilecek nokta sayisi (pozitif tamsayi).
            veri_artirma (bool): Rastgele dondurme / olcekleme / gurultu (SADECE egitim setinde acin).
            gecersizleri_ele (bool): Baslangicta tum vakalari dogrulayip bozuk olanlari ayiklar.
            guvenli_mod (bool): Okuma hatasi olan vakalarda egitimi cokertmek yerine guvenli vaka ikamesi yapar.
            ornek_listesi (list): Dogrudan (obj, json) listesi verilirse klasor taramasi yapilmaz
                                  (hasta bazli bolme sonrasi kullanilir).
            onbellek_dizini (str): Verilirse her mesh ilk okunusta .npz olarak saklanir; sonraki
                                   epoch'lar .obj ayristirmadan ~20-50 kat hizli yuklenir.
            donme_acisi_derece (float): Veri artirmada dikey (Z) eksen etrafinda maks. donme acisi.
            sabit_ornekleme (bool): True ise her vaka hep ayni noktalarla orneklenir (dogrulama
                                    metriklerinin epoch'tan epoch'a karsilastirilabilir olmasi icin).
        """
        super().__init__()

        veri_koku = ek_parametreler.get("data_root", veri_koku)
        parcalar = ek_parametreler.get("parts", parcalar)
        nokta_sayisi = ek_parametreler.get("num_points", nokta_sayisi)
        veri_artirma = ek_parametreler.get("augment", ek_parametreler.get("augmentation", veri_artirma))
        gecersizleri_ele = ek_parametreler.get("filter_invalid", gecersizleri_ele)
        guvenli_mod = ek_parametreler.get("safe_mode", guvenli_mod)
        ornek_listesi = ek_parametreler.get("samples", ornek_listesi)
        onbellek_dizini = ek_parametreler.get("cache_dir", onbellek_dizini)

        if not isinstance(nokta_sayisi, int) or nokta_sayisi <= 0:
            raise ValueError(f"nokta_sayisi pozitif bir tamsayi olmalidir, alinan: {nokta_sayisi}")

        self.veri_koku = veri_koku
        self.nokta_sayisi = nokta_sayisi
        self.veri_artirma = bool(veri_artirma)
        self.guvenli_mod = bool(guvenli_mod)
        self.donme_acisi = float(np.radians(donme_acisi_derece))
        self.onbellek_dizini = onbellek_dizini
        self.sabit_ornekleme = bool(sabit_ornekleme)
        if onbellek_dizini:
            os.makedirs(onbellek_dizini, exist_ok=True)

        if ornek_listesi is not None:
            self.ornekler = [tuple(o) for o in ornek_listesi]
            parcalar = parcalar or []
        else:
            if not isinstance(veri_koku, str):
                raise TypeError(f"veri_koku string olmalidir, alinan: {type(veri_koku).__name__}")
            if not os.path.exists(veri_koku):
                raise FileNotFoundError(f"Veri seti kok dizini bulunamadi: {os.path.abspath(veri_koku)}")
            if parcalar is None:
                parcalar = ["data_part_1"]
            elif not isinstance(parcalar, (list, tuple)):
                raise TypeError(f"parcalar liste veya demet (tuple) olmalidir, alinan: {type(parcalar).__name__}")
            self.ornekler = ornekleri_tara(veri_koku, parcalar)

        # Takma ozellikler (Ingilizce API uyumlulugu)
        self.data_root = self.veri_koku
        self.num_points = self.nokta_sayisi
        self.augment = self.veri_artirma
        self.safe_mode = self.guvenli_mod

        if gecersizleri_ele and self.ornekler:
            gecerli = []
            for obj_dosya, json_dosya in self.ornekler:
                sonuc = validate_dental_sample(obj_dosya, json_dosya, min_kose=100)
                if sonuc.gecerli_mi:
                    gecerli.append((obj_dosya, json_dosya))
                else:
                    logger.warning(f"Bozuk vaka ayiklandi ({obj_dosya}): {sonuc.hata_mesaji}")
            if len(self.ornekler) - len(gecerli) > 0:
                logger.info(f"On dogrulama ile {len(self.ornekler) - len(gecerli)} adet gecersiz vaka ayiklandi.")
            self.ornekler = gecerli

        kaynak = f"[{', '.join(map(str, parcalar))}] icinde" if ornek_listesi is None else "Verilen listede"
        logger.info(f"{kaynak} toplam {len(self.ornekler)} adet etiketli vaka bulundu.")

    @property
    def samples(self):
        return self.ornekler

    @samples.setter
    def samples(self, value):
        self.ornekler = value

    def __len__(self):
        return len(self.ornekler)

    # Geriye donuk uyumluluk
    def _normalize_points(self, noktalar):
        return noktalari_normalize_et(noktalar)

    def _augment(self, oznitelikler):
        """Z (dikey/okluzal) ekseni etrafinda kucuk donme, olcekleme ve Gaussian gurultu.

        Teeth3DS ceneleri hizali geldigi icin tam 360 derece donme yerine kucuk bir aci
        kullanilir; tam donme sag/sol dislerin ayrimini gereksiz yere zorlastirir.
        """
        aci = np.random.uniform(-self.donme_acisi, self.donme_acisi)
        c, s = np.cos(aci), np.sin(aci)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=oznitelikler.dtype)
        oznitelikler[:, :3] = oznitelikler[:, :3] @ R.T
        if oznitelikler.shape[1] >= 6:
            oznitelikler[:, 3:6] = oznitelikler[:, 3:6] @ R.T
        oznitelikler[:, :3] *= np.random.uniform(0.9, 1.1)
        oznitelikler[:, :3] += np.random.normal(0.0, 0.005, size=(len(oznitelikler), 3)).astype(oznitelikler.dtype)
        return oznitelikler

    def _onbellek_yolu(self, obj_yolu):
        anahtar = hashlib.md5(os.path.abspath(obj_yolu).encode("utf-8")).hexdigest()[:12]
        return os.path.join(self.onbellek_dizini, f"{hasta_kimligi(obj_yolu)}_{anahtar}_v2.npz")

    def ham_ornek(self, indeks):
        """Tam cozunurlukte (koseler, normaller, etiketler, hasta) dondurur; onbellegi kullanir."""
        obj_yolu, json_yolu = self.ornekler[indeks]
        if self.onbellek_dizini:
            yol = self._onbellek_yolu(obj_yolu)
            if os.path.exists(yol):
                try:
                    d = np.load(yol, allow_pickle=False)
                    return d["v"].astype(np.float32), d["n"].astype(np.float32), d["l"].astype(np.int64), str(d["h"])
                except Exception:
                    pass  # bozuk onbellek -> yeniden olustur
            k, n, l, h = mesh_oku(obj_yolu, json_yolu)
            gecici = yol + ".tmp.npz"
            np.savez(gecici, v=k, n=n.astype(np.float16), l=l.astype(np.uint8), h=np.array(h))
            os.replace(gecici, yol)
            return k, n, l, h
        return mesh_oku(obj_yolu, json_yolu)

    def _load_sample(self, indeks):
        koseler, normaller, etiketler, hasta = self.ham_ornek(indeks)

        kose_sayisi = len(koseler)
        rng = np.random.RandomState(indeks) if self.sabit_ornekleme else np.random
        secim = rng.choice(kose_sayisi, self.nokta_sayisi, replace=kose_sayisi < self.nokta_sayisi)

        konum = noktalari_normalize_et(koseler[secim])
        oznitelikler = np.concatenate([konum, normaller[secim]], axis=-1).astype(np.float32)
        if not np.all(np.isfinite(oznitelikler)):
            oznitelikler = np.nan_to_num(oznitelikler, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.veri_artirma:
            oznitelikler = self._augment(oznitelikler)

        return {
            "points": torch.from_numpy(np.ascontiguousarray(oznitelikler.T)),
            "labels": torch.from_numpy(etiketler[secim].astype(np.int64)),
            "patient_id": hasta,
        }

    def __getitem__(self, indeks):
        if indeks < 0 or indeks >= len(self.ornekler):
            raise IndexError(f"Indeks sinir disi: {indeks} (Veri seti boyutu: {len(self.ornekler)})")

        if not self.guvenli_mod:
            return self._load_sample(indeks)

        guncel = indeks
        for deneme in range(1, 6):
            try:
                return self._load_sample(guncel)
            except Exception as hata:
                logger.warning(
                    f"Bozuk veri ornegi ({self.ornekler[guncel][0]}): {hata}. "
                    f"Guvenli mod devrede, baska ornek deneniyor ({deneme}/5)..."
                )
                guncel = random.randint(0, len(self.ornekler) - 1)
        raise RuntimeError("Ust uste 5 vaka yuklenemedi, veri seti butunlugu bozulmus olabilir.")

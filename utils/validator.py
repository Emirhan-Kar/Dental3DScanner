import os
import sys
import json
import argparse
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
import open3d as o3d

# Proje kok dizinini sys.path'e ekle
KOK_DIZIN = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK_DIZIN not in sys.path:
    sys.path.insert(0, KOK_DIZIN)

from utils.logger import get_logger

kayitci = get_logger(__name__)

# Gecerli FDI Dis Siniflari (0: Dis Eti, 11-48: Disler)
GECERLI_FDI_SINIFLARI = set([0] + [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48
])
VALID_FDI_CLASSES = GECERLI_FDI_SINIFLARI


def get_safe_path(dosya_yolu: str, **ek_parametreler) -> str:
    """
    Windows'ta Turkce veya ozel karakter iceren dosya yollarini
    C++ cekirdekli kutuphaneler (Open3D, OpenCV) icin guvenli 8.3 kisa yola donusturur.
    """
    dosya_yolu = ek_parametreler.get("path", dosya_yolu)
    if os.name == "nt" and os.path.exists(dosya_yolu):
        try:
            import ctypes
            arabellek = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(str(dosya_yolu), arabellek, 1024):
                return arabellek.value
        except Exception:
            pass
    return str(dosya_yolu)


class ValidationResult:
    """Veri dogrulama sonuc nesnesi."""
    def __init__(
        self,
        gecerli_mi: bool = True,
        hata_mesaji: Optional[str] = None,
        uyari_mesaji: Optional[str] = None,
        detaylar: Optional[Dict[str, Any]] = None,
        **ek_parametreler
    ):
        self.gecerli_mi = ek_parametreler.get("is_valid", gecerli_mi)
        self.hata_mesaji = ek_parametreler.get("error_message", hata_mesaji)
        self.uyari_mesaji = ek_parametreler.get("warning_message", uyari_mesaji)
        self.detaylar = ek_parametreler.get("details", detaylar if detaylar is not None else {})

    @property
    def is_valid(self):
        return self.gecerli_mi

    @property
    def error_message(self):
        return self.hata_mesaji

    @property
    def warning_message(self):
        return self.uyari_mesaji

    @property
    def details(self):
        return self.detaylar

    def __bool__(self):
        return bool(self.gecerli_mi)


def validate_mesh(obj_yolu: str, min_kose: int = 100, **ek_parametreler) -> ValidationResult:
    """
    3D Mesh dosyasinin (.obj, .ply, .stl) fiziksel ve geometrik gecerliligini denetler.
    """
    obj_yolu = ek_parametreler.get("obj_path", obj_yolu)
    min_kose = ek_parametreler.get("min_vertices", min_kose)

    if not isinstance(obj_yolu, str) or not obj_yolu.strip():
        return ValidationResult(False, "Dosya yolu bos veya string tipinde degil.")

    if not os.path.exists(obj_yolu):
        return ValidationResult(False, f"Dosya mevcut degil: {obj_yolu}")

    try:
        dosya_boyutu = os.path.getsize(obj_yolu)
        if dosya_boyutu == 0:
            return ValidationResult(False, f"Dosya boyutu 0 bayt (bos dosya): {obj_yolu}")
    except OSError as hata:
        return ValidationResult(False, f"Dosya boyutuna erisilemedi: {hata}")

    guvenli_yol = get_safe_path(obj_yolu)
    try:
        mesh = o3d.io.read_triangle_mesh(guvenli_yol)
    except Exception as hata:
        return ValidationResult(False, f"Open3D mesh dosyasini okuyamadi: {hata}")

    koseler = np.asarray(mesh.vertices)
    kose_sayisi = len(koseler)

    if kose_sayisi < min_kose:
        return ValidationResult(
            False,
            f"Kose sayisi yetersiz ({kose_sayisi} < {min_kose}): {obj_yolu}",
            detaylar={"vertex_count": kose_sayisi}
        )

    # NaN / Inf kontrolu
    if not np.all(np.isfinite(koseler)):
        nan_sayisi = np.sum(~np.isfinite(koseler))
        return ValidationResult(
            False,
            f"Mesh koordinatlarinda {nan_sayisi} adet NaN/Inf degeri tespit edildi.",
            detaylar={"vertex_count": kose_sayisi, "nan_count": int(nan_sayisi)}
        )

    # Boyut ve cap denetimi
    kutu_min = np.min(koseler, axis=0)
    kutu_maks = np.max(koseler, axis=0)
    boyut = kutu_maks - kutu_min
    kosegen = np.linalg.norm(boyut)

    if kosegen < 1e-4:
        return ValidationResult(
            False,
            f"Mesh geometrisi dejenere (cap cok kucuk: {kosegen:.6f} mm): {obj_yolu}",
            detaylar={"vertex_count": kose_sayisi, "bounding_box_diagonal": float(kosegen)}
        )

    ucgen_sayisi = len(mesh.triangles)
    return ValidationResult(
        True,
        detaylar={
            "vertex_count": kose_sayisi,
            "triangle_count": ucgen_sayisi,
            "bounding_box_diagonal": float(kosegen),
            "file_size_bytes": dosya_boyutu
        }
    )


def validate_annotation(
    json_yolu: str,
    kose_sayisi: Optional[int] = None,
    yuz_sayisi: Optional[int] = None,
    **ek_parametreler
) -> ValidationResult:
    """
    JSON etiket dosyasinin formatini, semasini ve FDI standart uyumlulugunu denetler.
    """
    json_yolu = ek_parametreler.get("json_path", json_yolu)
    kose_sayisi = ek_parametreler.get("vertex_count", kose_sayisi)
    yuz_sayisi = ek_parametreler.get("face_count", yuz_sayisi)

    if not isinstance(json_yolu, str) or not json_yolu.strip():
        return ValidationResult(False, "JSON dosya yolu bos veya string degil.")

    if not os.path.exists(json_yolu):
        return ValidationResult(False, f"JSON etiket dosyasi bulunamadi: {json_yolu}")

    try:
        with open(json_yolu, "r", encoding="utf-8") as dosya:
            veri = json.load(dosya)
    except json.JSONDecodeError as hata:
        return ValidationResult(False, f"JSON formati bozuk: {hata}")
    except Exception as hata:
        return ValidationResult(False, f"JSON dosyasi acilamadi: {hata}")

    if not isinstance(veri, dict):
        return ValidationResult(False, "JSON kok elemani bir sozluk (dict) olmalidir.")

    ham_etiketler = veri.get("labels")
    if ham_etiketler is None:
        return ValidationResult(False, "JSON icinde 'labels' alani bulunamadi.")

    if not isinstance(ham_etiketler, list):
        return ValidationResult(False, f"'labels' bir liste olmalidir, alinan: {type(ham_etiketler).__name__}")

    etiket_sayisi = len(ham_etiketler)
    if etiket_sayisi == 0:
        return ValidationResult(False, "Etiket listesi ('labels') bos.")

    # FDI deger kontrolu
    benzersiz_etiketler = set()
    gecersiz_etiketler = set()

    for etiket in ham_etiketler:
        if not isinstance(etiket, (int, np.integer)):
            gecersiz_etiketler.add(f"Tip:{type(etiket).__name__}")
        else:
            tamsayi_etiket = int(etiket)
            benzersiz_etiketler.add(tamsayi_etiket)
            if tamsayi_etiket not in GECERLI_FDI_SINIFLARI:
                gecersiz_etiketler.add(tamsayi_etiket)

    if gecersiz_etiketler:
        return ValidationResult(
            False,
            f"Gecersiz FDI dis etiketleri tespit edildi: {list(gecersiz_etiketler)[:5]}",
            detaylar={"invalid_labels": list(gecersiz_etiketler), "total_labels": etiket_sayisi}
        )

    uyari_metni = None
    if kose_sayisi is not None and yuz_sayisi is not None:
        if etiket_sayisi != kose_sayisi and etiket_sayisi != yuz_sayisi:
            uyari_metni = (
                f"Etiket sayisi ({etiket_sayisi}), ne kose sayisi ({kose_sayisi}) "
                f"ne de poligon sayisi ({yuz_sayisi}) ile tam eslesiyor."
            )

    return ValidationResult(
        True,
        uyari_mesaji=uyari_metni,
        detaylar={
            "label_count": etiket_sayisi,
            "unique_classes_count": len(benzersiz_etiketler),
            "classes": sorted(list(benzersiz_etiketler)),
            "patient_id": veri.get("id_patient", "unknown")
        }
    )


def validate_dental_sample(
    obj_yolu: str,
    json_yolu: str,
    min_kose: int = 100,
    **ek_parametreler
) -> ValidationResult:
    """
    Hem 3D mesh hem de JSON etiket dosyasini birlikte butuncul olarak dogrular.
    """
    obj_yolu = ek_parametreler.get("obj_path", obj_yolu)
    json_yolu = ek_parametreler.get("json_path", json_yolu)
    min_kose = ek_parametreler.get("min_vertices", min_kose)

    mesh_sonucu = validate_mesh(obj_yolu, min_kose=min_kose)
    if not mesh_sonucu.gecerli_mi:
        return ValidationResult(False, f"Mesh hatasi: {mesh_sonucu.hata_mesaji}", detaylar=mesh_sonucu.detaylar)

    kose_adedi = mesh_sonucu.detaylar.get("vertex_count")
    yuz_adedi = mesh_sonucu.detaylar.get("triangle_count")

    etiket_sonucu = validate_annotation(json_yolu, kose_sayisi=kose_adedi, yuz_sayisi=yuz_adedi)
    if not etiket_sonucu.gecerli_mi:
        return ValidationResult(False, f"Etiket hatasi: {etiket_sonucu.hata_mesaji}", detaylar=etiket_sonucu.detaylar)

    birlesik_detaylar = {
        "mesh": mesh_sonucu.detaylar,
        "annotation": etiket_sonucu.detaylar
    }
    return ValidationResult(
        True,
        uyari_mesaji=etiket_sonucu.uyari_mesaji,
        detaylar=birlesik_detaylar
    )


def validate_point_cloud(nokta_bulutu: o3d.geometry.PointCloud, min_nokta: int = 10, **ek_parametreler) -> ValidationResult:
    """
    Open3D nokta bulutu nesnesini denetler.
    """
    nokta_bulutu = ek_parametreler.get("pcd", nokta_bulutu)
    min_nokta = ek_parametreler.get("min_points", min_nokta)

    if nokta_bulutu is None or not isinstance(nokta_bulutu, o3d.geometry.PointCloud):
        return ValidationResult(False, "Girdi nesnesi gecerli bir open3d.geometry.PointCloud degil.")

    noktalar = np.asarray(nokta_bulutu.points)
    nokta_sayisi = len(noktalar)

    if nokta_sayisi < min_nokta:
        return ValidationResult(False, f"Nokta sayisi yetersiz ({nokta_sayisi} < {min_nokta}).")

    if not np.all(np.isfinite(noktalar)):
        nan_sayisi = int(np.sum(~np.isfinite(noktalar)))
        return ValidationResult(False, f"Nokta bulutunda {nan_sayisi} adet NaN/Inf koordinati bulundu.")

    return ValidationResult(True, detaylar={"point_count": nokta_sayisi})


def validate_image_file(gorsel_yolu: str, min_genislik: int = 10, min_yukseklik: int = 10, **ek_parametreler) -> ValidationResult:
    """
    2D gorsel dosyasinin (.png, .jpg vb.) fiziksel ve format gecerliligini denetler.
    """
    gorsel_yolu = ek_parametreler.get("image_path", gorsel_yolu)
    min_genislik = ek_parametreler.get("min_width", min_genislik)
    min_yukseklik = ek_parametreler.get("min_height", min_yukseklik)

    if not os.path.exists(gorsel_yolu):
        return ValidationResult(False, f"Gorsel dosyasi bulunamadi: {gorsel_yolu}")

    try:
        ham_veri = np.fromfile(gorsel_yolu, dtype=np.uint8)
        gorsel = cv2.imdecode(ham_veri, cv2.IMREAD_COLOR)
    except Exception as hata:
        return ValidationResult(False, f"OpenCV gorseli okuyamadi: {hata}")

    if gorsel is None:
        return ValidationResult(False, f"Gorsel dosyasi bozuk veya acilamiyor: {gorsel_yolu}")

    yukseklik, genislik = gorsel.shape[:2]
    if yukseklik < min_yukseklik or genislik < min_genislik:
        return ValidationResult(False, f"Gorsel cozunurlugu cok kucuk ({genislik}x{yukseklik} < {min_genislik}x{min_yukseklik}).")

    kanal_sayisi = gorsel.shape[2] if gorsel.ndim == 3 else 1
    return ValidationResult(True, detaylar={"width": genislik, "height": yukseklik, "channels": kanal_sayisi})


def scan_dataset(
    veri_koku: str = "data/3DTeethData",
    parcalar: Optional[List[str]] = None,
    min_kose: int = 100,
    maks_ornek: Optional[int] = None,
    **ek_parametreler
) -> Dict[str, Any]:
    """
    Veri seti dizinini tarayarak tum .obj ve .json ciftlerini dogrular ve rapor uretir.
    """
    import glob

    veri_koku = ek_parametreler.get("data_root", veri_koku)
    parcalar = ek_parametreler.get("parts", parcalar)
    min_kose = ek_parametreler.get("min_vertices", min_kose)
    maks_ornek = ek_parametreler.get("max_samples", maks_ornek)

    if parcalar is None:
        parcalar = ["data_part_1"]

    kayitci.info(f"Veri seti dogrulama taramasi baslatiliyor: {veri_koku}, Parcalar: {parcalar}")

    toplam_taranan = 0
    gecerli_sayisi = 0
    gecersiz_ornekler = []
    uyari_listesi = []

    for parca in parcalar:
        parca_yolu = os.path.join(veri_koku, str(parca))
        if not os.path.exists(parca_yolu):
            kayitci.warning(f"Alt parca klasoru bulunamadi: {parca_yolu}")
            continue

        obj_dosyalari = glob.glob(os.path.join(parca_yolu, "*", "*", "*.obj"))
        for obj_dosya in obj_dosyalari:
            if maks_ornek and toplam_taranan >= maks_ornek:
                break

            ana_dizin = os.path.dirname(obj_dosya)
            temel_ad = os.path.splitext(os.path.basename(obj_dosya))[0]
            json_dosya = os.path.join(ana_dizin, f"{temel_ad}.json")

            toplam_taranan += 1

            if not os.path.exists(json_dosya):
                gecersiz_ornekler.append({
                    "sample": obj_dosya,
                    "error": "Eslesen .json etiket dosyasi eksik."
                })
                continue

            sonuc = validate_dental_sample(obj_dosya, json_dosya, min_kose=min_kose)
            if sonuc.gecerli_mi:
                gecerli_sayisi += 1
                if sonuc.uyari_mesaji:
                    uyari_listesi.append({"sample": obj_dosya, "warning": sonuc.uyari_mesaji})
            else:
                gecersiz_ornekler.append({
                    "sample": obj_dosya,
                    "error": sonuc.hata_mesaji
                })

    rapor = {
        "total_scanned": toplam_taranan,
        "valid_count": gecerli_sayisi,
        "invalid_count": len(gecersiz_ornekler),
        "warning_count": len(uyari_listesi),
        "valid_ratio_percent": (gecerli_sayisi / toplam_taranan * 100) if toplam_taranan > 0 else 0.0,
        "invalid_samples": gecersiz_ornekler,
        "warnings": uyari_listesi
    }

    kayitci.info(
        f"Dogrulama Tamamlandi -> Taranan: {toplam_taranan} | "
        f"Gecerli: {gecerli_sayisi} (%{rapor['valid_ratio_percent']:.1f}) | "
        f"Gecersiz: {len(gecersiz_ornekler)}"
    )
    return rapor


if __name__ == "__main__":
    ayristirici = argparse.ArgumentParser(description="Teeth3DS Veri Seti Dogrulama Araci")
    ayristirici.add_argument("--data-root", type=str, default="data/3DTeethData", help="Veri seti ana dizini")
    ayristirici.add_argument("--parts", nargs="+", default=["data_part_1"], help="Taranacak parcalar")
    ayristirici.add_argument("--min-vertices", type=int, default=100, help="Minimum kabul edilen kose sayisi")
    ayristirici.add_argument("--max-samples", type=int, default=None, help="Maksimum taranacak ornek sayisi")

    argumanlar = ayristirici.parse_args()
    sonuc_raporu = scan_dataset(
        veri_koku=argumanlar.data_root,
        parcalar=argumanlar.parts,
        min_kose=argumanlar.min_vertices,
        maks_ornek=argumanlar.max_samples
    )

    print("\n--- Veri Seti Dogrulama Raporu ---")
    print(f"Toplam Taranan  : {sonuc_raporu['total_scanned']}")
    print(f"Gecerli Vaka    : {sonuc_raporu['valid_count']} (%{sonuc_raporu['valid_ratio_percent']:.1f})")
    print(f"Gecersiz Vaka   : {sonuc_raporu['invalid_count']}")
    print(f"Uyari Sayisi    : {sonuc_raporu['warning_count']}")

    if sonuc_raporu["invalid_samples"]:
        print("\nIlk 5 Hata Ornegi:")
        for hata_ornek in sonuc_raporu["invalid_samples"][:5]:
            print(f" - {hata_ornek['sample']}: {hata_ornek['error']}")

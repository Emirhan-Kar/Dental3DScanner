import os
import sys
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2
import open3d as o3d

# Proje kok dizinini sys.path'e ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.logger import get_logger

logger = get_logger(__name__)

# Gecerli FDI Dis Siniflari (0: Dis Eti, 11-48: Disler)
VALID_FDI_CLASSES = set([0] + [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48
])


def get_safe_path(path: str) -> str:
    """
    Windows'ta Turkce veya ozel karakter iceren dosya yollarini
    C++ cekirdekli kutuphaneler (Open3D, OpenCV) icin guvenli 8.3 kisa yola donusturur.
    """
    if os.name == "nt" and os.path.exists(path):
        try:
            import ctypes
            buffer = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, 1024):
                return buffer.value
        except Exception:
            pass
    return str(path)


@dataclass
class ValidationResult:
    """Veri dogrulama sonuc nesnesi."""
    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self):
        return self.is_valid


def validate_mesh(obj_path: str, min_vertices: int = 100) -> ValidationResult:
    """
    3D Mesh dosyasinin (.obj, .ply, .stl) fiziksel ve geometrik gecerliligini denetler.

    Denetimler:
    - Dosya varligi, okunabilirligi ve boyutu (>0 bayt).
    - Open3D ile basarili parse edilme.
    - Kose (vertex) sayisi >= min_vertices.
    - Tum koordinatlarin sonlu (finite, NaN veya Inf icermeyen) olmasi.
    - Sinirlayici kutu (bounding box) capi > 0 (noktalar tek bir noktaya cokmemis).

    Args:
        obj_path (str): 3D model dosya yolu.
        min_vertices (int): Kabul edilebilir minimum kose sayisi.

    Returns:
        ValidationResult: Dogrulama sonucu ve detaylari.
    """
    if not isinstance(obj_path, str) or not obj_path.strip():
        return ValidationResult(False, "Dosya yolu bos veya string tipinde degil.")

    if not os.path.exists(obj_path):
        return ValidationResult(False, f"Dosya mevcut degil: {obj_path}")

    try:
        file_size = os.path.getsize(obj_path)
        if file_size == 0:
            return ValidationResult(False, f"Dosya boyutu 0 bayt (bos dosya): {obj_path}")
    except OSError as e:
        return ValidationResult(False, f"Dosya boyutuna erisilemedi: {e}")

    safe_path = get_safe_path(obj_path)
    try:
        mesh = o3d.io.read_triangle_mesh(safe_path)
    except Exception as e:
        return ValidationResult(False, f"Open3D mesh dosyasini okuyamadi: {e}")

    vertices = np.asarray(mesh.vertices)
    num_v = len(vertices)

    if num_v < min_vertices:
        return ValidationResult(
            False,
            f"Kose sayisi yetersiz ({num_v} < {min_vertices}): {obj_path}",
            details={"vertex_count": num_v}
        )

    # NaN / Inf kontrolu
    if not np.all(np.isfinite(vertices)):
        nan_count = np.sum(~np.isfinite(vertices))
        return ValidationResult(
            False,
            f"Mesh koordinatlarinda {nan_count} adet NaN/Inf degeri tespit edildi.",
            details={"vertex_count": num_v, "nan_count": int(nan_count)}
        )

    # Boyut ve cap denetimi
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    extent = bbox_max - bbox_min
    diagonal = np.linalg.norm(extent)

    if diagonal < 1e-4:
        return ValidationResult(
            False,
            f"Mesh geometrisi dejenere (cap cok kucuk: {diagonal:.6f} mm): {obj_path}",
            details={"vertex_count": num_v, "bounding_box_diagonal": float(diagonal)}
        )

    num_triangles = len(mesh.triangles)
    return ValidationResult(
        True,
        details={
            "vertex_count": num_v,
            "triangle_count": num_triangles,
            "bounding_box_diagonal": float(diagonal),
            "file_size_bytes": file_size
        }
    )


def validate_annotation(
    json_path: str,
    vertex_count: Optional[int] = None,
    face_count: Optional[int] = None
) -> ValidationResult:
    """
    JSON etiket dosyasinin formatini, semasini ve FDI standart uyumlulugunu denetler.

    Denetimler:
    - Dosya varligi ve gecerli JSON formati.
    - 'labels' listesi mevcudiyeti.
    - Etiketlerin tamsayi ve gecerli FDI numaralari (0, 11-48) olmasi.
    - Varsa vertex_count veya face_count ile uzunluk uyumlulugu.

    Args:
        json_path (str): JSON dosya yolu.
        vertex_count (int, optional): Karsilik gelen mesh'in kose sayisi.
        face_count (int, optional): Karsilik gelen mesh'in yuz (poligon) sayisi.

    Returns:
        ValidationResult: Dogrulama sonucu.
    """
    if not isinstance(json_path, str) or not json_path.strip():
        return ValidationResult(False, "JSON dosya yolu bos veya string degil.")

    if not os.path.exists(json_path):
        return ValidationResult(False, f"JSON etiket dosyasi bulunamadi: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ValidationResult(False, f"JSON formati bozuk: {e}")
    except Exception as e:
        return ValidationResult(False, f"JSON dosyasi acilamadi: {e}")

    if not isinstance(data, dict):
        return ValidationResult(False, "JSON kok elemani bir sozluk (dict) olmalidir.")

    raw_labels = data.get("labels")
    if raw_labels is None:
        return ValidationResult(False, "JSON icinde 'labels' alani bulunamadi.")

    if not isinstance(raw_labels, list):
        return ValidationResult(False, f"'labels' bir liste olmalidir, alinan: {type(raw_labels).__name__}")

    num_labels = len(raw_labels)
    if num_labels == 0:
        return ValidationResult(False, "Etiket listesi ('labels') bos.")

    # FDI deger kontrolu
    unique_labels = set()
    invalid_labels = set()

    for idx, lbl in enumerate(raw_labels):
        if not isinstance(lbl, (int, np.integer)):
            invalid_labels.add(f"Tip:{type(lbl).__name__}")
        else:
            int_lbl = int(lbl)
            unique_labels.add(int_lbl)
            if int_lbl not in VALID_FDI_CLASSES:
                invalid_labels.add(int_lbl)

    if invalid_labels:
        return ValidationResult(
            False,
            f"Gecersiz FDI dis etiketleri tespit edildi: {list(invalid_labels)[:5]}",
            details={"invalid_labels": list(invalid_labels), "total_labels": num_labels}
        )

    warning_msg = None
    # Uzunluk karsilastirmasi
    if vertex_count is not None and face_count is not None:
        if num_labels != vertex_count and num_labels != face_count:
            warning_msg = (
                f"Etiket sayisi ({num_labels}), ne kose sayisi ({vertex_count}) "
                f"ne de poligon sayisi ({face_count}) ile tam eslesiyor."
            )

    return ValidationResult(
        True,
        warning_message=warning_msg,
        details={
            "label_count": num_labels,
            "unique_classes_count": len(unique_labels),
            "classes": sorted(list(unique_labels)),
            "patient_id": data.get("id_patient", "unknown")
        }
    )


def validate_dental_sample(
    obj_path: str,
    json_path: str,
    min_vertices: int = 100
) -> ValidationResult:
    """
    Hem 3D mesh hem de JSON etiket dosyasini birlikte butuncul olarak dogrular.
    """
    mesh_res = validate_mesh(obj_path, min_vertices=min_vertices)
    if not mesh_res.is_valid:
        return ValidationResult(False, f"Mesh hatasi: {mesh_res.error_message}", details=mesh_res.details)

    v_count = mesh_res.details.get("vertex_count")
    t_count = mesh_res.details.get("triangle_count")

    anno_res = validate_annotation(json_path, vertex_count=v_count, face_count=t_count)
    if not anno_res.is_valid:
        return ValidationResult(False, f"Etiket hatasi: {anno_res.error_message}", details=anno_res.details)

    combined_details = {
        "mesh": mesh_res.details,
        "annotation": anno_res.details
    }
    return ValidationResult(
        True,
        warning_message=anno_res.warning_message,
        details=combined_details
    )


def validate_point_cloud(pcd: o3d.geometry.PointCloud, min_points: int = 10) -> ValidationResult:
    """
    Open3D nokta bulutu nesnesini denetler.
    """
    if pcd is None or not isinstance(pcd, o3d.geometry.PointCloud):
        return ValidationResult(False, "Girdi nesnesi gecerli bir open3d.geometry.PointCloud degil.")

    pts = np.asarray(pcd.points)
    num_pts = len(pts)

    if num_pts < min_points:
        return ValidationResult(False, f"Nokta sayisi yetersiz ({num_pts} < {min_points}).")

    if not np.all(np.isfinite(pts)):
        nan_count = int(np.sum(~np.isfinite(pts)))
        return ValidationResult(False, f"Nokta bulutunda {nan_count} adet NaN/Inf koordinati bulundu.")

    return ValidationResult(True, details={"point_count": num_pts})


def validate_image_file(image_path: str, min_width: int = 10, min_height: int = 10) -> ValidationResult:
    """
    2D gorsel dosyasinin (.png, .jpg vb.) fiziksel ve format gecerliligini denetler.
    Windows Unicode yollarini guvenle okumak icin np.fromfile + imdecode kullanir.
    """
    if not os.path.exists(image_path):
        return ValidationResult(False, f"Gorsel dosyasi bulunamadi: {image_path}")

    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        return ValidationResult(False, f"OpenCV gorseli okuyamadi: {e}")

    if img is None:
        return ValidationResult(False, f"Gorsel dosyasi bozuk veya acilamiyor: {image_path}")

    h, w = img.shape[:2]
    if h < min_height or w < min_width:
        return ValidationResult(False, f"Gorsel cozunurlugu cok kucuk ({w}x{h} < {min_width}x{min_height}).")

    return ValidationResult(True, details={"width": w, "height": h, "channels": img.shape[2] if img.ndim == 3 else 1})


def scan_dataset(
    data_root: str = "data/3DTeethData",
    parts: Optional[List[str]] = None,
    min_vertices: int = 100,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Veri seti dizinini tarayarak tum .obj ve .json ciftlerini dogrular ve rapor uretir.

    Args:
        data_root (str): Veri seti kok dizini.
        parts (list, optional): Taranacak alt klasorler (orn: ["data_part_1"]).
        min_vertices (int): Minimum kabul edilen vertex sayisi.
        max_samples (int, optional): Maksimum taranacak vaka limiti (hizli kontrol icin).

    Returns:
        dict: Dogrulama istatistikleri ve tespit edilen sorunlar.
    """
    import glob

    if parts is None:
        parts = ["data_part_1"]

    logger.info(f"Veri seti dogrulama taramasi baslatiliyor: {data_root}, Parcalar: {parts}")

    total_scanned = 0
    valid_count = 0
    invalid_samples = []
    warnings_list = []

    for part in parts:
        part_path = os.path.join(data_root, str(part))
        if not os.path.exists(part_path):
            logger.warning(f"Alt parca klasoru bulunamadi: {part_path}")
            continue

        obj_files = glob.glob(os.path.join(part_path, "*", "*", "*.obj"))
        for obj_f in obj_files:
            if max_samples and total_scanned >= max_samples:
                break

            base_dir = os.path.dirname(obj_f)
            base_name = os.path.splitext(os.path.basename(obj_f))[0]
            json_f = os.path.join(base_dir, f"{base_name}.json")

            total_scanned += 1

            if not os.path.exists(json_f):
                invalid_samples.append({
                    "sample": obj_f,
                    "error": "Eslesen .json etiket dosyasi eksik."
                })
                continue

            res = validate_dental_sample(obj_f, json_f, min_vertices=min_vertices)
            if res.is_valid:
                valid_count += 1
                if res.warning_message:
                    warnings_list.append({"sample": obj_f, "warning": res.warning_message})
            else:
                invalid_samples.append({
                    "sample": obj_f,
                    "error": res.error_message
                })

    report = {
        "total_scanned": total_scanned,
        "valid_count": valid_count,
        "invalid_count": len(invalid_samples),
        "warning_count": len(warnings_list),
        "valid_ratio_percent": (valid_count / total_scanned * 100) if total_scanned > 0 else 0.0,
        "invalid_samples": invalid_samples,
        "warnings": warnings_list
    }

    logger.info(
        f"Dogrulama Tamamlandi -> Taranan: {total_scanned} | "
        f"Gecerli: {valid_count} (%{report['valid_ratio_percent']:.1f}) | "
        f"Gecersiz: {len(invalid_samples)}"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teeth3DS Veri Seti Dogrulama Araci")
    parser.add_argument("--data-root", type=str, default="data/3DTeethData", help="Veri seti ana dizini")
    parser.add_argument("--parts", nargs="+", default=["data_part_1"], help="Taranacak parcalar")
    parser.add_argument("--min-vertices", type=int, default=100, help="Minimum kabul edilen kose sayisi")
    parser.add_argument("--max-samples", type=int, default=None, help="Maksimum taranacak ornek sayisi")

    args = parser.parse_args()
    report = scan_dataset(
        data_root=args.data_root,
        parts=args.parts,
        min_vertices=args.min_vertices,
        max_samples=args.max_samples
    )

    print("\n--- Veri Seti Dogrulama Raporu ---")
    print(f"Toplam Taranan  : {report['total_scanned']}")
    print(f"Gecerli Vaka    : {report['valid_count']} (%{report['valid_ratio_percent']:.1f})")
    print(f"Gecersiz Vaka   : {report['invalid_count']}")
    print(f"Uyari Sayisi    : {report['warning_count']}")

    if report["invalid_samples"]:
        print("\nIlk 5 Hata Ornegi:")
        for err in report["invalid_samples"][:5]:
            print(f" - {err['sample']}: {err['error']}")

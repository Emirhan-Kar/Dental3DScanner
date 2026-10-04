import copy
from typing import Tuple, Optional, Union
import numpy as np
import open3d as o3d

from utils.logger import get_logger

logger = get_logger(__name__)


def validate_point_cloud_input(pcd: o3d.geometry.PointCloud, name: str = "pcd"):
    """Nokta bulutu nesnesinin tip ve gecerlilik kontrolunu yapar."""
    if pcd is None:
        raise TypeError(f"{name} None olamaz. Gecerli bir open3d.geometry.PointCloud nesnesi bekleniyor.")
    if not isinstance(pcd, o3d.geometry.PointCloud):
        raise TypeError(f"{name} tipi open3d.geometry.PointCloud olmalidir, alinan: {type(pcd).__name__}")
    if len(pcd.points) == 0:
        raise ValueError(f"{name} bos nokta bulutu iceriyor (0 nokta).")


def remove_statistical_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Istatiksel Aykiri Deger Filtresi (Statistical Outlier Removal - SOR).

    Her noktanin en yakin k komsusuna olan ortalama mesafesini hesaplar.
    Mesafe ortalamasi Gauss dagiliminin disinda kalan (std_ratio carpanindan uzak)
    parazit ve yansima noktalarini eler.

    Args:
        pcd (o3d.geometry.PointCloud): Girdi nokta bulutu.
        nb_neighbors (int): Analiz edilecek en yakin komsu sayisi (k).
        std_ratio (float): Standart sapma esik carpani.

    Returns:
        Tuple[o3d.geometry.PointCloud, np.ndarray]: (Filtrelenmis_bulut, Secilen_indeksler).
    """
    validate_point_cloud_input(pcd, "pcd")

    if not isinstance(nb_neighbors, int) or nb_neighbors <= 0:
        raise ValueError(f"nb_neighbors pozitif bir tam sayi olmalidir, alinan: {nb_neighbors}")
    if not isinstance(std_ratio, (int, float)) or std_ratio <= 0:
        raise ValueError(f"std_ratio pozitif bir sayi olmalidir, alinan: {std_ratio}")

    initial_pts = len(pcd.points)
    if initial_pts <= nb_neighbors:
        logger.warning(f"Nokta sayisi ({initial_pts}) komsu sayisindan ({nb_neighbors}) az veya esit. Filtre uygulanamadi.")
        return copy.deepcopy(pcd), np.arange(initial_pts)

    try:
        clean_pcd, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=float(std_ratio)
        )
        logger.debug(
            f"SOR Filtresi uygulandi: {initial_pts} -> {len(clean_pcd.points)} nokta "
            f"(Elenen: {initial_pts - len(clean_pcd.points)})"
        )
        return clean_pcd, np.asarray(inlier_indices)
    except Exception as e:
        logger.error(f"SOR filtresi calistirilirken hata olustu: {e}")
        raise RuntimeError(f"Statistical Outlier Removal basarisiz: {e}")


def remove_radius_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_points: int = 16,
    radius: float = 1.0
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Yaricap Tabanli Aykiri Deger Filtresi (Radius Outlier Removal - ROR).

    Belirli bir yaricaptaki kure icerisinde yeterli sayida komsusu bulunmayan
    izole noktalari ve agiz ici tukuruk parazitlerini eler.

    Args:
        pcd (o3d.geometry.PointCloud): Girdi nokta bulutu.
        nb_points (int): Yaricap icerisinde bulunmasi gereken asgari komsu sayisi.
        radius (float): Arama kuresi yaricapi (mm).

    Returns:
        Tuple[o3d.geometry.PointCloud, np.ndarray]: (Filtrelenmis_bulut, Secilen_indeksler).
    """
    validate_point_cloud_input(pcd, "pcd")

    if not isinstance(nb_points, int) or nb_points <= 0:
        raise ValueError(f"nb_points pozitif bir tam sayi olmalidir, alinan: {nb_points}")
    if not isinstance(radius, (int, float)) or radius <= 0:
        raise ValueError(f"radius pozitif bir sayi olmalidir, alinan: {radius}")

    initial_pts = len(pcd.points)
    try:
        clean_pcd, inlier_indices = pcd.remove_radius_outlier(
            nb_points=nb_points,
            radius=float(radius)
        )
        logger.debug(
            f"ROR Filtresi uygulandi: {initial_pts} -> {len(clean_pcd.points)} nokta "
            f"(Elenen: {initial_pts - len(clean_pcd.points)})"
        )
        return clean_pcd, np.asarray(inlier_indices)
    except Exception as e:
        logger.error(f"ROR filtresi calistirilirken hata olustu: {e}")
        raise RuntimeError(f"Radius Outlier Removal basarisiz: {e}")


def crop_bounding_box(
    pcd: o3d.geometry.PointCloud,
    min_bound: np.ndarray,
    max_bound: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    3D Uzamsal Sinirlayici Kutu (Bounding Box) Filtresi.
    Agiz ici tarama calisma hacmi disindaki alanlari keser.

    Args:
        pcd (o3d.geometry.PointCloud): Girdi nokta bulutu.
        min_bound (np.ndarray): [min_x, min_y, min_z] koordinatlari.
        max_bound (np.ndarray): [max_x, max_y, max_z] koordinatlari.

    Returns:
        o3d.geometry.PointCloud: Kirpilmis nokta bulutu.
    """
    validate_point_cloud_input(pcd, "pcd")

    min_b = np.asarray(min_bound, dtype=np.float64).reshape(3)
    max_b = np.asarray(max_bound, dtype=np.float64).reshape(3)

    if np.any(min_b >= max_b):
        raise ValueError(f"min_bound tum boyutlarda max_bound'dan kucuk olmalidir: {min_b} vs {max_b}")

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)
    cropped_pcd = pcd.crop(bbox)
    return cropped_pcd


def filter_dental_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: Optional[float] = None,
    sor_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
    ror_points: Optional[int] = None,
    ror_radius: Optional[float] = None
) -> o3d.geometry.PointCloud:
    """
    Dental 3D Nokta Bulutu Butunlesik Filtreleme Boru Hatti.
    Sirasiyla:
    1. Voxel Downsampling (Istege bagli)
    2. Statistical Outlier Removal (SOR)
    3. Radius Outlier Removal (Istege bagli ROR)

    Args:
        pcd (o3d.geometry.PointCloud): Ham nokta bulutu.
        voxel_size (float, optional): Seyreltme voxel boyutu (mm).
        sor_neighbors (int): SOR komsu sayisi.
        sor_std_ratio (float): SOR sapma carpani.
        ror_points (int, optional): ROR komsu sayisi.
        ror_radius (float, optional): ROR kure yaricapi.

    Returns:
        o3d.geometry.PointCloud: Temizlenmis dental nokta bulutu.
    """
    validate_point_cloud_input(pcd, "pcd")
    current_pcd = copy.deepcopy(pcd)

    # 1. Voxel Seyreltme
    if voxel_size is not None and voxel_size > 0:
        current_pcd = current_pcd.voxel_down_sample(voxel_size)

    # 2. SOR (Istatiksel Aykiri Deger Temizligi)
    if len(current_pcd.points) > sor_neighbors:
        current_pcd, _ = remove_statistical_outliers(
            current_pcd,
            nb_neighbors=sor_neighbors,
            std_ratio=sor_std_ratio
        )

    # 3. ROR (Yaricap Tabanli Temizlik)
    if ror_points is not None and ror_radius is not None and len(current_pcd.points) > ror_points:
        current_pcd, _ = remove_radius_outliers(
            current_pcd,
            nb_points=ror_points,
            radius=ror_radius
        )

    return current_pcd

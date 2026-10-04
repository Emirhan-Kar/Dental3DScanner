import os
import sys
import numpy as np
import open3d as o3d

# Proje kok dizinini sys.path'e ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.logger import get_logger
from utils.validator import get_safe_path

logger = get_logger(__name__)

# FDI 32 Dis + 1 Dis Eti Renk Paleti (Sabit seed ile tutarli renkler)
def get_fdi_color_palette() -> np.ndarray:
    """
    Her FDI sinifi icin deterministik ve gorsel olarak ayirt edilebilir bir renk paleti uretir.
    Sinif 0 (Dis Eti / Gingiva) icin anatomik pembe renk atanir.
    """
    palette = np.zeros((33, 3), dtype=np.float64)
    palette[0] = [0.85, 0.45, 0.55]  # Sinif 0: Gingiva / Dis Eti (Pembe)

    rng = np.random.RandomState(42)
    for i in range(1, 33):
        # Disler icin parlak, fark edilebilir renkler
        palette[i] = rng.uniform(0.15, 0.95, size=3)
    return palette


class DentalViewer:
    """
    3D Dis Nokta Bulutu ve Mesh Gorsellestirme Arayuzu (Open3D Tabanli).
    """
    def __init__(self, window_name: str = "Dental 3D Scanner Viewer", width: int = 1280, height: int = 720):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.palette = get_fdi_color_palette()

    def show_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        window_name: str = None,
        color: list = None,
        point_size: float = 2.0
    ):
        """
        Nokta bulutunu 3D ekranda interaktif gosterir.

        Args:
            pcd (o3d.geometry.PointCloud): Gosterilecek nokta bulutu.
            window_name (str, optional): Pencere basligi.
            color (list, optional): [R, G, B] formunda tekil renk (0.0 - 1.0).
            point_size (float): Nokta render boyutu.
        """
        if pcd is None or not isinstance(pcd, o3d.geometry.PointCloud) or len(pcd.points) == 0:
            logger.warning("Gorsellestirilecek nokta bulutu bos veya gecersiz.")
            return

        title = window_name or self.window_name
        logger.info(f"3D Nokta Bulutu penceresi aciliyor: {len(pcd.points)} nokta...")

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=self.width, height=self.height)

        pcd_copy = o3d.geometry.PointCloud(pcd)
        if color is not None:
            pcd_copy.paint_uniform_color(color)
        elif not pcd_copy.has_colors():
            pcd_copy.paint_uniform_color([0.7, 0.7, 0.7])  # Varsayilan gri

        vis.add_geometry(pcd_copy)

        render_opts = vis.get_render_option()
        render_opts.point_size = point_size
        render_opts.background_color = np.array([0.1, 0.1, 0.12])  # Koyu arka plan

        vis.run()
        vis.destroy_window()

    def show_segmented_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        predictions: np.ndarray,
        window_name: str = "AI Diş Segmentasyon Sonucu"
    ):
        """
        Tahmin edilen FDI etiketlerine gore renklendirilmis 3D dis modelini gosterir.

        Args:
            pcd (o3d.geometry.PointCloud): Nokta bulutu.
            predictions (np.ndarray): Her nokta icin sinif tahmini (N,).
            window_name (str): Pencere basligi.
        """
        if pcd is None or len(pcd.points) == 0:
            logger.warning("Segmentasyon gorsellestirme: Nokta bulutu bos.")
            return

        if len(predictions) != len(pcd.points):
            logger.warning(
                f"Tahmin boyutu ({len(predictions)}) nokta sayisiyla ({len(pcd.points)}) uyusmuyor."
            )

        pcd_colored = o3d.geometry.PointCloud(pcd)
        colors = np.zeros((len(predictions), 3), dtype=np.float64)

        for i, pred in enumerate(predictions):
            cls_idx = int(pred) if 0 <= int(pred) < len(self.palette) else 0
            colors[i] = self.palette[cls_idx]

        pcd_colored.colors = o3d.utility.Vector3dVector(colors)
        self.show_point_cloud(pcd_colored, window_name=window_name, point_size=3.0)

    def show_mesh(self, mesh_or_path, window_name: str = "3D Diş Modeli"):
        """3D mesh dosyasini veya TriangleMesh nesnesini gosterir."""
        if isinstance(mesh_or_path, str):
            safe_p = get_safe_path(mesh_or_path)
            try:
                mesh = o3d.io.read_triangle_mesh(safe_p)
            except Exception as e:
                logger.error(f"Mesh okunamadi: {e}")
                return
        else:
            mesh = mesh_or_path

        if mesh is None or len(mesh.vertices) == 0:
            logger.warning("Gosterilecek mesh bos.")
            return

        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        logger.info(f"3D Mesh penceresi aciliyor: {len(mesh.vertices)} kose, {len(mesh.triangles)} ucgen...")
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=window_name,
            width=self.width,
            height=self.height
        )

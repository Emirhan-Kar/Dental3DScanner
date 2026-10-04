import os
import sys
import numpy as np
import open3d as o3d

# Proje kok dizinini sys.path'e ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.logger import get_logger
from utils.config import load_config
from utils.validator import get_safe_path, validate_point_cloud

logger = get_logger(__name__)


class MeshGenerator:
    """
    3D Nokta Bulutundan Katı Diş Yüzeyi (Mesh) Rekonstrüksiyon Modülü.
    Dentistry CAD/CAM ve 3D baskı (STL / OBJ) için Poisson ve Ball-Pivoting algoritmalarını kullanır.
    """
    def __init__(
        self,
        method: str = "poisson",
        poisson_depth: int = 9,
        density_trim_percentile: float = 5.0,
        smooth_iterations: int = 3
    ):
        """
        Args:
            method (str): 'poisson' veya 'ball_pivoting'.
            poisson_depth (int): Poisson rekonstrüksiyon sekizli ağaç (octree) derinliği (6-12).
            density_trim_percentile (float): Düşük yoğunluklu dış yüzeyleri kırpma yüzdesi (0-50).
            smooth_iterations (int): Yüzey pürüzsüzleştirme iterasyon sayısı (>= 0).
        """
        if method not in ("poisson", "ball_pivoting"):
            raise ValueError(f"Desteklenmeyen rekonstrüksiyon yöntemi: '{method}'. 'poisson' veya 'ball_pivoting' olmalıdır.")

        if not isinstance(poisson_depth, int) or not (5 <= poisson_depth <= 13):
            raise ValueError(f"poisson_depth 5 ile 13 arasında bir tamsayı olmalıdır, alınan: {poisson_depth}")

        if not isinstance(density_trim_percentile, (int, float)) or not (0.0 <= density_trim_percentile < 50.0):
            raise ValueError(f"density_trim_percentile 0 ile 50 arasında olmalıdır, alınan: {density_trim_percentile}")

        if not isinstance(smooth_iterations, int) or smooth_iterations < 0:
            raise ValueError(f"smooth_iterations 0 veya pozitif bir tamsayı olmalıdır, alınan: {smooth_iterations}")

        self.method = method
        self.poisson_depth = poisson_depth
        self.density_trim_percentile = float(density_trim_percentile)
        self.smooth_iterations = smooth_iterations

    @classmethod
    def from_config(cls, config=None):
        """Merkezi konfigürasyondan MeshGenerator nesnesi oluşturur."""
        if config is None:
            config = load_config()

        mesh_cfg = getattr(config, "mesh_generator", None)
        if mesh_cfg:
            return cls(
                method=getattr(mesh_cfg, "method", "poisson"),
                poisson_depth=getattr(mesh_cfg, "poisson_depth", 9),
                density_trim_percentile=getattr(mesh_cfg, "density_trim_percentile", 5.0),
                smooth_iterations=getattr(mesh_cfg, "smooth_iterations", 3)
            )
        return cls()

    def generate_mesh(
        self,
        pcd: o3d.geometry.PointCloud,
        method: str = None,
        filter_outliers: bool = True
    ) -> o3d.geometry.TriangleMesh:
        """
        Nokta bulutunu katı 3D üçgen modele (TriangleMesh) dönüştürür.
        """
        val_res = validate_point_cloud(pcd, min_points=30)
        if not val_res.is_valid:
            raise ValueError(f"Mesh üretimi için nokta bulutu geçersiz: {val_res.error_message}")

        if filter_outliers and len(pcd.points) > 30:
            from processing.filters import filter_dental_point_cloud
            pcd = filter_dental_point_cloud(pcd, sor_neighbors=20, sor_std_ratio=2.0)

        selected_method = method or self.method

        if selected_method == "poisson":
            return self.reconstruct_poisson(pcd)
        elif selected_method == "ball_pivoting":
            return self.reconstruct_ball_pivoting(pcd)
        else:
            raise ValueError(f"Bilinmeyen yöntem: {selected_method}")

    def reconstruct_poisson(
        self,
        pcd: o3d.geometry.PointCloud,
        depth: int = None,
        density_trim_percentile: float = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Screened Poisson Yüzey Rekonstrüksiyonu uygular.
        Su geçirmez (watertight) ve pürüzsüz katı diş anatomisi modelleri üretir.
        """
        depth = depth or self.poisson_depth
        density_trim = density_trim_percentile if density_trim_percentile is not None else self.density_trim_percentile

        pcd_work = o3d.geometry.PointCloud(pcd)

        # Normalleri hazırla ve tutarlı yönlendir
        if not pcd_work.has_normals():
            logger.info("Nokta bulutu normalleri hesaplaniyor...")
            pcd_work.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30)
            )

        try:
            pcd_work.orient_normals_consistent_tangent_plane(k=15)
        except Exception:
            pass

        logger.info(f"Poisson Yuzey Rekonstruksiyonu hesaplaniyor (derinlik={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_work, depth=depth
        )

        # Dusuk yogunluklu (hayali dis hacim) kisimlari kirp
        if density_trim > 0 and len(densities) > 0:
            densities_arr = np.asarray(densities)
            density_threshold = np.percentile(densities_arr, density_trim)
            vertices_to_remove = densities_arr < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
            logger.info(f"Poisson dis hacim kirpildi (kirpma esigi: {density_threshold:.3f})")

        return self.smooth_and_clean_mesh(mesh)

    def reconstruct_ball_pivoting(
        self,
        pcd: o3d.geometry.PointCloud,
        radii: list = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Ball Pivoting Algorithm (BPA) ile yüzey oluşturur.
        Noktaların tam üzerinden geçen hassas üçgenleme sağlar.
        """
        pcd_work = o3d.geometry.PointCloud(pcd)

        if not pcd_work.has_normals():
            pcd_work.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30)
            )

        if radii is None:
            distances = pcd_work.compute_nearest_neighbor_distance()
            avg_dist = float(np.mean(distances)) if len(distances) > 0 else 0.5
            radii = [avg_dist * 1.5, avg_dist * 3.0, avg_dist * 5.0]

        logger.info(f"Ball Pivoting Yuzey Rekonstruksiyonu hesaplaniyor (radii={radii})...")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd_work, o3d.utility.DoubleVector(radii)
        )

        return self.smooth_and_clean_mesh(mesh)

    def smooth_and_clean_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        iterations: int = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Mesh geometrisini temizler (dejenere üçgenler, tekil kenarlar) ve yumuşatır.
        """
        if mesh is None or len(mesh.vertices) == 0:
            return mesh

        iterations = iterations if iterations is not None else self.smooth_iterations

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

        if iterations > 0:
            mesh = mesh.filter_smooth_simple(number_of_iterations=iterations)

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        logger.info(
            f"Kati Model hazirlandi: {len(mesh.vertices)} kose (vertex), "
            f"{len(mesh.triangles)} ucgen (triangle)."
        )
        return mesh

    def export_mesh(self, mesh: o3d.geometry.TriangleMesh, output_path: str) -> bool:
        """
        Oluşturulan yüzey modelini STL, OBJ veya PLY formatında diske kaydeder.
        """
        if mesh is None or len(mesh.vertices) == 0:
            logger.error("Kaydedilecek mesh nesnesi bos!")
            return False

        try:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            safe_out = get_safe_path(output_path)
            success = o3d.io.write_triangle_mesh(safe_out, mesh)
            if success:
                logger.info(f"Kati dis modeli (STL/OBJ) basariyla kaydedildi: {output_path}")
                return True
            else:
                logger.error(f"Open3D mesh dosyasını yazamadı: {output_path}")
                return False
        except Exception as e:
            logger.error(f"Mesh kaydedilirken hata oluştu ({output_path}): {e}")
            return False

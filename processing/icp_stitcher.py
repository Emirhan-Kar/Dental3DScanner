import copy
import numpy as np
import open3d as o3d
from utils.logger import get_logger

logger = get_logger(__name__)


class PointCloudStitcher:
    def __init__(self, voxel_size=0.4):
        """
        Güçlendirilmiş Dental ICP Stitcher (RANSAC Global + Point-to-Plane ICP).

        Args:
            voxel_size (float): Voxel grid boyutu (mm). Pozitif bir sayı olmalıdır.

        Raises:
            ValueError: voxel_size <= 0 ise
            TypeError: voxel_size sayısal bir değer değilse
        """
        if not isinstance(voxel_size, (int, float)):
            raise TypeError(f"voxel_size sayısal bir değer olmalıdır, alınan: {type(voxel_size).__name__}")
        if voxel_size <= 0:
            raise ValueError(f"voxel_size pozitif bir değer olmalıdır, alınan: {voxel_size}")

        self.voxel_size = float(voxel_size)
        self.global_pcd = o3d.geometry.PointCloud()

    @classmethod
    def from_config(cls, config=None):
        """Konfigurasyon nesnesinden veya config.yaml dosyasindan PointCloudStitcher olusturur."""
        if config is None:
            from utils.config import load_config
            config = load_config()
        stitch_cfg = config.icp_stitcher
        return cls(
            voxel_size=stitch_cfg.voxel_size
        )

    def _validate_pcd(self, pcd, name="pcd"):
        """Nokta bulutunun geçerliliğini denetler."""
        if pcd is None:
            raise TypeError(f"{name} None olamaz. Geçerli bir open3d.geometry.PointCloud nesnesi bekleniyor.")
        if not isinstance(pcd, o3d.geometry.PointCloud):
            raise TypeError(f"{name} tipi open3d.geometry.PointCloud olmalıdır, alınan: {type(pcd).__name__}")
        if len(pcd.points) == 0:
            raise ValueError(f"{name} boş nokta bulutu içeriyor (0 nokta).")

    def preprocess_and_compute_fpfh(self, pcd):
        """
        Noktaları seyreltir, normalleri ve FPFH geometrik özniteliklerini çıkarır.

        Args:
            pcd (o3d.geometry.PointCloud): Giriş nokta bulutu

        Returns:
            tuple: (pcd_down, fpfh)

        Raises:
            ValueError: Nokta sayısı yetersiz ise
            RuntimeError: Open3D işlemleri başarısız olursa
        """
        self._validate_pcd(pcd, "pcd")

        try:
            pcd_down = pcd.voxel_down_sample(self.voxel_size)
        except Exception as e:
            raise RuntimeError(f"Voxel downsampling işlemi başarısız: {e}")

        num_down_points = len(pcd_down.points)
        if num_down_points < 4:
            raise ValueError(
                f"Seyreltme sonrası nokta sayısı çok az ({num_down_points}). "
                f"voxel_size ({self.voxel_size}) diş modeline göre çok büyük olabilir."
            )

        try:
            # Normalleri hesapla
            pcd_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2.5, max_nn=30
                )
            )

            # FPFH (Fast Point Feature Histograms) özniteliklerini çıkar
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 5.0, max_nn=100
                )
            )
        except Exception as e:
            raise RuntimeError(f"Normal tahmini veya FPFH öznitelik çıkarımı başarısız: {e}")

        return pcd_down, fpfh

    def align_and_merge(self, source_pcd, target_pcd=None):
        """
        Önce RANSAC ile kaba hizalama, ardından Point-to-Plane ICP ile hassas kilitleme yapar.

        Args:
            source_pcd (o3d.geometry.PointCloud): Hizalanacak yeni nokta bulutu
            target_pcd (o3d.geometry.PointCloud, optional): Hedef model. None ise global_pcd kullanılır.

        Returns:
            np.ndarray: 4x4 homojen dönüşüm matrisi

        Raises:
            TypeError / ValueError: Girdi nokta bulutları geçersiz veya boş ise
            RuntimeError: Hizalama adımlarında kritik hata meydana gelirse
        """
        self._validate_pcd(source_pcd, "source_pcd")

        if target_pcd is None:
            if len(self.global_pcd.points) == 0:
                self.global_pcd = copy.deepcopy(source_pcd)
                return np.identity(4, dtype=np.float64)
            target_pcd = self.global_pcd
        else:
            self._validate_pcd(target_pcd, "target_pcd")

        # 1. Aşama: Özellik Çıkarımı
        source_down, source_fpfh = self.preprocess_and_compute_fpfh(source_pcd)
        target_down, target_fpfh = self.preprocess_and_compute_fpfh(target_pcd)

        # 2. Aşama: Global RANSAC Hizalama (Büyük açı ve kaymaları kurtarır)
        distance_threshold_ransac = self.voxel_size * 3.0
        init_transformation = np.identity(4, dtype=np.float64)

        try:
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh,
                mutual_filter=True,
                max_correspondence_distance=distance_threshold_ransac,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_ransac)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
            )
            if np.all(np.isfinite(result_ransac.transformation)):
                init_transformation = result_ransac.transformation
            else:
                logger.warning("RANSAC dönüşüm matrisinde geçersiz (NaN/Inf) değerler üretildi, Identity kullanılıyor.")
        except Exception as e:
            logger.warning(f"RANSAC küresel hizalama hatası ({e}), ICP doğrudan identity ile başlatılıyor.")

        # 3. Aşama: Hassas Point-to-Plane ICP (RANSAC matrisi başlangıç kabul edilir)
        distance_threshold_icp = self.voxel_size * 1.5
        try:
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down,
                distance_threshold_icp,
                init_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
            )
            final_trans = result_icp.transformation
            if not np.all(np.isfinite(final_trans)):
                raise ValueError("ICP sonucu elde edilen matris NaN veya Inf içeriyor.")
        except Exception as e:
            raise RuntimeError(f"Point-to-Plane ICP optimizasyonu başarısız: {e}")

        # 4. Kaynağı dönüştür ve ana modele ekle
        source_transformed = copy.deepcopy(source_pcd)
        source_transformed.transform(final_trans)

        self.global_pcd += source_transformed
        self.global_pcd = self.global_pcd.voxel_down_sample(self.voxel_size * 0.7)
        logger.info(f"Nokta bulutu basariyla hizalandi ve eklendi. Global nokta sayisi: {len(self.global_pcd.points)}")

        return final_trans

    def get_global_model(self):
        """Global birleşik modelin bir kopyasını döndürür."""
        return copy.deepcopy(self.global_pcd)

    def reset(self):
        """Global birleşik modeli sıfırlar."""
        self.global_pcd = o3d.geometry.PointCloud()
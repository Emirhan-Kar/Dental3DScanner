import copy
import numpy as np
import open3d as o3d
from utils.logger import get_logger

logger = get_logger(__name__)


class PointCloudStitcher:
    def __init__(self, voxel_size=0.4, ransac_distance_multiplier=3.0, icp_distance_multiplier=1.5,
                 icp_max_iterations=80, ransac_max_iterations=100000, icp_method="point_to_plane",
                 use_global_registration=True, seed=None, registration_mode=None,
                 min_fitness=0.45, max_rmse_ratio=0.55, ransac_retries=3):
        """
        Güçlendirilmiş Dental ICP Stitcher (RANSAC Global + Point-to-Plane ICP).

        Args:
            voxel_size (float): Voxel grid boyutu (mm). Pozitif bir sayı olmalıdır.
            ransac_distance_multiplier (float): RANSAC eşiği = voxel_size * çarpan.
            icp_distance_multiplier (float): ICP eşleşme mesafesi = voxel_size * çarpan.
            icp_max_iterations (int): ICP maksimum iterasyon.
            ransac_max_iterations (int): RANSAC maksimum iterasyon.
            icp_method (str): "point_to_plane" veya "point_to_point".
            use_global_registration (bool): False ise RANSAC atlanır, ICP doğrudan
                verilen başlangıç konumundan (identity) başlar.
            seed (int, optional): RANSAC için rastgelelik tohumu (tekrarlanabilir deneyler).
            registration_mode (str, optional): "global" (her zaman RANSAC + ICP), "local" (sadece ICP,
                başlangıç tahmininden) veya "hybrid" (önce ICP ile takip; sonuç kalite testini
                geçemezse RANSAC ile yeniden konumlanma). Verilmezse use_global_registration'a göre
                "global"/"local" seçilir.
            min_fitness (float): Hizalamanın geçerli sayılması için minimum örtüşen nokta oranı.
            max_rmse_ratio (float): Geçerlilik için inlier RMSE <= oran * ICP eşleşme mesafesi.
            ransac_retries (int): Hibrit modda RANSAC'ın farklı tohumlarla deneme sayısı.

        Raises:
            ValueError: voxel_size <= 0 ise
            TypeError: voxel_size sayısal bir değer değilse
        """
        if not isinstance(voxel_size, (int, float)):
            raise TypeError(f"voxel_size sayısal bir değer olmalıdır, alınan: {type(voxel_size).__name__}")
        if voxel_size <= 0:
            raise ValueError(f"voxel_size pozitif bir değer olmalıdır, alınan: {voxel_size}")

        if icp_method not in ("point_to_plane", "point_to_point"):
            raise ValueError(f"icp_method 'point_to_plane' veya 'point_to_point' olmalıdır, alınan: {icp_method}")

        self.voxel_size = float(voxel_size)
        self.ransac_distance_multiplier = float(ransac_distance_multiplier)
        self.icp_distance_multiplier = float(icp_distance_multiplier)
        self.icp_max_iterations = int(icp_max_iterations)
        self.ransac_max_iterations = int(ransac_max_iterations)
        self.icp_method = icp_method
        self.use_global_registration = bool(use_global_registration)
        if registration_mode is None:
            registration_mode = "global" if self.use_global_registration else "local"
        if registration_mode not in ("global", "local", "hybrid"):
            raise ValueError(f"registration_mode 'global', 'local' veya 'hybrid' olmalıdır, alınan: {registration_mode}")
        self.registration_mode = registration_mode
        self.min_fitness = float(min_fitness)
        self.max_rmse_ratio = float(max_rmse_ratio)
        self.ransac_retries = int(ransac_retries)
        self.seed = seed
        self.last_result = {}
        self.global_pcd = o3d.geometry.PointCloud()

    @classmethod
    def from_config(cls, config=None):
        """Konfigurasyon nesnesinden veya config.yaml dosyasindan PointCloudStitcher olusturur."""
        if config is None:
            from utils.config import load_config
            config = load_config()
        c = config.icp_stitcher
        return cls(
            voxel_size=c.voxel_size,
            ransac_distance_multiplier=getattr(c, "ransac_distance_multiplier", 3.0),
            icp_distance_multiplier=getattr(c, "icp_distance_multiplier", 1.5),
            icp_max_iterations=getattr(c, "icp_max_iterations", 80),
            ransac_max_iterations=getattr(c, "ransac_max_iterations", 100000),
            registration_mode=getattr(c, "registration_mode", "hybrid"),
            min_fitness=getattr(c, "min_fitness", 0.45),
            max_rmse_ratio=getattr(c, "max_rmse_ratio", 0.55),
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

        final_trans = self.register(source_pcd, target_pcd)

        # 4. Kaynağı dönüştür ve ana modele ekle
        source_transformed = copy.deepcopy(source_pcd)
        source_transformed.transform(final_trans)

        self.global_pcd += source_transformed
        self.global_pcd = self.global_pcd.voxel_down_sample(self.voxel_size * 0.7)
        logger.info(f"Nokta bulutu basariyla hizalandi ve eklendi. Global nokta sayisi: {len(self.global_pcd.points)}")

        return final_trans

    def _ransac(self, src_down, src_fpfh, tgt_down, tgt_fpfh, tohum):
        if tohum is not None:
            o3d.utility.random.seed(int(tohum))
        esik = self.voxel_size * self.ransac_distance_multiplier
        try:
            r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_down, tgt_down, src_fpfh, tgt_fpfh,
                mutual_filter=True,
                max_correspondence_distance=esik,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(esik)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(self.ransac_max_iterations, 0.999)
            )
            if np.all(np.isfinite(r.transformation)) and r.fitness > 0:
                return r.transformation
        except Exception as e:
            logger.warning(f"RANSAC küresel hizalama hatası: {e}")
        return None

    def _icp(self, src_down, tgt_down, init):
        tahminci = (o3d.pipelines.registration.TransformationEstimationPointToPlane()
                    if self.icp_method == "point_to_plane"
                    else o3d.pipelines.registration.TransformationEstimationPointToPoint())
        try:
            r = o3d.pipelines.registration.registration_icp(
                src_down, tgt_down, self.voxel_size * self.icp_distance_multiplier, init, tahminci,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_max_iterations))
        except Exception as e:
            raise RuntimeError(f"ICP optimizasyonu başarısız: {e}")
        if not np.all(np.isfinite(r.transformation)):
            raise RuntimeError("ICP sonucu elde edilen matris NaN veya Inf içeriyor.")
        return r

    def _gecerli_mi(self, r):
        """Hizalama kalite testi: yeterli örtüşme ve düşük kalıntı hata.

        Deneylerde (data/four, 12 hasta) doğru hizalamalarda inlier RMSE <= ~0.30 mm,
        yanlışlarda >= ~0.36 mm çıktı (voxel 0.4 mm, ICP mesafesi 0.6 mm).
        """
        return (r.fitness >= self.min_fitness and
                r.inlier_rmse <= self.max_rmse_ratio * self.voxel_size * self.icp_distance_multiplier)

    @staticmethod
    def _duzeltme_buyuklugu(T, T0, merkez):
        """T'nin T0'a gore farki: kaynak merkezinin yer degistirmesi (mm) ve donme acisi (derece)."""
        D = T @ np.linalg.inv(T0)
        c = np.append(merkez, 1.0)
        oteleme = float(np.linalg.norm((D @ c)[:3] - merkez))
        aci = float(np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))))
        return oteleme, aci

    def register(self, source_pcd, target_pcd, init_transformation=None, max_correction_mm=None,
                 max_correction_deg=None):
        """Kaynağı hedefe hizalayan 4x4 dönüşümü bulur (modeli değiştirmez).

        max_correction_mm / max_correction_deg (makullük kapısı): Başlangıç tahmininin (ör. IMU)
        hatasının sınırlı olduğu biliniyorsa, bundan çok büyük düzeltme öneren adaylar
        (benzer görünen başka bir dişe oturma gibi) reddedilir.

        Sonuç ayrıntıları (fitness, inlier RMSE, RANSAC kullanıldı mı, geçerli mi)
        self.last_result'ta saklanır.
        """
        self._validate_pcd(source_pcd, "source_pcd")
        self._validate_pcd(target_pcd, "target_pcd")
        init = np.identity(4, dtype=np.float64) if init_transformation is None else np.asarray(init_transformation, dtype=np.float64)

        if self.registration_mode == "local":
            src_down = source_pcd.voxel_down_sample(self.voxel_size)
            tgt_down = target_pcd.voxel_down_sample(self.voxel_size)
            for p in (src_down, tgt_down):
                p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2.5, max_nn=30))
        else:
            src_down, src_fpfh = self.preprocess_and_compute_fpfh(source_pcd)
            tgt_down, tgt_fpfh = self.preprocess_and_compute_fpfh(target_pcd)

        adaylar = []   # (sonuc, ransac_kullanildi)
        if self.registration_mode in ("local", "hybrid"):
            r = self._icp(src_down, tgt_down, init)
            adaylar.append((r, False))

        if self.registration_mode == "global" or (self.registration_mode == "hybrid" and not self._gecerli_mi(adaylar[0][0])):
            deneme = 1 if self.registration_mode == "global" else max(1, self.ransac_retries)
            for k in range(deneme):
                tohum = None if self.seed is None else self.seed + k
                T0 = self._ransac(src_down, src_fpfh, tgt_down, tgt_fpfh, tohum)
                if T0 is None:
                    if self.registration_mode == "global":
                        logger.warning("RANSAC geçerli bir dönüşüm bulamadı, ICP başlangıç tahminiyle başlatılıyor.")
                        adaylar.append((self._icp(src_down, tgt_down, init), False))
                    continue
                r = self._icp(src_down, tgt_down, T0)
                adaylar.append((r, True))
                if self._gecerli_mi(r):
                    break

        # Makullük kapısı: izin verilen düzeltmeden büyük adayları ele
        reddedilen = 0
        if max_correction_mm is not None or max_correction_deg is not None:
            merkez = np.asarray(source_pcd.get_center())
            uygun = []
            for r, rk in adaylar:
                ot, ac = self._duzeltme_buyuklugu(r.transformation, init, merkez)
                if (max_correction_mm is None or ot <= max_correction_mm) and \
                   (max_correction_deg is None or ac <= max_correction_deg):
                    uygun.append((r, rk))
                else:
                    reddedilen += 1
            if not uygun:
                self.last_result = {"fitness": 0.0, "inlier_rmse": float("inf"), "ransac_used": False,
                                    "valid": False, "candidates": len(adaylar), "rejected_by_gate": reddedilen}
                return init
            adaylar = uygun

        # En iyi aday: once kalite testini gecenler, sonra en yuksek fitness / en dusuk RMSE
        en_iyi, ransac_ok = max(adaylar, key=lambda a: (self._gecerli_mi(a[0]), a[0].fitness - a[0].inlier_rmse))
        if self.registration_mode == "hybrid" and not self._gecerli_mi(en_iyi):
            logger.warning(f"Hizalama kalite testini geçemedi (fitness={en_iyi.fitness:.2f}, "
                           f"rmse={en_iyi.inlier_rmse:.3f}); en iyi aday kullanılıyor.")
        self.last_result = {"fitness": float(en_iyi.fitness), "inlier_rmse": float(en_iyi.inlier_rmse),
                            "ransac_used": bool(ransac_ok), "valid": bool(self._gecerli_mi(en_iyi)),
                            "candidates": len(adaylar), "rejected_by_gate": reddedilen}
        return en_iyi.transformation

    def get_global_model(self):
        """Global birleşik modelin bir kopyasını döndürür."""
        return copy.deepcopy(self.global_pcd)

    def reset(self):
        """Global birleşik modeli sıfırlar."""
        self.global_pcd = o3d.geometry.PointCloud()
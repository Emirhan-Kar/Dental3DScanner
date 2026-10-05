import os
import sys
import time
import argparse
import numpy as np
import cv2
import open3d as o3d

# Proje kok dizinini sys.path'e ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from processing.laser_detector import LaserDetector
from processing.triangulation import Triangulator
from processing.icp_stitcher import PointCloudStitcher
from processing.mesh_generator import MeshGenerator
from ui.viewer import DentalViewer
from utils.logger import setup_logger, get_logger
from utils.config import load_config, get_base_parser
from utils.validator import get_safe_path
from utils.video_stream import ThreadedVideoStream

logger = get_logger(__name__)


class DentalScannerPipeline:
    """
    Agiz Ici 3D Dental Tarama Ana Motoru (End-to-End Orchestrator).
    Akis: Video/Kamera Karesi -> Lazer Tespiti -> 3D Nirengi (Triangulation) -> ICP Hizalama -> Katı Model Rekonstrüksiyonu (STL/OBJ).
    """
    def __init__(
        self,
        laser_detector: LaserDetector = None,
        triangulator: Triangulator = None,
        stitcher: PointCloudStitcher = None,
        mesh_generator: MeshGenerator = None,
        config = None
    ):
        if config is None:
            config = load_config()

        self.config = config
        self.detector = laser_detector or LaserDetector.from_config(config)
        self.triangulator = triangulator or Triangulator.from_config(config)
        self.stitcher = stitcher or PointCloudStitcher.from_config(config)
        self.mesh_generator = mesh_generator or MeshGenerator.from_config(config)
        self.viewer = DentalViewer(window_name="Dental Scanner 3D Model")

        self.processed_frames = 0
        self.total_laser_points_detected = 0

    @classmethod
    def from_config(cls, config=None):
        """Konfigurasyon dosyasindan butunlesik tarayici motoru olusturur."""
        if config is None:
            config = load_config()
        return cls(config=config)

    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Tek bir video karesini isler:
        1. 2D Lazer tespiti (subpixel)
        2. 3D Koordinat hesaplama (triangulation)
        3. Nokta bulutu birlestirme (ICP)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {"status": "empty_frame", "points_3d": np.empty((0, 3)), "global_points": 0}

        self.processed_frames += 1

        # 1. Adim: Lazer Cizgisi Tespiti
        peaks, mask = self.detector.detect_subpixel_peaks(frame_bgr)
        self.total_laser_points_detected += len(peaks)

        if not peaks:
            return {
                "status": "no_laser",
                "peaks": [],
                "mask": mask,
                "points_3d": np.empty((0, 3)),
                "global_points": len(self.stitcher.global_pcd.points)
            }

        # 2. Adim: 2D Pikselleri 3D Uzaya Donustur (Triangulation)
        points_3d = self.triangulator.pixels_to_3d(peaks)

        # 3. Adim: 3D Dilimi Global Modele Ekle (ICP Stitching)
        if len(points_3d) >= 10:
            slice_pcd = o3d.geometry.PointCloud()
            slice_pcd.points = o3d.utility.Vector3dVector(points_3d)
            try:
                self.stitcher.align_and_merge(slice_pcd)
            except Exception as e:
                logger.warning(f"Kare #{self.processed_frames} hizalama uyarisi: {e}")

        return {
            "status": "success",
            "peaks": peaks,
            "mask": mask,
            "points_3d": points_3d,
            "global_points": len(self.stitcher.global_pcd.points)
        }

    def run_video(
        self,
        video_source,
        max_frames: int = None,
        show_2d: bool = False,
        save_output: str = "output/scan_result.ply",
        threaded: bool = True
    ):
        """
        Bir video dosyasindan veya canli kamera akisindan tarama gerceklestirir.

        Args:
            video_source: Kamera indeksi (int) veya video dosya yolu (str).
            max_frames (int, optional): Maksimum islenecek kare adedi.
            show_2d (bool): 2D gercek zamanli monitor penceresi gosterimi.
            save_output (str): Kaydedilecek PLY nokta bulutu yolu.
            threaded (bool): Asenkron / coklu is parcacikli kare yakalama modu.
        """
        logger.info(f"Tarama akisi baslatiliyor: {video_source} (Asenkron: {threaded})")

        start_time = time.time()
        frame_idx = 0

        if threaded:
            stream = ThreadedVideoStream(src=video_source)
            if not stream.opened:
                logger.error(f"Video kaynagi acilamadi: {video_source}")
                return None
            stream.start()
            logger.info(f"Asenkron video akisi hazir (FPS: {stream.fps:.1f}, Cozunurluk: {stream.width}x{stream.height})")

            try:
                while stream.is_opened():
                    if max_frames and frame_idx >= max_frames:
                        logger.info(f"Maksimum kare limitine ulasildi ({max_frames}).")
                        break

                    ret, frame = stream.read(timeout=0.5)
                    if not ret or frame is None:
                        break

                    frame_idx += 1
                    result = self.process_frame(frame)

                    # 2D Takip ve Bilgi Ekrani
                    if show_2d:
                        vis_frame = self.detector.draw_peaks(frame, result.get("peaks", []))
                        global_pts = result.get("global_points", 0)
                        fps = frame_idx / (time.time() - start_time + 1e-6)

                        cv2.putText(vis_frame, f"Kare: {frame_idx} | FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(vis_frame, f"Lazer Nokta: {len(result.get('peaks', []))} | Global 3D: {global_pts}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(vis_frame, "Cikis: Q tusuna basin", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                        cv2.imshow("Dental Scanner 2D Monitor", vis_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            logger.info("Kullanici tarafindan tarama durduruldu (Q).")
                            break

                    if frame_idx % 20 == 0:
                        logger.info(f"Kare #{frame_idx} | Global Nokta Sayisi: {len(self.stitcher.global_pcd.points)}")

            finally:
                stream.stop()
                if show_2d:
                    cv2.destroyAllWindows()
        else:
            try:
                cam_idx = int(video_source)
                cap = cv2.VideoCapture(cam_idx)
            except (ValueError, TypeError):
                cap = cv2.VideoCapture(str(video_source))

            if not cap.isOpened():
                logger.error(f"Video kaynagi acilamadi: {video_source}")
                return None

            total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video baglantisi kuruldu. Toplam kare: {total_input_frames if total_input_frames > 0 else 'Canli Akis'}")

            try:
                while cap.isOpened():
                    if max_frames and frame_idx >= max_frames:
                        logger.info(f"Maksimum kare limitine ulasildi ({max_frames}).")
                        break

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    frame_idx += 1
                    result = self.process_frame(frame)

                    # 2D Takip ve Bilgi Ekrani
                    if show_2d:
                        vis_frame = self.detector.draw_peaks(frame, result.get("peaks", []))
                        global_pts = result.get("global_points", 0)
                        fps = frame_idx / (time.time() - start_time + 1e-6)

                        cv2.putText(vis_frame, f"Kare: {frame_idx} | FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(vis_frame, f"Lazer Nokta: {len(result.get('peaks', []))} | Global 3D: {global_pts}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(vis_frame, "Cikis: Q tusuna basin", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                        cv2.imshow("Dental Scanner 2D Monitor", vis_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            logger.info("Kullanici tarafindan tarama durduruldu (Q).")
                            break

                    if frame_idx % 20 == 0:
                        logger.info(f"Kare #{frame_idx} | Global Nokta Sayisi: {len(self.stitcher.global_pcd.points)}")

            finally:
                cap.release()
                if show_2d:
                    cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        logger.info(f"Tarama bitti! {frame_idx} kare, {elapsed:.2f} saniyede islendi ({frame_idx/(elapsed+1e-6):.1f} FPS).")
        logger.info(f"Uretilen toplam global nokta sayisi: {len(self.stitcher.global_pcd.points)}")

        if save_output:
            self.export_global_model(save_output)

        return self.stitcher.get_global_model()

    def run_simulation(
        self,
        num_slices: int = 80,
        show_2d: bool = False,
        save_output: str = "output/simulation_scan.ply"
    ):
        """
        Fiziksel donanim veya video olmadiginda tum pipeline'i dogrulamak icin
        anatomik olarak dogru 3D dis modeli ureten gelismis simulasyon.

        Yontem:
        - Parametrik U-seklinde dental ark egrisi olusturur.
        - 16 dis, herbiri anatomik boyutlarla (mesio-distal, bukko-lingual
          genislik ve kron yuksekligi) ayri ayri 3D hacim olarak modellenir.
        - Disler, yuvarlatilmis kutu (superellipsoid-benzeri) seklinde uretilir.
        - Kalin dis eti (gingiva) kaidesi tum disleri birbirine baglar.
        - Ground-truth pozisyonlar bilindiginden ICP atlanir.
        - Sonucta Poisson yuzey rekonstruksiyonuna uygun yogun bir nokta
          bulutu elde edilir.
        """
        logger.info(f"Sentetik 3D Dental Tarama Simulasyonu baslatiliyor ({num_slices} dilim)...")

        all_points = []
        rng = np.random.RandomState(42)

        # ─── Dental Ark Egrisi Parametreleri (mm) ───
        arch_half_w = 23.0    # Arkin yari genisligi
        arch_depth  = 28.0    # On-arka derinlik (on dislerden arka dislere)
        gum_h       = 6.0     # Dis eti kaide yuksekligi

        # ─── 16 Dis Anatomik Tanimi ───
        # (t_pozisyon, MD_genislik, BL_genislik, kron_yuksekligi, okluzal_duzluk)
        # t: ark egrisi uzerinde pozisyon [-1, +1], MD: mesio-distal, BL: bukko-lingual
        teeth = [
            # Sag taraf (arkadan one)
            (-0.93, 4.8, 5.5, 3.5, 0.8),    # Sag 3. Molar (20 yas disi)
            (-0.80, 5.2, 6.0, 4.2, 0.8),    # Sag 2. Molar
            (-0.67, 5.8, 6.5, 4.8, 0.7),    # Sag 1. Molar
            (-0.55, 3.8, 4.8, 5.2, 0.5),    # Sag 2. Premolar
            (-0.44, 3.5, 4.2, 5.8, 0.4),    # Sag 1. Premolar
            (-0.31, 3.8, 4.0, 7.2, 0.2),    # Sag Kanin (kose disi)
            (-0.17, 3.2, 3.2, 6.8, 0.15),   # Sag Lateral Kesici
            (-0.05, 4.5, 3.5, 8.0, 0.1),    # Sag Santral Kesici
            # Sol taraf (onden arkaya)
            ( 0.05, 4.5, 3.5, 8.0, 0.1),    # Sol Santral Kesici
            ( 0.17, 3.2, 3.2, 6.8, 0.15),   # Sol Lateral Kesici
            ( 0.31, 3.8, 4.0, 7.2, 0.2),    # Sol Kanin
            ( 0.44, 3.5, 4.2, 5.8, 0.4),    # Sol 1. Premolar
            ( 0.55, 3.8, 4.8, 5.2, 0.5),    # Sol 2. Premolar
            ( 0.67, 5.8, 6.5, 4.8, 0.7),    # Sol 1. Molar
            ( 0.80, 5.2, 6.0, 4.2, 0.8),    # Sol 2. Molar
            ( 0.93, 4.8, 5.5, 3.5, 0.8),    # Sol 3. Molar
        ]

        def arch_xz(t):
            """Ark egrisi uzerinde (x, z) pozisyonu"""
            return t * arch_half_w, arch_depth * t ** 2

        def arch_frame(t):
            """Ark egrisinin t noktasindaki teget ve normal vektorleri"""
            tx, tz = arch_half_w, 2.0 * arch_depth * t
            length = np.sqrt(tx ** 2 + tz ** 2) + 1e-8
            tang = np.array([tx / length, 0.0, tz / length])
            norm = np.array([-tz / length, 0.0, tx / length])
            return tang, norm

        up = np.array([0.0, 1.0, 0.0])

        # ─── Her Dis Icin 3D Hacimsel Nokta Bulutu Uret ───
        pts_per_tooth = max(num_slices * 8, 500)

        for tooth_idx, (t_pos, md, bl, crown_h, flatness) in enumerate(teeth):
            cx, cz = arch_xz(t_pos)
            tang, norm = arch_frame(t_pos)
            center = np.array([cx, 0.0, cz])
            hmd, hbl = md / 2.0, bl / 2.0

            for _ in range(pts_per_tooth):
                # u: mesio-distal [-1,1], v: bukko-lingual [-1,1], w: yukseklik [0,1]
                u = rng.uniform(-1, 1)
                v = rng.uniform(-1, 1)
                w = rng.uniform(0, 1)

                # Yuzey onceliklendirmesi (%60 yuzey noktasi -> daha iyi mesh)
                if rng.uniform() < 0.60:
                    face = rng.randint(5)
                    if face == 0:
                        u = rng.choice([-1.0, 1.0])
                    elif face == 1:
                        v = rng.choice([-1.0, 1.0])
                    elif face == 2:
                        w = 1.0
                    elif face == 3:
                        u = rng.choice([-1.0, 1.0]) * rng.uniform(0.85, 1.0)
                        v *= rng.uniform(0.5, 1.0)
                    else:
                        v = rng.choice([-1.0, 1.0]) * rng.uniform(0.85, 1.0)
                        w = rng.uniform(0.5, 1.0)

                # Kron yukari dogru hafif daralir
                taper = 1.0 - 0.15 * w

                # Okluzal yuzey detayi: molar dislerde cuspler (cikinti)
                occlusal_mod = 0.0
                if w > 0.90 and flatness > 0.3:
                    cusp_freq = 3.0 if flatness > 0.6 else 2.0
                    occlusal_mod = 0.8 * np.sin(u * cusp_freq * np.pi) * np.sin(v * cusp_freq * np.pi)

                # Lokal koordinatlarda pozisyon
                lx = u * hmd * taper
                lz = v * hbl * taper
                ly = gum_h + w * crown_h + occlusal_mod

                # Kenar yuvarlama (koseler iceri cekilir)
                corner_r2 = u ** 2 + v ** 2
                if corner_r2 > 1.5:
                    scale = np.sqrt(1.5 / corner_r2)
                    lx *= scale
                    lz *= scale

                # Dunya koordinatlarina donustur
                pt = center + tang * lx + norm * lz + up * ly
                pt += rng.normal(0, 0.04, 3)  # tarayici gurultusu
                all_points.append(tuple(pt))

            # 2D gorsellestirme (istege bagli — dis bazli ilerleme)
            if show_2d:
                h_img, w_img = 480, 720
                vis_frame = np.zeros((h_img, w_img, 3), dtype=np.uint8)
                # Tum mevcut noktalari ciz (okluzal gorunum — ustunden bakis)
                for pt in all_points:
                    u_px = int(w_img / 2 + pt[0] * 7)
                    v_px = int(h_img / 2 + pt[2] * 5 - 60)
                    if 0 <= u_px < w_img and 0 <= v_px < h_img:
                        # Yukseklige gore renk (mavi=dis eti, sari=dis kronu)
                        brightness = min(255, int(80 + pt[1] * 15))
                        cv2.circle(vis_frame, (u_px, v_px), 1, (0, brightness, 255), -1)
                pct = int((tooth_idx + 1) / len(teeth) * 100)
                cv2.putText(vis_frame, f"Dis {tooth_idx+1}/{len(teeth)} (%{pct})",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Toplam Nokta: {len(all_points)}",
                            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.imshow("Dental Scanner 2D Monitor (Simulasyon)", vis_frame)
                if (cv2.waitKey(80) & 0xFF) == ord('q'):
                    break

        # ─── Dis Eti (Gingiva) Kaidesi ───
        gum_steps = num_slices * 3
        pts_per_gum = max(25, num_slices // 2)

        for i in range(gum_steps):
            t = -1.0 + 2.0 * i / max(gum_steps - 1, 1)
            cx, cz = arch_xz(t)
            tang, norm = arch_frame(t)
            center = np.array([cx, 0.0, cz])

            # En yakin disin BL genisligine gore dis eti genisligi
            gum_bl = 6.0
            for t_pos, _, bl_w, _, _ in teeth:
                if abs(t - t_pos) < 0.065:
                    gum_bl = bl_w * 0.5 + 2.5
                    break

            for _ in range(pts_per_gum):
                # Dis eti kesiti: yari-eliptik profil (U-sekli)
                angle = rng.uniform(0, np.pi)
                bl_offset = gum_bl * np.cos(angle)
                h_offset = gum_h * np.sin(angle)

                pt = center + norm * bl_offset + up * h_offset
                pt += rng.normal(0, 0.06, 3)
                all_points.append(tuple(pt))

            # 2D gorsellestirme (dis eti ilerleme)
            if show_2d and i % max(1, gum_steps // 8) == 0:
                h_img, w_img = 480, 720
                vis_frame = np.zeros((h_img, w_img, 3), dtype=np.uint8)
                for pt in all_points:
                    u_px = int(w_img / 2 + pt[0] * 7)
                    v_px = int(h_img / 2 + pt[2] * 5 - 60)
                    if 0 <= u_px < w_img and 0 <= v_px < h_img:
                        brightness = min(255, int(80 + pt[1] * 15))
                        cv2.circle(vis_frame, (u_px, v_px), 1, (0, brightness, 255), -1)
                cv2.putText(vis_frame, f"Dis Eti Kaidesi Olusturuluyor...",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2)
                cv2.putText(vis_frame, f"Toplam Nokta: {len(all_points)}",
                            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.imshow("Dental Scanner 2D Monitor (Simulasyon)", vis_frame)
                cv2.waitKey(40)

        if show_2d:
            cv2.destroyAllWindows()

        # ─── Nokta Bulutunu Olustur ───
        points_array = np.array(all_points, dtype=np.float64)
        global_pcd = o3d.geometry.PointCloud()
        global_pcd.points = o3d.utility.Vector3dVector(points_array)

        # Normalleri hesapla (mesh uretimi icin gerekli)
        global_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
        )

        # Hafif voxel downsampling (tekrarlanan noktalari temizle)
        global_pcd = global_pcd.voxel_down_sample(voxel_size=0.15)

        # Stitcher'in global modeline ata (mesh uretimi ve export icin)
        self.stitcher.global_pcd = global_pcd

        logger.info(f"Simulasyon tamamlandi! Global Nokta Sayisi: {len(global_pcd.points)}")

        if save_output:
            self.export_global_model(save_output)

        return self.stitcher.get_global_model()

    def generate_and_export_mesh(
        self,
        output_path: str = "output/dental_model.stl",
        method: str = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Mevcut birleşik nokta bulutundan katı yüzey üçgen modeli (STL/OBJ) üretir ve kaydeder.
        """
        global_pcd = self.stitcher.get_global_model()
        if len(global_pcd.points) < 30:
            logger.warning(f"Mesh uretimi icin nokta sayisi yetersiz: {len(global_pcd.points)} < 30")
            return None

        logger.info(f"Kati model uretimi baslatiliyor ({len(global_pcd.points)} nokta)...")
        mesh = self.mesh_generator.generate_mesh(global_pcd, method=method)

        if output_path:
            self.mesh_generator.export_mesh(mesh, output_path)

        return mesh

    def export_global_model(self, output_path: str):
        """Olusan birlesik nokta bulutunu PLY veya PCD olarak kaydeder."""
        global_pcd = self.stitcher.get_global_model()
        if len(global_pcd.points) == 0:
            logger.warning("Kaydedilecek nokta bulutu bulunamadi (0 nokta).")
            return False

        try:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            safe_out = get_safe_path(output_path)
            o3d.io.write_point_cloud(safe_out, global_pcd)
            logger.info(f"3D Nokta Bulutu basariyla kaydedildi: {output_path} ({len(global_pcd.points)} nokta)")
            return True
        except Exception as e:
            logger.error(f"Model kaydedilemedi ({output_path}): {e}")
            return False


if __name__ == "__main__":
    setup_logger()
    parser = get_base_parser(description="Dental 3D Scanner - Agiz Ici Tarama ve Rekonstruksiyon Motoru")
    parser.add_argument("--input", type=str, default=None, help="Video dosyasi yolu veya canli kamera indeksi (orn: 0)")
    parser.add_argument("--output", type=str, default="output/scan_result.ply", help="Cikti nokta bulutu dosyasi (.ply)")
    parser.add_argument("--export-mesh", type=str, default=None, help="Cikti katı dis modeli dosyasi (.stl veya .obj)")
    parser.add_argument("--mesh-method", type=str, choices=["poisson", "ball_pivoting"], default=None, help="Yuzey rekonstruksiyon yontemi")
    parser.add_argument("--show-2d", action="store_true", help="2D Lazer algilama monitor penceresini goster")
    parser.add_argument("--show-3d", action="store_true", help="Tarama bitince 3D Open3D penceresini ac")
    parser.add_argument("--simulation", action="store_true", help="Sentetik dis tarama simulasyonu calistir")
    parser.add_argument("--max-frames", type=int, default=None, help="Maksimum islenecek kare sayisi")
    parser.add_argument("--slices", type=int, default=150, help="Simulasyon dilim sayisi (varsayilan: 80, daha fazla = daha detayli model)")
    parser.add_argument("--no-threaded", action="store_true", help="Coklu is parcacikli video okuyucuyu devre disi birak")

    args = parser.parse_args()
    cfg = load_config(args.config)

    pipeline = DentalScannerPipeline.from_config(cfg)

    # Girdi kaynagi secimi
    if args.simulation or (args.input is None and not os.path.exists("data/TestVideo")):
        logger.info("Video kaynagi verilmedi, otomatik Sentetik Simulasyon modu baslatiliyor...")
        final_model = pipeline.run_simulation(
            num_slices=args.slices,
            show_2d=args.show_2d,
            save_output=args.output
        )
    else:
        video_src = args.input
        if not video_src:
            import glob
            video_files = glob.glob("data/TestVideo/*.mp4") + glob.glob("data/TestVideo/*.avi")
            if video_files:
                video_src = video_files[0]
            else:
                video_src = 0

        final_model = pipeline.run_video(
            video_source=video_src,
            max_frames=args.max_frames,
            show_2d=args.show_2d,
            save_output=args.output,
            threaded=not args.no_threaded
        )

    # Katı Diş Modeli Üretimi (STL / OBJ)
    generated_mesh = None
    if args.export_mesh:
        generated_mesh = pipeline.generate_and_export_mesh(
            output_path=args.export_mesh,
            method=args.mesh_method
        )

    # 3D Gorsellestirme
    if args.show_3d:
        viewer = DentalViewer(window_name="Nihai 3D Dis Modeli")
        if generated_mesh is not None:
            viewer.show_mesh(generated_mesh, window_name="3D Katı Diş Yüzeyi (STL/OBJ)")
        elif final_model is not None and len(final_model.points) > 0:
            viewer.show_point_cloud(final_model, point_size=3.0)

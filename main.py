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
        num_slices: int = 40,
        show_2d: bool = False,
        save_output: str = "output/simulation_scan.ply"
    ):
        """
        Fiziksel donanim veya video olmadiginda tum pipeline'i dogrulamak icin
        yapay dis arkini tarayan sentetik bir simülasyon akisi calistirir.
        """
        logger.info(f"Sentetik 3D Dental Tarama Simulasyonu baslatiliyor ({num_slices} dilim)...")
        h, w = 400, 600

        for slice_idx in range(num_slices):
            synthetic_frame = np.zeros((h, w, 3), dtype=np.uint8)
            u_start = 80 + slice_idx * 5
            u_end = min(w - 50, u_start + 300)

            phase = slice_idx / 8.0
            for u in range(u_start, u_end):
                v_center = int(200 + 35 * np.sin((u + slice_idx * 10) / 40.0 + phase))
                if 5 <= v_center < h - 5:
                    for dv in range(-3, 4):
                        intensity = int(255 * np.exp(-(dv**2) / 3.0))
                        synthetic_frame[v_center + dv, u] = (0, 0, intensity)

            result = self.process_frame(synthetic_frame)

            if show_2d:
                vis_frame = self.detector.draw_peaks(synthetic_frame, result.get("peaks", []))
                cv2.putText(vis_frame, f"Simulasyon Dilimi: {slice_idx+1}/{num_slices}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Dental Scanner 2D Monitor (Simulasyon)", vis_frame)
                if (cv2.waitKey(20) & 0xFF) == ord('q'):
                    break

        if show_2d:
            cv2.destroyAllWindows()

        logger.info(f"Simulasyon tamamlandi! Global Nokta Sayisi: {len(self.stitcher.global_pcd.points)}")

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
    parser.add_argument("--no-threaded", action="store_true", help="Coklu is parcacikli video okuyucuyu devre disi birak")

    args = parser.parse_args()
    cfg = load_config(args.config)

    pipeline = DentalScannerPipeline.from_config(cfg)

    # Girdi kaynagi secimi
    if args.simulation or (args.input is None and not os.path.exists("data/TestVideo")):
        logger.info("Video kaynagi verilmedi, otomatik Sentetik Simulasyon modu baslatiliyor...")
        final_model = pipeline.run_simulation(
            num_slices=30,
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

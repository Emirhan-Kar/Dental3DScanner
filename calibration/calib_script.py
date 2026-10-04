import os
import sys
import cv2
import yaml
import numpy as np

# Proje kok dizinini sys.path'e ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.logger import get_logger
from utils.config import load_config, get_base_parser

logger = get_logger(__name__)


class CameraCalibrator:
    def __init__(
        self,
        chessboard_size=None,
        square_size=None,
        output_path=None,
        config=None
    ):
        """
        Kamera Kalibrasyon Modulu. Parametreler verilmezse config.yaml'dan yuklenir.

        Args:
            chessboard_size (tuple, optional): Ic kose sayisi (sutun, satir), orn. (9, 6).
            square_size (float, optional): Her bir karenin milimetre cinsinden kenar uzunlugu.
            output_path (str, optional): Katsayilarin kaydedilecegi YAML dosyasinin yolu.
            config (ConfigNode, optional): Merkezi konfigurasyon nesnesi.

        Raises:
            TypeError / ValueError: Parametre tipleri veya degerleri gecersiz ise.
        """
        if config is None and (chessboard_size is None or square_size is None or output_path is None):
            config = load_config()

        calib_cfg = config.calibration if config else None

        # Parametre secimi (Dogrudan verilen > Config > Varsayilan)
        if chessboard_size is None:
            chessboard_size = tuple(calib_cfg.chessboard_size) if calib_cfg else (9, 6)
        if square_size is None:
            square_size = calib_cfg.square_size if calib_cfg else 25.0
        if output_path is None:
            output_path = calib_cfg.output_path if calib_cfg else "calibration/camera_params.yaml"

        if not isinstance(chessboard_size, (tuple, list)) or len(chessboard_size) != 2:
            raise ValueError(f"chessboard_size 2 elemanli (sutun, satir) bir tuple olmalidir, alinan: {chessboard_size}")

        cols, rows = chessboard_size
        if not (isinstance(cols, int) and isinstance(rows, int)) or cols <= 2 or rows <= 2:
            raise ValueError(f"chessboard_size boyutlari en az 3x3 olmalidir, alinan: {chessboard_size}")

        if not isinstance(square_size, (int, float)) or square_size <= 0:
            raise ValueError(f"square_size pozitif bir sayi olmalidir, alinan: {square_size}")

        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError("output_path gecerli bir dosya yolu olmalidir.")

        self.chessboard_size = (cols, rows)
        self.square_size = float(square_size)
        self.output_path = output_path.strip()

        # Gercek dunya 3D koordinat sablonu (Z=0 duzlemi)
        self.objp = np.zeros(
            (chessboard_size[0] * chessboard_size[1], 3), np.float32
        )
        self.objp[:, :2] = (
            np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * self.square_size
        )

        self.obj_points = []  # 3D noktalar
        self.img_points = []  # 2D noktalar

    def process_images(self, image_folder="data/2DTeethData"):
        """
        Klasordeki kalibrasyon fotograflarini tarar ve satranc tahtasi koselerini bulur.

        Args:
            image_folder (str): Kalibrasyon gorsellerinin bulundugu klasor yolu.

        Returns:
            bool: Kalibrasyon basarili ve kaydedildiyse True, aksi halde False.

        Raises:
            FileNotFoundError: image_folder mevcut degilse.
            ValueError: Klasorde uygun uzantida gorsel yoksa.
        """
        if not isinstance(image_folder, str):
            raise TypeError(f"image_folder string olmalidir, alinan: {type(image_folder).__name__}")

        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Klasor bulunamadi: {os.path.abspath(image_folder)}")

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        images = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(valid_extensions)
        ]

        if not images:
            raise ValueError(f"'{image_folder}' klasorunde gecerli resim dosyasi bulunamadi ({valid_extensions}).")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        img_shape = None
        successful_frames = 0

        self.obj_points.clear()
        self.img_points.clear()

        for fname in images:
            try:
                img = cv2.imread(fname)
                if img is None:
                    logger.warning(f"Gorsel acilamadi veya bozuk, atlandi: {fname}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                current_shape = gray.shape[::-1]  # (width, height)

                if img_shape is None:
                    img_shape = current_shape
                elif img_shape != current_shape:
                    logger.warning(f"Gorsel boyutlari farkli ({current_shape} != {img_shape}), atlandi: {fname}")
                    continue

                ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

                if ret and corners is not None:
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    self.obj_points.append(self.objp)
                    self.img_points.append(corners2)
                    successful_frames += 1
                    logger.info(f"Koseler bulundu ve islendi: {os.path.basename(fname)}")
                else:
                    logger.debug(f"Koseler bulunamadi: {os.path.basename(fname)}")

            except Exception as e:
                logger.error(f"Gorsel islenirken beklenmeyen hata ({fname}): {e}")
                continue

        if successful_frames == 0 or len(self.obj_points) == 0:
            logger.error("Hicbir gorselde satranc tahtasi koseleri tespit edilemedi!")
            return False

        if successful_frames < 3:
            logger.warning(f"Yalnizca {successful_frames} gorsel islendi. Guvenilir kalibrasyon icin en az 10-15 farkli aci onerilir.")

        return self.calibrate_and_save(img_shape)

    def calibrate_and_save(self, img_shape):
        """
        Kamera kalibrasyonu yapar ve parametreleri YAML dosyasina kaydeder.

        Args:
            img_shape (tuple): Gorsel genislik ve yukseklik (width, height).

        Returns:
            bool: Basariliysa True.
        """
        if not self.obj_points or not self.img_points:
            raise ValueError("Kalibrasyon icin toplanmis nokta verisi yok.")

        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points, self.img_points, img_shape, None, None
            )
        except cv2.error as e:
            logger.error(f"OpenCV Kamera Kalibrasyonu basarisiz oldu: {e}")
            return False

        data = {
            "camera_matrix": mtx.tolist(),
            "dist_coeff": dist.tolist(),
            "reprojection_error": float(ret),
            "image_width": int(img_shape[0]),
            "image_height": int(img_shape[1])
        }

        try:
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(self.output_path, 'w', encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)

            logger.info(f"Kalibrasyon Basarili! Geri Projeksiyon Hatasi (RMS): {ret:.4f} piksel")
            logger.info(f"Parametreler kaydedildi: {self.output_path}")
            return True

        except Exception as e:
            logger.error(f"Parametreler YAML dosyasina kaydedilemedi: {e}")
            return False


if __name__ == "__main__":
    parser = get_base_parser(description="Optik ve Kamera Kalibrasyon Araci")
    parser.add_argument("--folder", type=str, default=None, help="Kalibrasyon fotograflari klasoru")
    parser.add_argument("--cols", type=int, default=None, help="Satranc tahtasi sutun kose sayisi")
    parser.add_argument("--rows", type=int, default=None, help="Satranc tahtasi satir kose sayisi")
    parser.add_argument("--square-size", type=float, default=None, help="Kare kenar uzunlugu (mm)")
    parser.add_argument("--output", type=str, default=None, help="Kalibrasyon parametreleri cikis dosyasi (.yaml)")

    args = parser.parse_args()
    cfg = load_config(args.config)

    board_size = None
    if args.cols is not None and args.rows is not None:
        board_size = (args.cols, args.rows)

    calibrator = CameraCalibrator(
        chessboard_size=board_size,
        square_size=args.square_size,
        output_path=args.output,
        config=cfg
    )

    folder = args.folder if args.folder else cfg.calibration.image_folder
    if os.path.exists(folder):
        calibrator.process_images(folder)
    else:
        logger.warning(f"Kalibrasyon klasoru bulunamadi: {folder}")
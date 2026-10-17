import os
import sys
import cv2
import yaml
import numpy as np

# Proje kok dizinini sys.path'e ekle
KOK_DIZIN = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK_DIZIN not in sys.path:
    sys.path.insert(0, KOK_DIZIN)

from utils.logger import get_logger
from utils.config import load_config, get_base_parser

kayitci = get_logger(__name__)


class CameraCalibrator:
    def __init__(
        self,
        satranc_boyutu=None,
        kare_boyutu=None,
        cikis_yolu=None,
        konfig=None,
        **ek_parametreler
    ):
        """
        Kamera Kalibrasyon Modulu. Parametreler verilmezse config.yaml'dan yuklenir.

        Args:
            satranc_boyutu (tuple, optional): Ic kose sayisi (sutun, satir), orn. (9, 6).
            kare_boyutu (float, optional): Her bir karenin milimetre cinsinden kenar uzunlugu.
            cikis_yolu (str, optional): Katsayilarin kaydedilecegi YAML dosyasinin yolu.
            konfig (ConfigNode, optional): Merkezi konfigurasyon nesnesi.

        Raises:
            TypeError / ValueError: Parametre tipleri veya degerleri gecersiz ise.
        """
        satranc_boyutu = ek_parametreler.get("chessboard_size", satranc_boyutu)
        kare_boyutu = ek_parametreler.get("square_size", kare_boyutu)
        cikis_yolu = ek_parametreler.get("output_path", cikis_yolu)
        konfig = ek_parametreler.get("config", konfig)

        if konfig is None and (satranc_boyutu is None or kare_boyutu is None or cikis_yolu is None):
            konfig = load_config()

        kalibrasyon_konfig = konfig.calibration if konfig else None

        # Parametre secimi (Dogrudan verilen > Config > Varsayilan)
        if satranc_boyutu is None:
            satranc_boyutu = tuple(kalibrasyon_konfig.chessboard_size) if kalibrasyon_konfig else (9, 6)
        if kare_boyutu is None:
            kare_boyutu = kalibrasyon_konfig.square_size if kalibrasyon_konfig else 25.0
        if cikis_yolu is None:
            cikis_yolu = kalibrasyon_konfig.output_path if kalibrasyon_konfig else "calibration/camera_params.yaml"

        if not isinstance(satranc_boyutu, (tuple, list)) or len(satranc_boyutu) != 2:
            raise ValueError(f"satranc_boyutu 2 elemanli (sutun, satir) bir tuple olmalidir, alinan: {satranc_boyutu}")

        sutun, satir = satranc_boyutu
        if not (isinstance(sutun, int) and isinstance(satir, int)) or sutun <= 2 or satir <= 2:
            raise ValueError(f"satranc_boyutu boyutlari en az 3x3 olmalidir, alinan: {satranc_boyutu}")

        if not isinstance(kare_boyutu, (int, float)) or kare_boyutu <= 0:
            raise ValueError(f"kare_boyutu pozitif bir sayi olmalidir, alinan: {kare_boyutu}")

        if not isinstance(cikis_yolu, str) or not cikis_yolu.strip():
            raise ValueError("cikis_yolu gecerli bir dosya yolu olmalidir.")

        self.satranc_boyutu = (sutun, satir)
        self.kare_boyutu = float(kare_boyutu)
        self.cikis_yolu = cikis_yolu.strip()

        # Geriye donuk uyumluluk ozellikleri
        self.chessboard_size = self.satranc_boyutu
        self.square_size = self.kare_boyutu
        self.output_path = self.cikis_yolu

        # Gercek dunya 3D koordinat sablonu (Z=0 duzlemi)
        self.nesne_sablonu = np.zeros((sutun * satir, 3), np.float32)
        self.nesne_sablonu[:, :2] = (
            np.mgrid[0:sutun, 0:satir].T.reshape(-1, 2) * self.kare_boyutu
        )

        self.nesne_noktalari = []   # 3D noktalar
        self.goruntu_noktalari = []  # 2D noktalar

    def process_images(self, goruntu_klasoru="data/2DTeethData"):
        """
        Klasordeki kalibrasyon fotograflarini tarar ve satranc tahtasi koselerini bulur.

        Args:
            goruntu_klasoru (str): Kalibrasyon gorsellerinin bulundugu klasor yolu.

        Returns:
            bool: Kalibrasyon basarili ve kaydedildiyse True, aksi halde False.

        Raises:
            FileNotFoundError: goruntu_klasoru mevcut degilse.
            ValueError: Klasorde uygun uzantida gorsel yoksa.
        """
        if not isinstance(goruntu_klasoru, str):
            raise TypeError(f"goruntu_klasoru string olmalidir, alinan: {type(goruntu_klasoru).__name__}")

        if not os.path.exists(goruntu_klasoru):
            raise FileNotFoundError(f"Klasor bulunamadi: {os.path.abspath(goruntu_klasoru)}")

        gecerli_uzantilar = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        gorseller = [
            os.path.join(goruntu_klasoru, dosya_adi)
            for dosya_adi in os.listdir(goruntu_klasoru)
            if dosya_adi.lower().endswith(gecerli_uzantilar)
        ]

        if not gorseller:
            raise ValueError(f"'{goruntu_klasoru}' klasorunde gecerli resim dosyasi bulunamadi ({gecerli_uzantilar}).")

        durdurma_kriteri = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        goruntu_boyutu = None
        basarili_kare_sayisi = 0

        self.nesne_noktalari.clear()
        self.goruntu_noktalari.clear()

        for dosya_yolu in gorseller:
            try:
                goruntu = cv2.imread(dosya_yolu)
                if goruntu is None:
                    kayitci.warning(f"Gorsel acilamadi veya bozuk, atlandi: {dosya_yolu}")
                    continue

                gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
                guncel_boyut = gri.shape[::-1]  # (genislik, yukseklik)

                if goruntu_boyutu is None:
                    goruntu_boyutu = guncel_boyut
                elif goruntu_boyutu != guncel_boyut:
                    kayitci.warning(f"Gorsel boyutlari farkli ({guncel_boyut} != {goruntu_boyutu}), atlandi: {dosya_yolu}")
                    continue

                bulundu, koseler = cv2.findChessboardCorners(gri, self.satranc_boyutu, None)

                if bulundu and koseler is not None:
                    hassas_koseler = cv2.cornerSubPix(
                        gri, koseler, (11, 11), (-1, -1), durdurma_kriteri
                    )
                    self.nesne_noktalari.append(self.nesne_sablonu)
                    self.goruntu_noktalari.append(hassas_koseler)
                    basarili_kare_sayisi += 1
                    kayitci.info(f"Koseler bulundu ve islendi: {os.path.basename(dosya_yolu)}")
                else:
                    kayitci.debug(f"Koseler bulunamadi: {os.path.basename(dosya_yolu)}")

            except Exception as hata:
                kayitci.error(f"Gorsel islenirken beklenmeyen hata ({dosya_yolu}): {hata}")
                continue

        if basarili_kare_sayisi == 0 or len(self.nesne_noktalari) == 0:
            kayitci.error("Hicbir gorselde satranc tahtasi koseleri tespit edilemedi!")
            return False

        if basarili_kare_sayisi < 3:
            kayitci.warning(
                f"Yalnizca {basarili_kare_sayisi} gorsel islendi. "
                f"Guvenilir kalibrasyon icin en az 10-15 farkli aci onerilir."
            )

        return self.calibrate_and_save(goruntu_boyutu)

    def calibrate_and_save(self, goruntu_boyutu):
        """
        Kamera kalibrasyonu yapar ve parametreleri YAML dosyasina kaydeder.

        Args:
            goruntu_boyutu (tuple): Gorsel boyutu (genislik, yukseklik).

        Returns:
            bool: Basariliysa True.
        """
        if not self.nesne_noktalari or not self.goruntu_noktalari:
            raise ValueError("Kalibrasyon icin toplanmis nokta verisi yok.")

        try:
            rms_hata, kamera_matrisi, bozulma_katsayilari, donme_vektorleri, oteleme_vektorleri = cv2.calibrateCamera(
                self.nesne_noktalari, self.goruntu_noktalari, goruntu_boyutu, None, None
            )
        except cv2.error as hata:
            kayitci.error(f"OpenCV Kamera Kalibrasyonu basarisiz oldu: {hata}")
            return False

        # NOT: YAML anahtarlari diger moduller tarafindan okundugu icin Ingilizce birakildi.
        cikti_verisi = {
            "camera_matrix": kamera_matrisi.tolist(),
            "dist_coeff": bozulma_katsayilari.tolist(),
            "reprojection_error": float(rms_hata),
            "image_width": int(goruntu_boyutu[0]),
            "image_height": int(goruntu_boyutu[1])
        }

        try:
            cikis_dizini = os.path.dirname(self.cikis_yolu)
            if cikis_dizini:
                os.makedirs(cikis_dizini, exist_ok=True)

            with open(self.cikis_yolu, 'w', encoding="utf-8") as dosya:
                yaml.dump(cikti_verisi, dosya, default_flow_style=False)

            kayitci.info(f"Kalibrasyon Basarili! Geri Projeksiyon Hatasi (RMS): {rms_hata:.4f} piksel")
            kayitci.info(f"Parametreler kaydedildi: {self.cikis_yolu}")
            return True

        except Exception as hata:
            kayitci.error(f"Parametreler YAML dosyasina kaydedilemedi: {hata}")
            return False


if __name__ == "__main__":
    ayristirici = get_base_parser(description="Optik ve Kamera Kalibrasyon Araci")
    ayristirici.add_argument("--folder", type=str, default=None, help="Kalibrasyon fotograflari klasoru")
    ayristirici.add_argument("--cols", type=int, default=None, help="Satranc tahtasi sutun kose sayisi")
    ayristirici.add_argument("--rows", type=int, default=None, help="Satranc tahtasi satir kose sayisi")
    ayristirici.add_argument("--square-size", type=float, default=None, help="Kare kenar uzunlugu (mm)")
    ayristirici.add_argument("--output", type=str, default=None, help="Kalibrasyon parametreleri cikis dosyasi (.yaml)")

    argumanlar = ayristirici.parse_args()
    yuklu_konfig = load_config(argumanlar.config)

    tahta_boyutu = None
    if argumanlar.cols is not None and argumanlar.rows is not None:
        tahta_boyutu = (argumanlar.cols, argumanlar.rows)

    kalibrator = CameraCalibrator(
        satranc_boyutu=tahta_boyutu,
        kare_boyutu=argumanlar.square_size,
        cikis_yolu=argumanlar.output,
        konfig=yuklu_konfig
    )

    klasor = argumanlar.folder if argumanlar.folder else yuklu_konfig.calibration.image_folder
    if os.path.exists(klasor):
        kalibrator.process_images(klasor)
    else:
        kayitci.warning(f"Kalibrasyon klasoru bulunamadi: {klasor}")
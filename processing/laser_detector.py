import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class LaserDetector:

  def __init__(
      self,
      lower_hsv=(0, 100, 100),
      upper_hsv=(10, 255, 255),
      intensity_threshold=50,
  ):
    """Lazer tespit modülü.

    lower_hsv: Lazer rengi alt eşiği (varsayılan kırmızı)
    upper_hsv: Lazer rengi üst eşiği
    intensity_threshold: Alt-piksel hesaplamasına dahil edilecek minimum parlaklık

    Raises:
        ValueError: HSV değerleri geçersiz aralıkta ise
        TypeError: Parametreler beklenen türde değilse
    """
    # Parametre doğrulama
    if not isinstance(intensity_threshold, (int, float)):
      raise TypeError(f"intensity_threshold sayısal olmalıdır, alınan: {type(intensity_threshold).__name__}")
    if intensity_threshold < 0 or intensity_threshold > 255:
      raise ValueError(f"intensity_threshold 0-255 arasında olmalıdır, alınan: {intensity_threshold}")

    try:
      self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
      self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
    except (ValueError, TypeError) as e:
      raise ValueError(f"HSV değerleri geçerli bir dizi olmalıdır: {e}")

    if self.lower_hsv.shape != (3,) or self.upper_hsv.shape != (3,):
      raise ValueError(f"HSV dizileri 3 elemanlı olmalıdır. lower: {self.lower_hsv.shape}, upper: {self.upper_hsv.shape}")

    self.intensity_threshold = intensity_threshold

  @classmethod
  def from_config(cls, config=None):
    """Konfigurasyon nesnesinden veya config.yaml dosyasindan LaserDetector olusturur."""
    if config is None:
      from utils.config import load_config
      config = load_config()
    ld_cfg = config.laser_detector
    return cls(
        lower_hsv=ld_cfg.lower_hsv,
        upper_hsv=ld_cfg.upper_hsv,
        intensity_threshold=ld_cfg.intensity_threshold
    )

  def _validate_frame(self, frame_bgr):
    """Giriş karesinin geçerliliğini doğrular.

    Raises:
        TypeError: frame None veya numpy dizisi değilse
        ValueError: frame boyutları geçersizse (3 kanallı BGR bekleniyour)
    """
    if frame_bgr is None:
      raise TypeError("Giriş karesi None olamaz.")
    if not isinstance(frame_bgr, np.ndarray):
      raise TypeError(f"Giriş karesi numpy.ndarray olmalıdır, alınan: {type(frame_bgr).__name__}")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
      raise ValueError(
        f"Giriş karesi 3 kanallı (BGR) olmalıdır. "
        f"Alınan boyut: {frame_bgr.shape}"
      )
    if frame_bgr.size == 0:
      raise ValueError("Giriş karesi boş (0 piksel).")

  def extract_mask(self, frame_bgr):
    """Kareden lazer rengini HSV uzayında izole eder.

    Args:
        frame_bgr: BGR formatında numpy dizisi (H, W, 3)

    Returns:
        mask: İkili maske (H, W) — lazer bölgeleri beyaz

    Raises:
        TypeError: frame geçersiz türde ise
        ValueError: frame boyutları hatalı ise
        RuntimeError: OpenCV renk dönüşümü başarısız olursa
    """
    self._validate_frame(frame_bgr)

    try:
      hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
      raise RuntimeError(f"BGR→HSV renk dönüşümü başarısız: {e}")

    # Kırmızı renk HSV uzayında 0 ve 180 çevresinde iki bölgeye ayrılır
    if self.lower_hsv[0] <= 10:
      mask1 = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
      lower_red2 = np.array([170, self.lower_hsv[1], self.lower_hsv[2]])
      upper_red2 = np.array([180, self.upper_hsv[1], self.upper_hsv[2]])
      mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
      mask = cv2.bitwise_or(mask1, mask2)
    else:
      mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

    return mask

  def detect_subpixel_peaks(self, frame_bgr):
    """Her sütun (u) için lazer çizgisinin alt-piksel dikey konumunu (v) hesaplar.

    Args:
        frame_bgr: BGR formatında numpy dizisi (H, W, 3)

    Returns:
        peaks: Alt-piksel tepe noktaları listesi [(u_0, v_0), (u_1, v_1), ...]
        mask: Lazer maskeleme sonucu (H, W)

    Raises:
        TypeError: frame geçersiz türde ise
        ValueError: frame boyutları hatalı ise
        RuntimeError: Görüntü işleme sırasında beklenmeyen hata
    """
    # _validate_frame extract_mask içinde çağrılıyor
    try:
      mask = self.extract_mask(frame_bgr)
    except (TypeError, ValueError):
      raise  # Doğrulama hatalarını olduğu gibi ilet
    except Exception as e:
      raise RuntimeError(f"Maske çıkarımı sırasında hata: {e}")

    try:
      gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
      raise RuntimeError(f"BGR→GRAY dönüşümü başarısız: {e}")

    filtered = cv2.bitwise_and(gray, gray, mask=mask)

    height, width = filtered.shape
    peaks = []

    # Her sütunu dikeyde tara
    for u in range(width):
      column = filtered[:, u].astype(np.float32)

      # Eşik değerin üzerindeki pikselleri filtrele
      valid_indices = np.where(column >= self.intensity_threshold)[0]

      if len(valid_indices) > 0:
        intensities = column[valid_indices]
        total_intensity = np.sum(intensities)

        if total_intensity > 0:
          # Ağırlık Merkezi (Center of Mass) alt-piksel formülü
          v_subpixel = np.sum(valid_indices * intensities) / total_intensity
          peaks.append((float(u), float(v_subpixel)))

    logger.debug(f"Tespit edilen alt-piksel tepe noktasi sayisi: {len(peaks)}")
    return peaks, mask

  def draw_peaks(self, frame_bgr, peaks):
    """Tespit edilen alt-piksel tepe noktalarını görsel üzerine yeşil noktalarla çizer.

    Args:
        frame_bgr: BGR formatında numpy dizisi (H, W, 3)
        peaks: Tepe noktaları listesi [(u, v), ...]

    Returns:
        vis_frame: Üzerine noktalar çizilmiş kare kopyası

    Raises:
        TypeError: frame veya peaks geçersiz türde ise
    """
    self._validate_frame(frame_bgr)

    if peaks is None:
      peaks = []
    if not isinstance(peaks, (list, tuple)):
      raise TypeError(f"peaks bir liste veya tuple olmalıdır, alınan: {type(peaks).__name__}")

    vis_frame = frame_bgr.copy()
    height, width = vis_frame.shape[:2]

    for peak in peaks:
      try:
        u, v = peak
        # Çizim koordinatlarının kare sınırları içinde olduğunu doğrula
        u_int = int(round(u))
        v_int = int(round(v))
        if 0 <= u_int < width and 0 <= v_int < height:
          cv2.circle(vis_frame, (u_int, v_int), 1, (0, 255, 0), -1)
      except (ValueError, TypeError):
        # Geçersiz tepe noktasını atla, diğerlerine devam et
        continue

    return vis_frame
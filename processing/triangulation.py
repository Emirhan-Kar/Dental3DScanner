import os
import cv2
import numpy as np
import yaml
from utils.logger import get_logger

logger = get_logger(__name__)


class Triangulator:

  def __init__(
      self,
      camera_matrix=None,
      dist_coeffs=None,
      laser_plane=np.array([0.0, -1.0, 0.5, -150.0], dtype=np.float32),
  ):
    """3D Üçgenleme (Triangulation) Modülü.

    camera_matrix: 3x3 Intrinsic kamera matrisi (fx, fy, cx, cy)
    dist_coeffs: Lens distorsiyon katsayıları
    laser_plane: Lazer düzlem katsayıları [A, B, C, D] (Ax + By + Cz + D = 0)

    Raises:
        ValueError: Kamera matrisi veya lazer düzlem katsayıları geçersiz ise
    """
    if camera_matrix is None:
      # Varsayılan 720p pinhole kamera matrisi (simülasyon için)
      self.K = np.array(
          [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
          dtype=np.float32,
      )
    else:
      try:
        self.K = np.array(camera_matrix, dtype=np.float32)
      except (ValueError, TypeError) as e:
        raise ValueError(f"camera_matrix geçerli bir sayısal diziye dönüştürülemedi: {e}")

      if self.K.shape != (3, 3):
        raise ValueError(f"camera_matrix 3x3 boyutunda olmalıdır, alınan: {self.K.shape}")

      # Odak uzaklıklarının pozitif olduğunu kontrol et
      if self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
        raise ValueError(
          f"Odak uzaklıkları (fx, fy) pozitif olmalıdır. "
          f"fx={self.K[0, 0]}, fy={self.K[1, 1]}"
        )

    self.dist_coeffs = (
        np.zeros(5, dtype=np.float32) if dist_coeffs is None else np.array(dist_coeffs, dtype=np.float32)
    )

    # Lazer düzlem katsayılarını doğrula
    try:
      self.laser_plane = np.array(laser_plane, dtype=np.float32)
    except (ValueError, TypeError) as e:
      raise ValueError(f"laser_plane geçerli bir sayısal diziye dönüştürülemedi: {e}")

    if self.laser_plane.shape != (4,):
      raise ValueError(f"laser_plane 4 elemanlı olmalıdır [A, B, C, D], alınan boyut: {self.laser_plane.shape}")

    # Normal vektörünün sıfır olup olmadığını kontrol et (A, B, C hepsi 0 olamaz)
    normal_magnitude = np.linalg.norm(self.laser_plane[:3])
    if normal_magnitude < 1e-10:
      raise ValueError("Lazer düzlem normal vektörü [A, B, C] sıfır olamaz.")

    self.fx = self.K[0, 0]
    self.fy = self.K[1, 1]
    self.cx = self.K[0, 2]
    self.cy = self.K[1, 2]

  @classmethod
  def from_yaml(cls, yaml_path, laser_plane=None):
    """YAML dosyasından kamera parametrelerini okuyarak nesne oluşturur.

    Args:
        yaml_path: Kamera parametreleri YAML dosyasının yolu
        laser_plane: Opsiyonel lazer düzlem katsayıları [A, B, C, D]

    Returns:
        Triangulator nesnesi

    Raises:
        FileNotFoundError: YAML dosyası bulunamazsa (uyarı verir, varsayılan döndürür)
        ValueError: YAML dosyası geçersiz veya eksik alan içeriyorsa
    """
    if not isinstance(yaml_path, str):
      raise TypeError(f"yaml_path bir string olmalıdır, alınan: {type(yaml_path).__name__}")

    if not os.path.exists(yaml_path):
      logger.warning(f"Kamera parametre dosyasi bulunamadi ({yaml_path}), varsayilan matris kullaniliyor.")
      return cls()

    try:
      with open(yaml_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    except yaml.YAMLError as e:
      raise ValueError(f"YAML dosyası ayrıştırılamadı ({yaml_path}): {e}")
    except PermissionError:
      raise PermissionError(f"YAML dosyasına erişim izni yok: {yaml_path}")
    except OSError as e:
      raise OSError(f"YAML dosyası okunamadı ({yaml_path}): {e}")

    if params is None:
      raise ValueError(f"YAML dosyası boş: {yaml_path}")

    # Gerekli alanları kontrol et
    required_keys = ["camera_matrix", "dist_coeff"]
    missing_keys = [k for k in required_keys if k not in params]
    if missing_keys:
      raise ValueError(
        f"YAML dosyasında gerekli alanlar eksik: {missing_keys}. "
        f"Mevcut anahtarlar: {list(params.keys())}"
      )

    try:
      k_mat = np.array(params["camera_matrix"], dtype=np.float32)
      dist = np.array(params["dist_coeff"], dtype=np.float32)
    except (ValueError, TypeError) as e:
      raise ValueError(f"YAML'daki kamera parametreleri sayısal diziye dönüştürülemedi: {e}")

    if laser_plane is None:
      laser_plane = np.array([0.0, -1.0, 0.5, -150.0], dtype=np.float32)

    return cls(camera_matrix=k_mat, dist_coeffs=dist, laser_plane=laser_plane)

  @classmethod
  def from_config(cls, config=None):
    """Konfigurasyon nesnesinden veya config.yaml dosyasindan Triangulator olusturur."""
    if config is None:
      from utils.config import load_config
      config = load_config()

    cam_cfg = config.camera
    laser_plane = getattr(cam_cfg, "laser_plane", [0.0, -1.0, 0.5, -150.0])
    yaml_path = getattr(cam_cfg, "params_yaml", "calibration/camera_params.yaml")

    if yaml_path and os.path.exists(yaml_path) and os.path.getsize(yaml_path) > 0:
      try:
        return cls.from_yaml(yaml_path, laser_plane=laser_plane)
      except Exception as e:
        logger.warning(f"Kamera parametre dosyasi okunamadi ({e}), fallback matris kullaniliyor.")

    return cls(
        camera_matrix=cam_cfg.fallback_matrix,
        dist_coeffs=cam_cfg.dist_coeffs,
        laser_plane=laser_plane
    )

  def pixels_to_3d(self, peaks):
    """Tespit edilen 2D piksel listesini [(u, v), ...] 3D nokta bulutuna (Nx3) dönüştürür.

    Args:
        peaks: 2D piksel koordinatları listesi [(u0, v0), (u1, v1), ...]

    Returns:
        points_3d: 3D nokta bulutu (Nx3) numpy dizisi

    Raises:
        TypeError: peaks geçersiz türde ise
        ValueError: peaks formatı hatalı ise
    """
    if peaks is None:
      raise TypeError("peaks None olamaz. Boş liste [] gönderin.")

    if not isinstance(peaks, (list, tuple, np.ndarray)):
      raise TypeError(f"peaks bir liste, tuple veya numpy dizisi olmalıdır, alınan: {type(peaks).__name__}")

    if len(peaks) == 0:
      return np.empty((0, 3), dtype=np.float32)

    try:
      pts_2d = np.array(peaks, dtype=np.float32)
    except (ValueError, TypeError) as e:
      raise ValueError(f"peaks sayısal diziye dönüştürülemedi: {e}")

    if pts_2d.ndim != 2 or pts_2d.shape[1] != 2:
      raise ValueError(
        f"peaks (N, 2) boyutunda olmalıdır, alınan: {pts_2d.shape}. "
        f"Her eleman (u, v) çifti olmalıdır."
      )

    # NaN veya Inf değerleri kontrol et
    if np.any(~np.isfinite(pts_2d)):
      nan_count = np.sum(~np.isfinite(pts_2d)) // 2
      logger.warning(f"peaks icinde {nan_count} adet NaN/Inf deger tespit edildi, bunlar filtreleniyor.")
      valid_rows = np.all(np.isfinite(pts_2d), axis=1)
      pts_2d = pts_2d[valid_rows]
      if len(pts_2d) == 0:
        return np.empty((0, 3), dtype=np.float32)

    u = pts_2d[:, 0]
    v = pts_2d[:, 1]

    # Normalize kamera koordinatları (Ray Direction)
    x_n = (u - self.cx) / self.fx
    y_n = (v - self.cy) / self.fy

    # Lazer düzlemi: A*X + B*Y + C*Z + D = 0 -> X = Z*x_n, Y = Z*y_n
    # Z*(A*x_n + B*y_n + C) + D = 0  =>  Z = -D / (A*x_n + B*y_n + C)
    A, B, C, D = self.laser_plane
    denominator = A * x_n + B * y_n + C

    # Sıfıra bölme (ışının düzleme paralel olması) hatalarını engelle
    valid_mask = np.abs(denominator) > 1e-6
    denominator = np.where(valid_mask, denominator, np.nan)

    z = -D / denominator
    x = z * x_n
    y = z * y_n

    # Sadece kameranın önündeki (Z > 0) geçerli noktaları filtrele
    valid_points_mask = (z > 0) & (~np.isnan(z)) & valid_mask
    points_3d = np.stack(
        [x[valid_points_mask], y[valid_points_mask], z[valid_points_mask]],
        axis=-1,
    )

    filtered_count = len(peaks) - points_3d.shape[0]
    if filtered_count > 0 and len(peaks) > 0:
      filter_ratio = filtered_count / len(peaks) * 100
      if filter_ratio > 50:
        logger.warning(
          f"Giris noktalarinin %{filter_ratio:.1f}'i filtrelendi "
          f"({filtered_count}/{len(peaks)}). Lazer duzlem parametrelerini kontrol edin."
        )

    logger.debug(f"Uretilen 3D nokta sayisi: {points_3d.shape[0]}")
    return points_3d.astype(np.float32)
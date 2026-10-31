import pytest
import numpy as np
from processing.laser_detector import LaserDetector


class TestLaserDetector:
    def test_init_defaults(self):
        """Varsayilan parametrelerle baslatma."""
        detector = LaserDetector()
        assert detector.intensity_threshold == 50
        assert detector.lower_hsv.shape == (3,)
        assert detector.upper_hsv.shape == (3,)

    def test_init_invalid_threshold(self):
        """Gecersiz esik degeri ile baslatma hata firlatmali."""
        with pytest.raises(ValueError):
            LaserDetector(intensity_threshold=-10)
        with pytest.raises(ValueError):
            LaserDetector(intensity_threshold=300)
        with pytest.raises(TypeError):
            LaserDetector(intensity_threshold="elli")

    def test_init_invalid_hsv(self):
        """Gecersiz HSV boyutlari hata firlatmali."""
        with pytest.raises(ValueError):
            LaserDetector(lower_hsv=(0, 100))

    def test_extract_mask_success(self, synthetic_laser_frame):
        """Sentetik lazer karesinden maske cikarimi."""
        detector = LaserDetector()
        mask = detector.extract_mask(synthetic_laser_frame)
        assert mask is not None
        assert mask.shape == synthetic_laser_frame.shape[:2]
        assert np.count_nonzero(mask) > 0

    def test_extract_mask_invalid_inputs(self):
        """Gecersiz kare tipleri TypeError veya ValueError firlatmali."""
        detector = LaserDetector()
        with pytest.raises(TypeError):
            detector.extract_mask(None)
        with pytest.raises(ValueError):
            detector.extract_mask(np.zeros((100, 100), dtype=np.uint8))  # 2D BGR degil
        with pytest.raises(ValueError):
            detector.extract_mask(np.empty((0, 0, 3), dtype=np.uint8))  # Bos

    def test_detect_subpixel_peaks(self, synthetic_laser_frame):
        """Alt-piksel tepe noktalarinin tespiti ve deger dogrulamasi."""
        detector = LaserDetector()
        peaks, mask = detector.detect_subpixel_peaks(synthetic_laser_frame)

        assert isinstance(peaks, list)
        assert len(peaks) > 50  # Sentetik karede yaklasik 500 tepe var
        for u, v in peaks:
            assert isinstance(u, float)
            assert isinstance(v, float)
            assert 0 <= u < synthetic_laser_frame.shape[1]
            assert 0 <= v < synthetic_laser_frame.shape[0]

    def test_draw_peaks(self, synthetic_laser_frame):
        """Tespit edilen noktalarin gorsel uzerine cizimi."""
        detector = LaserDetector()
        peaks = [(100.2, 200.5), (105.1, 202.3)]
        vis = detector.draw_peaks(synthetic_laser_frame, peaks)
        assert vis.shape == synthetic_laser_frame.shape
        assert vis is not synthetic_laser_frame  # Kopyalanmis olmali

    def test_draw_peaks_invalid_inputs(self, synthetic_laser_frame):
        """Gecersiz tepe noktasi listesi hatasi."""
        detector = LaserDetector()
        with pytest.raises(TypeError):
            detector.draw_peaks(synthetic_laser_frame, "gecersiz_peaks")

    def test_from_config(self, default_config):
        """Konfigurasyondan LaserDetector olusturma."""
        detector = LaserDetector.from_config(default_config)
        assert detector.intensity_threshold == default_config.laser_detector.intensity_threshold


def test_iki_parcali_sutunda_sahte_nokta_yok():
    """Ayni sutunda iki ayri cizgi parcasi varsa, tepe noktasi ikisinin arasina dusmemeli."""
    import numpy as np
    from processing.laser_detector import LaserDetector
    img = np.zeros((200, 50, 3), dtype=np.uint8)
    img[49:52, :, 2] = 255      # guclu parca (v~50)
    img[149:151, :, 2] = 230    # daha zayif parca (v~150)
    tepe, _ = LaserDetector().detect_subpixel_peaks(img)
    v = np.array([p[1] for p in tepe])
    assert np.all(np.abs(v - 50) < 1.0)
    eski, _ = LaserDetector(single_peak=False).detect_subpixel_peaks(img)
    assert np.all(np.abs(np.array([p[1] for p in eski]) - 50) > 10)   # eski davranis: araya duser

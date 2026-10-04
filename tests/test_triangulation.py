import pytest
import numpy as np
from processing.triangulation import Triangulator


class TestTriangulator:
    def test_init_defaults(self):
        """Varsayilan pinhole matrisi ile baslatma."""
        tri = Triangulator()
        assert tri.K.shape == (3, 3)
        assert tri.fx > 0 and tri.fy > 0
        assert tri.laser_plane.shape == (4,)

    def test_init_invalid_matrix(self):
        """Gecersiz kamera matrisi kontrolleri."""
        with pytest.raises(ValueError):
            Triangulator(camera_matrix=[[1, 0], [0, 1]])  # 2x2 matris

        with pytest.raises(ValueError):
            Triangulator(camera_matrix=[[-800.0, 0, 0], [0, 800.0, 0], [0, 0, 1]])  # Negatif fx

    def test_init_invalid_laser_plane(self):
        """Gecersiz lazer duzlemi katsayilari."""
        with pytest.raises(ValueError):
            Triangulator(laser_plane=[0.0, 0.0, 0.0, 10.0])  # Normal sifir olamaz

        with pytest.raises(ValueError):
            Triangulator(laser_plane=[1.0, 2.0])  # 4 elemanli olmali

    def test_pixels_to_3d_success(self, sample_peaks):
        """2D piksellerden 3D nokta bulutu uretimi."""
        tri = Triangulator()
        cloud = tri.pixels_to_3d(sample_peaks)

        assert isinstance(cloud, np.ndarray)
        assert cloud.ndim == 2
        assert cloud.shape[1] == 3
        assert cloud.dtype == np.float32
        # Uretilen tum noktalar kameranin onunde (Z > 0) olmali
        if cloud.shape[0] > 0:
            assert np.all(cloud[:, 2] > 0)

    def test_pixels_to_3d_empty_and_none(self):
        """Bos ve None girdiler icin guvenli davranis."""
        tri = Triangulator()
        empty_cloud = tri.pixels_to_3d([])
        assert empty_cloud.shape == (0, 3)

        with pytest.raises(TypeError):
            tri.pixels_to_3d(None)

    def test_pixels_to_3d_invalid_shape(self):
        """(N, 2) disindaki boyutlar ValueError firlatmali."""
        tri = Triangulator()
        with pytest.raises(ValueError):
            tri.pixels_to_3d(np.ones((10, 3)))  # 3D nokta verilmis

    def test_from_config(self, default_config):
        """Konfigurasyondan Triangulator olusturma."""
        tri = Triangulator.from_config(default_config)
        assert tri.fx > 0
        assert tri.fy > 0

import os
import sys
import pytest
import numpy as np
import open3d as o3d

# Testlerin her zaman proje kok dizinini gormesini sagla
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.config import load_config, ConfigNode


@pytest.fixture
def synthetic_laser_frame():
    """Lazer cizgisi tespiti icin sentetik karanlik cerceve ve kirmizi lazer profili."""
    h, w = 400, 600
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for u in range(50, 550):
        v_center = int(200 + 40 * np.sin(u / 45.0))
        for dv in range(-3, 4):
            intensity = int(255 * np.exp(-(dv**2) / 3.0))
            frame[v_center + dv, u] = (0, 0, intensity)
    return frame


@pytest.fixture
def sample_peaks():
    """2D alt-piksel lazer tepe noktasi listesi [(u, v), ...]."""
    peaks = []
    for u in range(100, 500, 10):
        v = 240.0 + 20.0 * np.sin(u / 30.0)
        peaks.append((float(u), float(v)))
    return peaks


@pytest.fixture
def sample_point_cloud():
    """Open3D ile olusturulmus sentetik nokta bulutu."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=10.0, resolution=20)
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    return pcd


@pytest.fixture
def default_config():
    """configs/config.yaml veya varsayilan ConfigNode nesnesi."""
    return load_config()

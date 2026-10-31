import os
import sys
import numpy as np
import pytest
import open3d as o3d

from processing.icp_stitcher import PointCloudStitcher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import birlestirme_deneyi as bd  # noqa: E402


def _engebeli_yuzey(n=6000, tohum=0):
    """Simetrisi olmayan, dis yuzeyine benzer engebeli bir yuzey (mm)."""
    rng = np.random.RandomState(tohum)
    xy = rng.uniform(-10, 10, size=(n, 2))
    z = 2.0 * np.sin(xy[:, 0] / 2.5) * np.cos(xy[:, 1] / 3.0) + 0.05 * xy[:, 0] ** 2 / 10 + 0.3 * xy[:, 1]
    return np.column_stack([xy, z])


def _pcd(P):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(P)
    return p


class TestRegister:
    def test_gecersiz_yontem(self):
        with pytest.raises(ValueError):
            PointCloudStitcher(icp_method="nokta")

    @pytest.mark.parametrize("mod", ["local", "global", "hybrid"])
    def test_kucuk_hareketi_geri_bulur(self, mod):
        hedef = _engebeli_yuzey(tohum=0)
        temiz = _engebeli_yuzey(tohum=1)          # ayni yuzey, farkli ornekleme
        T = bd.rastgele_hareket(np.random.RandomState(3), "kucuk", temiz.mean(0))
        st = PointCloudStitcher(voxel_size=0.4, registration_mode=mod, seed=0)
        T_est = st.register(_pcd(bd.uygula(T, temiz)), _pcd(hedef))
        rmse, aci, _ = bd.hata_olc(T_est @ T, temiz)
        assert rmse < 0.1 and aci < 0.5
        assert st.last_result["fitness"] > 0.5
        assert st.last_result["valid"]

    def test_hibrit_buyuk_harekette_ransaca_gecer(self):
        hedef = _engebeli_yuzey(tohum=0)
        temiz = _engebeli_yuzey(tohum=1)
        T = bd.rastgele_hareket(np.random.RandomState(5), "buyuk", temiz.mean(0))
        st = PointCloudStitcher(voxel_size=0.4, registration_mode="hybrid", seed=0)
        T_est = st.register(_pcd(bd.uygula(T, temiz)), _pcd(hedef))
        rmse, _, _ = bd.hata_olc(T_est @ T, temiz)
        assert st.last_result["ransac_used"] and rmse < 0.1

    def test_gecersiz_mod(self):
        with pytest.raises(ValueError):
            PointCloudStitcher(registration_mode="hizli")

    def test_from_config_parametreleri_okur(self, default_config):
        st = PointCloudStitcher.from_config(default_config)
        assert st.icp_max_iterations == default_config.icp_stitcher.icp_max_iterations
        assert st.ransac_distance_multiplier == default_config.icp_stitcher.ransac_distance_multiplier


class TestDeneyYardimcilari:
    def test_ark_sirasi(self):
        alt = sorted([31, 41, 47, 33, 42], key=lambda n: bd.ark_sirasi(n, "lower"))
        assert alt == [47, 42, 41, 31, 33]
        ust = sorted([21, 11, 18, 23], key=lambda n: bd.ark_sirasi(n, "upper"))
        assert ust == [18, 11, 21, 23]

    def test_hata_olc_birim(self):
        P = np.random.rand(100, 3)
        rmse, aci, sapma = bd.hata_olc(np.eye(4), P)
        assert rmse == pytest.approx(0) and aci == pytest.approx(0, abs=1e-6)

    def test_hareket_buyuklugu(self):
        merkez = np.array([5.0, -3.0, -90.0])
        T = bd.rastgele_hareket(np.random.RandomState(0), "buyuk", merkez)
        aci = np.degrees(np.arccos((np.trace(T[:3, :3]) - 1) / 2))
        assert 25 <= aci <= 35
        assert 8 <= np.linalg.norm(bd.uygula(T, merkez[None])[0] - merkez) <= 12

    def test_sabit_hash(self):
        assert bd.sabit_hash("a", 1) == bd.sabit_hash("a", 1)

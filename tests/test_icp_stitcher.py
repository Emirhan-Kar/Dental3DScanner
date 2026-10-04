import copy
import pytest
import numpy as np
import open3d as o3d
from processing.icp_stitcher import PointCloudStitcher


class TestPointCloudStitcher:
    def test_init_valid(self):
        """Gecerli voxel_size ile baslatma."""
        stitcher = PointCloudStitcher(voxel_size=0.5)
        assert stitcher.voxel_size == 0.5
        assert len(stitcher.global_pcd.points) == 0

    def test_init_invalid_voxel_size(self):
        """Gecersiz veya negatif voxel_size degerleri."""
        with pytest.raises(ValueError):
            PointCloudStitcher(voxel_size=-0.2)
        with pytest.raises(ValueError):
            PointCloudStitcher(voxel_size=0.0)
        with pytest.raises(TypeError):
            PointCloudStitcher(voxel_size="yarim_milimetre")

    def test_validate_pcd(self):
        """Nokta bulutu gecersizlik denetimleri."""
        stitcher = PointCloudStitcher()
        with pytest.raises(TypeError):
            stitcher.align_and_merge(None)
        with pytest.raises(TypeError):
            stitcher.align_and_merge(np.zeros((10, 3)))
        with pytest.raises(ValueError):
            stitcher.align_and_merge(o3d.geometry.PointCloud())  # 0 nokta

    def test_preprocess_and_compute_fpfh(self, sample_point_cloud):
        """FPFH oznitelik cikarimi ve downsample testi."""
        stitcher = PointCloudStitcher(voxel_size=1.0)
        pcd_down, fpfh = stitcher.preprocess_and_compute_fpfh(sample_point_cloud)

        assert len(pcd_down.points) > 0
        assert fpfh.num() == len(pcd_down.points)
        assert fpfh.dimension() == 33  # FPFH 33-boyutlu histogramdir

    def test_align_and_merge_single_frame(self, sample_point_cloud):
        """Ilk kare eklendiginde kimlik (identity) matrisi donmeli ve global_pcd set edilmeli."""
        stitcher = PointCloudStitcher(voxel_size=1.0)
        trans = stitcher.align_and_merge(sample_point_cloud)

        np.testing.assert_allclose(trans, np.identity(4), atol=1e-5)
        assert len(stitcher.get_global_model().points) > 0

    def test_align_and_merge_two_frames(self, sample_point_cloud):
        """Iki cercevenin RANSAC ve ICP ile basariyla kilitlenmesi."""
        stitcher = PointCloudStitcher(voxel_size=1.0)

        # Hedef model
        target = copy.deepcopy(sample_point_cloud)

        # Kaynak model: Kucuk donus ve otelenme
        source = copy.deepcopy(sample_point_cloud)
        theta = np.radians(5.0)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0, 1.0],
            [np.sin(theta),  np.cos(theta), 0, 0.5],
            [0,              0,             1, 0.2],
            [0,              0,             0, 1.0]
        ])
        source.transform(R)

        # Hizala ve birlestir
        trans_first = stitcher.align_and_merge(target)
        trans_second = stitcher.align_and_merge(source)

        assert trans_second.shape == (4, 4)
        assert np.all(np.isfinite(trans_second))
        # Global model iki karenin birlesimiyle buyumus olmali
        assert len(stitcher.get_global_model().points) >= len(target.points) * 0.5

    def test_reset(self, sample_point_cloud):
        """Stitcher modelini sifirlama testi."""
        stitcher = PointCloudStitcher(voxel_size=1.0)
        stitcher.align_and_merge(sample_point_cloud)
        assert len(stitcher.get_global_model().points) > 0

        stitcher.reset()
        assert len(stitcher.get_global_model().points) == 0

    def test_from_config(self, default_config):
        """Konfigurasyondan PointCloudStitcher olusturma."""
        stitcher = PointCloudStitcher.from_config(default_config)
        assert stitcher.voxel_size == default_config.icp_stitcher.voxel_size

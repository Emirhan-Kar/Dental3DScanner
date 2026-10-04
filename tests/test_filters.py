import pytest
import numpy as np
import open3d as o3d

from processing.filters import (
    validate_point_cloud_input,
    remove_statistical_outliers,
    remove_radius_outliers,
    crop_bounding_box,
    filter_dental_point_cloud
)


@pytest.fixture
def sample_point_cloud():
    """Yogun bir kure ve 2 adet uzak aykiri nokta iceren test bulutu."""
    # 100 noktalı kure
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
    pcd = sphere_mesh.sample_points_uniformly(number_of_points=120)

    # 2 adet cok uzak aykiri nokta ekle
    outliers = np.array([
        [500.0, 500.0, 500.0],
        [-500.0, -500.0, -500.0]
    ])
    all_points = np.vstack([np.asarray(pcd.points), outliers])
    noisy_pcd = o3d.geometry.PointCloud()
    noisy_pcd.points = o3d.utility.Vector3dVector(all_points)
    return noisy_pcd


class TestPointCloudFilters:
    def test_input_validation(self):
        """Gecersiz girdi tipleri ve bos nokta bulutlari yakalanmalidir."""
        with pytest.raises(TypeError):
            validate_point_cloud_input(None)

        with pytest.raises(TypeError):
            validate_point_cloud_input("not_a_pcd")

        with pytest.raises(ValueError):
            empty_pcd = o3d.geometry.PointCloud()
            validate_point_cloud_input(empty_pcd)

    def test_sor_invalid_params(self, sample_point_cloud):
        """SOR icin gecersiz hiperparametre denetimi."""
        with pytest.raises(ValueError):
            remove_statistical_outliers(sample_point_cloud, nb_neighbors=0)

        with pytest.raises(ValueError):
            remove_statistical_outliers(sample_point_cloud, std_ratio=-1.0)

    def test_remove_statistical_outliers(self, sample_point_cloud):
        """SOR filtresi uzak aykiri noktalari basariyla temizlemelidir."""
        initial_count = len(sample_point_cloud.points)
        clean_pcd, inliers = remove_statistical_outliers(
            sample_point_cloud,
            nb_neighbors=15,
            std_ratio=1.5
        )

        assert len(clean_pcd.points) < initial_count
        assert len(clean_pcd.points) == len(inliers)
        # Uzak aykiri noktalar temizlenmis olmali (Maksimum koordinat 100'den kucuk)
        clean_pts = np.asarray(clean_pcd.points)
        assert np.max(np.abs(clean_pts)) < 100.0

    def test_remove_radius_outliers(self, sample_point_cloud):
        """ROR filtresi izole noktalari basariyla temizlemelidir."""
        clean_pcd, inliers = remove_radius_outliers(
            sample_point_cloud,
            nb_points=8,
            radius=6.0
        )
        assert len(clean_pcd.points) <= len(sample_point_cloud.points)
        assert len(clean_pcd.points) == len(inliers)

    def test_crop_bounding_box(self, sample_point_cloud):
        """Uzamsal sinirlayici kutu filtresi dogrulugu."""
        min_b = np.array([-10.0, -10.0, -10.0])
        max_b = np.array([10.0, 10.0, 10.0])

        cropped = crop_bounding_box(sample_point_cloud, min_b, max_b)
        cropped_pts = np.asarray(cropped.points)

        assert len(cropped_pts) > 0
        assert np.all(cropped_pts >= min_b)
        assert np.all(cropped_pts <= max_b)

        # Gecersiz sinirlar
        with pytest.raises(ValueError):
            crop_bounding_box(sample_point_cloud, max_b, min_b)

    def test_filter_dental_point_cloud_pipeline(self, sample_point_cloud):
        """Butunlesik dental temizleme fonksiyonu."""
        filtered = filter_dental_point_cloud(
            sample_point_cloud,
            voxel_size=0.5,
            sor_neighbors=10,
            sor_std_ratio=1.5
        )

        assert isinstance(filtered, o3d.geometry.PointCloud)
        assert len(filtered.points) > 0
        assert len(filtered.points) < len(sample_point_cloud.points)

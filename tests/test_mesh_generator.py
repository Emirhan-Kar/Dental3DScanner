import os
import pytest
import numpy as np
import open3d as o3d

from processing.mesh_generator import MeshGenerator


class TestMeshGenerator:
    def test_init_defaults(self):
        """MeshGenerator varsayılan parametreler."""
        mg = MeshGenerator()
        assert mg.method == "poisson"
        assert mg.poisson_depth == 9
        assert mg.density_trim_percentile == 5.0
        assert mg.smooth_iterations == 3

    def test_init_invalid_parameters(self):
        """Geçersiz parametre kontrolleri."""
        with pytest.raises(ValueError):
            MeshGenerator(method="gecersiz_yontem")
        with pytest.raises(ValueError):
            MeshGenerator(poisson_depth=3)
        with pytest.raises(ValueError):
            MeshGenerator(density_trim_percentile=60.0)
        with pytest.raises(ValueError):
            MeshGenerator(smooth_iterations=-1)

    def test_reconstruct_poisson(self, sample_point_cloud):
        """Poisson rekonstrüksiyonu ile mesh üretimi."""
        mg = MeshGenerator(poisson_depth=7, density_trim_percentile=5.0)
        mesh = mg.reconstruct_poisson(sample_point_cloud)

        assert mesh is not None
        assert isinstance(mesh, o3d.geometry.TriangleMesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0
        assert mesh.has_vertex_normals()

    def test_reconstruct_ball_pivoting(self, sample_point_cloud):
        """Ball Pivoting algoritması ile mesh üretimi."""
        mg = MeshGenerator()
        mesh = mg.reconstruct_ball_pivoting(sample_point_cloud)

        assert mesh is not None
        assert isinstance(mesh, o3d.geometry.TriangleMesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0

    def test_generate_mesh_dispatch(self, sample_point_cloud):
        """generate_mesh fonksiyonu dogru algoritmaya yonlendirmeli."""
        mg = MeshGenerator()
        mesh_p = mg.generate_mesh(sample_point_cloud, method="poisson")
        mesh_b = mg.generate_mesh(sample_point_cloud, method="ball_pivoting")

        assert len(mesh_p.triangles) > 0
        assert len(mesh_b.triangles) > 0

    def test_generate_mesh_invalid_input(self):
        """Yetersiz veya None nokta bulutu denetimi."""
        mg = MeshGenerator()
        with pytest.raises(ValueError):
            mg.generate_mesh(None)

        tiny_pcd = o3d.geometry.PointCloud()
        tiny_pcd.points = o3d.utility.Vector3dVector(np.random.rand(5, 3))
        with pytest.raises(ValueError):
            mg.generate_mesh(tiny_pcd)

    def test_export_mesh_stl_and_obj(self, sample_point_cloud, tmp_path):
        """STL ve OBJ disari aktarma testi."""
        mg = MeshGenerator(poisson_depth=7)
        mesh = mg.reconstruct_poisson(sample_point_cloud)

        stl_path = str(tmp_path / "model.stl")
        obj_path = str(tmp_path / "model.obj")

        assert mg.export_mesh(mesh, stl_path) is True
        assert os.path.exists(stl_path)
        assert os.path.getsize(stl_path) > 0

        assert mg.export_mesh(mesh, obj_path) is True
        assert os.path.exists(obj_path)
        assert os.path.getsize(obj_path) > 0

    def test_from_config(self, default_config):
        """Konfigürasyondan MeshGenerator oluşturma."""
        mg = MeshGenerator.from_config(default_config)
        assert mg.poisson_depth == default_config.mesh_generator.poisson_depth

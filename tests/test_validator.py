import os
import json
import pytest
import numpy as np
import open3d as o3d
import cv2

from utils.validator import (
    validate_mesh,
    validate_annotation,
    validate_dental_sample,
    validate_point_cloud,
    validate_image_file,
    ValidationResult,
    get_safe_path
)
from ai_model.dataset import DentalMeshDataset


@pytest.fixture
def valid_mesh_file(tmp_path):
    """Gecerli bir sentetik .obj mesh dosyasi olusturur."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=5.0, resolution=20)
    file_path = tmp_path / "valid_sphere.obj"
    safe_target = get_safe_path(str(file_path))
    o3d.io.write_triangle_mesh(safe_target, mesh)
    return safe_target


@pytest.fixture
def valid_annotation_file(tmp_path):
    """Gecerli bir sentetik .json FDI etiket dosyasi olusturur."""
    data = {
        "id_patient": "TEST_PATIENT_01",
        "labels": [0, 11, 12, 21, 22, 31, 41]
    }
    file_path = tmp_path / "valid_labels.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(file_path)


class TestMeshValidator:
    def test_validate_mesh_valid(self, valid_mesh_file):
        """Gecerli mesh dogrulamasi."""
        res = validate_mesh(valid_mesh_file, min_vertices=50)
        assert res.is_valid is True
        assert res.details["vertex_count"] > 50
        assert res.details["bounding_box_diagonal"] > 0

    def test_validate_mesh_nonexistent(self):
        """Mevcut olmayan dosya denetimi."""
        res = validate_mesh("olmayan_dosya.obj")
        assert res.is_valid is False
        assert "mevcut degil" in res.error_message

    def test_validate_mesh_empty_file(self, tmp_path):
        """0 baytlik bos dosya denetimi."""
        empty_file = tmp_path / "empty.obj"
        empty_file.touch()
        res = validate_mesh(str(empty_file))
        assert res.is_valid is False
        assert "0 bayt" in res.error_message

    def test_validate_mesh_too_few_vertices(self, tmp_path):
        """Minimumdan az kose iceren mesh denetimi."""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
        file_path = tmp_path / "tiny.obj"
        safe_target = get_safe_path(str(file_path))
        o3d.io.write_triangle_mesh(safe_target, mesh)

        res = validate_mesh(safe_target, min_vertices=100)
        assert res.is_valid is False
        assert "yetersiz" in res.error_message


class TestAnnotationValidator:
    def test_validate_annotation_valid(self, valid_annotation_file):
        """Gecerli FDI etiket dosyasi dogrulamasi."""
        res = validate_annotation(valid_annotation_file)
        assert res.is_valid is True
        assert res.details["label_count"] == 7
        assert res.details["patient_id"] == "TEST_PATIENT_01"

    def test_validate_annotation_corrupt_json(self, tmp_path):
        """Bozuk JSON syntax denetimi."""
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{bu: bozuk bir json,", encoding="utf-8")
        res = validate_annotation(str(corrupt_file))
        assert res.is_valid is False
        assert "bozuk" in res.error_message

    def test_validate_annotation_invalid_fdi(self, tmp_path):
        """Gecersiz FDI dis numaralari (orn: 99, 105)."""
        data = {"labels": [0, 11, 99, 105]}
        bad_fdi_file = tmp_path / "bad_fdi.json"
        with open(bad_fdi_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        res = validate_annotation(str(bad_fdi_file))
        assert res.is_valid is False
        assert "Gecersiz FDI" in res.error_message

    def test_validate_annotation_missing_labels(self, tmp_path):
        """'labels' anahtari bulunmayan JSON denetimi."""
        data = {"id_patient": "P01", "other_info": 123}
        missing_lbl_file = tmp_path / "no_labels.json"
        with open(missing_lbl_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        res = validate_annotation(str(missing_lbl_file))
        assert res.is_valid is False
        assert "labels" in res.error_message


class TestPointAndImageValidator:
    def test_validate_point_cloud_valid(self):
        """Gecerli nokta bulutu denetimi."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
        res = validate_point_cloud(pcd, min_points=50)
        assert res.is_valid is True

    def test_validate_point_cloud_invalid(self):
        """None veya yetersiz nokta bulutu denetimi."""
        assert validate_point_cloud(None).is_valid is False

        pcd = o3d.geometry.PointCloud()
        assert validate_point_cloud(pcd, min_points=10).is_valid is False

    def test_validate_image_file(self, tmp_path):
        """Gorsel dosya dogrulamasi (Windows Unicode guvenli yazma ve okuma)."""
        img_file = tmp_path / "test.png"
        img_array = (np.ones((50, 50, 3), dtype=np.uint8) * 128)
        _, buf = cv2.imencode(".png", img_array)
        with open(img_file, "wb") as f:
            f.write(buf)

        res = validate_image_file(str(img_file), min_width=20, min_height=20)
        assert res.is_valid is True
        assert res.details["width"] == 50

        assert validate_image_file("olmayan_resim.jpg").is_valid is False


class TestDatasetSafeMode:
    def test_safe_mode_fallback(self, tmp_path):
        """Bozuk bir vaka DataLoader'i cokertmemeli, baska vaka ikamesi yapmali."""
        # 1. Gecerli bir vaka olustur
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=5.0, resolution=20)
        valid_obj = tmp_path / "valid.obj"
        safe_valid_obj = get_safe_path(str(valid_obj))
        o3d.io.write_triangle_mesh(safe_valid_obj, mesh)

        valid_json = tmp_path / "valid.json"
        valid_json.write_text(json.dumps({"labels": [0, 11] * 1000}), encoding="utf-8")

        # 2. Bozuk bir vaka olustur (0 baytlik .obj)
        bad_obj = tmp_path / "bad.obj"
        bad_obj.touch()
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{}", encoding="utf-8")

        dataset = DentalMeshDataset(data_root=".", parts=[], num_points=100, safe_mode=True)
        dataset.samples = [
            (safe_valid_obj, str(valid_json)),
            (str(bad_obj), str(bad_json))
        ]

        # Bozuk vaka (indeks 1) cagirildiginda safe_mode devreye girip basariyla ornek dondurur
        sample = dataset[1]
        assert "points" in sample
        assert "labels" in sample
        assert sample["points"].shape[1] == 100

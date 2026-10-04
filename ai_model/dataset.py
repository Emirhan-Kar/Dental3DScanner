import os
import glob
import json
import logging
import random
import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d
from utils.logger import get_logger
from utils.validator import validate_dental_sample, get_safe_path

logger = get_logger(__name__)

# FDI Diş Standart Numaraları (11-48 arası 32 diş + 0 Diş Eti)
FDI_CLASSES = [0] + [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48
]

LABEL_TO_IDX = {fdi: idx for idx, fdi in enumerate(FDI_CLASSES)}
IDX_TO_LABEL = {idx: fdi for idx, fdi in enumerate(FDI_CLASSES)}


class DentalMeshDataset(data.Dataset):
    def __init__(
        self,
        data_root="data/3DTeethData",
        parts=None,
        num_points=8192,
        augment=False,
        filter_invalid=False,
        safe_mode=True
    ):
        """
        Teeth3DS Veri Yükleyici (PyTorch).

        Args:
            data_root (str): Veri seti kök dizini.
            parts (list): Yüklenecek parça klasörleri (ör. ["data_part_1"]).
            num_points (int): Her örnekten çekilecek nokta sayısı (pozitif tamsayı).
            augment (bool): Rastgele döndürme ve gürültü ekleme aktifliği.
            filter_invalid (bool): Başlangıçta tüm vakaları doğrulayıp bozuk olanları ayıklar.
            safe_mode (bool): Okuma hatası olan vakalarda eğitimi çökertmek yerine güvenli vaka ikamesi yapar.

        Raises:
            TypeError: Argüman tipleri geçersiz ise.
            ValueError: num_points <= 0 ise.
            FileNotFoundError: data_root mevcut değilse.
        """
        super().__init__()

        if not isinstance(data_root, str):
            raise TypeError(f"data_root string olmalıdır, alınan: {type(data_root).__name__}")
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Veri seti kök dizini bulunamadı: {os.path.abspath(data_root)}")

        if not isinstance(num_points, int) or num_points <= 0:
            raise ValueError(f"num_points pozitif bir tamsayı olmalıdır, alınan: {num_points}")

        if parts is None:
            parts = ["data_part_1"]
        elif not isinstance(parts, (list, tuple)):
            raise TypeError(f"parts liste veya demet (tuple) olmalıdır, alınan: {type(parts).__name__}")

        self.data_root = data_root
        self.num_points = num_points
        self.augment = bool(augment)
        self.safe_mode = bool(safe_mode)
        self.samples = []

        # Part klasörlerindeki tüm .obj ve .json çiftlerini topla
        for part in parts:
            part_path = os.path.join(data_root, str(part))
            if not os.path.exists(part_path):
                logger.warning(f"Belirtilen veri seti alt parçası bulunamadı, atlanıyor: {part_path}")
                continue

            obj_files = glob.glob(os.path.join(part_path, "*", "*", "*.obj"))
            for obj_f in obj_files:
                base_dir = os.path.dirname(obj_f)
                base_name = os.path.splitext(os.path.basename(obj_f))[0]
                json_f = os.path.join(base_dir, f"{base_name}.json")

                if os.path.exists(json_f):
                    self.samples.append((obj_f, json_f))
                else:
                    logger.debug(f"JSON etiketi bulunamadı, atlandı: {obj_f}")

        # İsteğe bağlı ön doğrulama ve bozuk vakaları ayıklama
        if filter_invalid and self.samples:
            valid_samples = []
            for obj_f, json_f in self.samples:
                res = validate_dental_sample(obj_f, json_f, min_vertices=100)
                if res.is_valid:
                    valid_samples.append((obj_f, json_f))
                else:
                    logger.warning(f"Bozuk vaka ayiklandi ({obj_f}): {res.error_message}")
            filtered_count = len(self.samples) - len(valid_samples)
            if filtered_count > 0:
                logger.info(f"On dogrulama ile {filtered_count} adet gecersiz vaka ayiklandi.")
            self.samples = valid_samples

        logger.info(f"[{', '.join(parts)}] icinde toplam {len(self.samples)} adet etiketli vaka bulundu.")

    def __len__(self):
        return len(self.samples)

    def _normalize_points(self, points):
        """Merkezleme ve birim küreye ölçekleme yapar."""
        if len(points) == 0:
            return points
        centroid = np.mean(points, axis=0)
        points -= centroid
        distances = np.sqrt(np.sum(points ** 2, axis=1))
        max_distance = np.max(distances) if len(distances) > 0 else 0.0
        if max_distance > 1e-8:
            points /= max_distance
        return points

    def _augment(self, points):
        """Z ekseni etrafında rastgele döndürme ve küçük Gaussian gürültü ekler."""
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,             0,              1]
        ], dtype=points.dtype)

        # Pozisyonları döndür
        points[:, :3] = np.dot(points[:, :3], rotation_matrix)
        # Varsa normalleri de aynı matrisle döndür
        if points.shape[1] >= 6:
            points[:, 3:6] = np.dot(points[:, 3:6], rotation_matrix)

        jitter = np.random.normal(0.0, 0.01, size=points[:, :3].shape).astype(points.dtype)
        points[:, :3] += jitter
        return points

    def _load_sample(self, idx):
        """Tek bir örneği yükler ve tensörlere dönüştürür."""
        obj_path, json_path = self.samples[idx]

        # 1. Mesh'i Open3D ile oku
        mesh = o3d.io.read_triangle_mesh(get_safe_path(obj_path))
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if len(vertices) == 0:
            raise ValueError(f"Mesh dosyası boş veya köşe noktası içermiyor: {obj_path}")

        try:
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            if len(normals) != len(vertices):
                normals = np.zeros_like(vertices, dtype=np.float32)
        except Exception:
            normals = np.zeros_like(vertices, dtype=np.float32)

        # 2. JSON etiketlerini oku
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_labels = meta.get("labels", [])
        if not isinstance(raw_labels, list):
            raw_labels = []

        if len(raw_labels) == len(vertices):
            vertex_labels = np.array(raw_labels, dtype=np.int64)
        else:
            vertex_labels = np.zeros(len(vertices), dtype=np.int64)
            faces = np.asarray(mesh.triangles)
            for f_idx, face in enumerate(faces):
                if f_idx < len(raw_labels):
                    lbl_val = raw_labels[f_idx]
                    if isinstance(lbl_val, (int, float)):
                        vertex_labels[face] = int(lbl_val)

        # 3. Sabit sayıda nokta örnekle
        num_v = len(vertices)
        if num_v >= self.num_points:
            choice = np.random.choice(num_v, self.num_points, replace=False)
        else:
            choice = np.random.choice(num_v, self.num_points, replace=True)

        selected_vertices = vertices[choice]
        selected_normals = normals[choice]
        selected_labels = vertex_labels[choice]

        # 4. Normalizasyon ve Veri Artırma
        selected_vertices = self._normalize_points(selected_vertices)
        features = np.concatenate([selected_vertices, selected_normals], axis=-1)

        # NaN/Inf temizliği
        if not np.all(np.isfinite(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.augment:
            features = self._augment(features)

        mapped_labels = np.array([LABEL_TO_IDX.get(int(lbl), 0) for lbl in selected_labels], dtype=np.int64)

        return {
            "points": torch.tensor(features, dtype=torch.float32).transpose(0, 1),
            "labels": torch.tensor(mapped_labels, dtype=torch.long),
            "patient_id": str(meta.get("id_patient", "unknown"))
        }

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"İndeks sınır dışı: {idx} (Veri seti boyutu: {len(self.samples)})")

        if not self.safe_mode:
            return self._load_sample(idx)

        # Güvenli mod: Bir vaka bozuksa en fazla 5 kez alternatif vaka dener
        attempts = 0
        current_idx = idx
        while attempts < 5:
            try:
                return self._load_sample(current_idx)
            except Exception as e:
                attempts += 1
                bad_obj = self.samples[current_idx][0]
                logger.warning(
                    f"Bozuk veri ornegi ({bad_obj}): {e}. "
                    f"Guvenli mod devrede, baska ornek deneniyor ({attempts}/5)..."
                )
                current_idx = random.randint(0, len(self.samples) - 1)

        # 5 deneme de başarısız olursa son hatayı fırlat
        raise RuntimeError(f"Ust uste 5 vaka yuklenemedi, veri seti butunlugu bozulmus olabilir.")
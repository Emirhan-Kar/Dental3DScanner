import os
import sys
import glob
import argparse
import torch
import numpy as np
import open3d as o3d

# Proje kok dizinini sys.path'e ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ai_model.network import DentalPointNetSeg
from ai_model.dataset import FDI_CLASSES, IDX_TO_LABEL
from ui.viewer import DentalViewer, get_fdi_color_palette
from utils.logger import get_logger
from utils.config import load_config, get_base_parser
from utils.validator import get_safe_path, validate_mesh

logger = get_logger(__name__)


def predict_dental_mesh(
    obj_path: str,
    checkpoint_path: str = None,
    num_points: int = 4096,
    config = None,
    no_gui: bool = False,
    output_ply: str = None
):
    """
    Egitilmis PointNet modeli ile 3D dis modeli uzerinde segmentasyon tahmini yapar.

    Args:
        obj_path (str): Segment edilecek 3D model yolu (.obj, .ply, .stl).
        checkpoint_path (str, optional): Model agirlik dosyasi (.pth).
        num_points (int): Modele beslenecek orneklem nokta sayisi.
        config (ConfigNode, optional): Merkezi konfigurasyon nesnesi.
        no_gui (bool): True ise 3D pencere acilmaz (otomasyon/test modu).
        output_ply (str, optional): Renklendirilmis tahmin nokta bulutunun kaydedilecegi yol.

    Returns:
        dict: {'predictions': ndarray, 'point_cloud': o3d.geometry.PointCloud, 'classes': list}
    """
    if config is None:
        config = load_config()

    if checkpoint_path is None:
        checkpoint_path = config.ai_model.checkpoint_path

    if not isinstance(obj_path, str) or not os.path.exists(obj_path):
        logger.error(f"Girdi mesh dosyasi bulunamadi: {obj_path}")
        return None

    # Mesh gecerliligini denetle
    mesh_val = validate_mesh(obj_path, min_vertices=50)
    if not mesh_val.is_valid:
        logger.error(f"Gecersiz mesh dosyasi: {mesh_val.error_message}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Yapay zeka cikarim cihazi: {device}")

    # 1. Modeli Yukle
    in_channels = getattr(config.ai_model, "in_channels", 6)
    num_classes = getattr(config.ai_model, "num_classes", 33)
    model = DentalPointNetSeg(in_channels=in_channels, num_classes=num_classes).to(device)

    if not os.path.exists(checkpoint_path):
        logger.warning(
            f"Checkpoint dosyasi bulunamadi ({checkpoint_path})! "
            f"Model baslatilmamis agirliklarla (mock inference) calisacak."
        )
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Model agirliklari basariyla yuklendi: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Checkpoint yuklenirken hata olustu: {e}")
            return None

    model.eval()

    # 2. Ham Mesh'i Oku ve Nokta Ornekle (Windows Unicode guvenli)
    safe_path = get_safe_path(obj_path)
    try:
        mesh = o3d.io.read_triangle_mesh(safe_path)
    except Exception as e:
        logger.error(f"Mesh dosyasi okunamadi: {e}")
        return None

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    logger.info(f"Mesh uzerinden {num_points} nokta ornekleniyor: {obj_path}")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    # 3. Normalizasyon (Merkezleme ve birim kureye olcekleme)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    scale = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
    points_normalized = points_centered / (scale + 1e-8)

    features = np.concatenate([points_normalized, normals], axis=1)  # (N, 6)
    tensor_input = torch.from_numpy(features.T).unsqueeze(0).to(device)  # (1, 6, N)

    # 4. Cikarim (Inference)
    logger.info("3D Nokta Bulutu Dis Segmentasyonu cikarimi yapiliyor...")
    with torch.no_grad():
        logits = model(tensor_input)  # (1, num_classes, N)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (N,)

    unique_classes = np.unique(preds)
    fdi_detected = [IDX_TO_LABEL.get(int(idx), 0) for idx in unique_classes]
    logger.info(f"Cikarim tamamlandi! Tespit edilen sinif sayisi: {len(unique_classes)}")
    logger.info(f"Tespit edilen FDI Diş Numaralari: {fdi_detected}")

    # 5. Noktalari Sinif Renk Paleti ile Boya
    palette = get_fdi_color_palette()
    point_colors = np.zeros((len(preds), 3), dtype=np.float64)
    for i, p in enumerate(preds):
        cls_idx = int(p) if 0 <= int(p) < len(palette) else 0
        point_colors[i] = palette[cls_idx]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # 6. Istege Bagli PLY Dosyasina Kaydet
    if output_ply:
        try:
            out_dir = os.path.dirname(output_ply)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            safe_out = get_safe_path(output_ply)
            o3d.io.write_point_cloud(safe_out, pcd)
            logger.info(f"Segmentasyon sonucu PLY olarak kaydedildi: {output_ply}")
        except Exception as e:
            logger.error(f"PLY dosyasi kaydedilemedi: {e}")

    # 7. Gorsellestir
    if not no_gui:
        viewer = DentalViewer(window_name="DentalPointNet 3D Segmentasyon")
        viewer.show_segmented_point_cloud(pcd, preds)

    return {
        "predictions": preds,
        "point_cloud": pcd,
        "classes": unique_classes.tolist(),
        "fdi_classes": fdi_detected,
        "num_points": len(points)
    }


if __name__ == "__main__":
    parser = get_base_parser(description="3D Nokta Bulutu Diş Segmentasyon Çıkarım Aracı")
    parser.add_argument("--mesh", type=str, default=None, help="Segment edilecek 3D mesh dosyasi (.obj, .ply)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Egitilmis model agirlik yolu (.pth)")
    parser.add_argument("--num-points", type=int, default=4096, help="Orneklem nokta sayisi")
    parser.add_argument("--output-ply", type=str, default=None, help="Cikti renklendirilmis nokta bulutu yolu (.ply)")
    parser.add_argument("--no-gui", action="store_true", help="Gorsel pencereyi acma (headless mod)")

    args = parser.parse_args()
    cfg = load_config(args.config)

    mesh_file = args.mesh
    if not mesh_file:
        sample_objs = glob.glob("data/3DTeethData/data_part_1/*/*/*.obj")
        if sample_objs:
            mesh_file = sample_objs[0]
            logger.info(f"Otomatik ornek vaka secildi: {mesh_file}")
        else:
            logger.error("Girdi mesh dosyasi belirtilmedi (--mesh <yol>) ve ornek veri bulunamadi.")
            sys.exit(1)

    predict_dental_mesh(
        obj_path=mesh_file,
        checkpoint_path=args.checkpoint,
        num_points=args.num_points,
        config=cfg,
        no_gui=args.no_gui,
        output_ply=args.output_ply
    )
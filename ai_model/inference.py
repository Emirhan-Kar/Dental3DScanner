import os
import sys
import glob
import json
import torch
import numpy as np
import open3d as o3d

# Proje kok dizinini sys.path'e ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ai_model.network import model_olustur, checkpoint_yukle
from ai_model.dataset import FDI_CLASSES, IDX_TO_LABEL, mesh_oku, noktalari_normalize_et, mesh_geometri_oku, o3d_mesh_olustur
from ui.viewer import DentalViewer, get_fdi_color_palette
from utils.logger import get_logger
from utils.config import load_config, get_base_parser
from utils.validator import get_safe_path, validate_mesh

logger = get_logger(__name__)


@torch.no_grad()
def koseleri_tahmin_et(model, koseler, normaller, nokta_sayisi=8192, cihaz="cpu", tur_sayisi=3, batch_boyutu=8, tohum=0):
    """Mesh'in TUM koseleri icin sinif tahmini yapar.

    Model sabit sayida (nokta_sayisi) nokta ile egitildigi icin, kose listesi her turda
    rastgele karistirilip nokta_sayisi'lik parcalara bolunur. Boylece her parca egitimdeki
    gibi tum ceneyi kapsayan rastgele bir alt kume olur. Her kose 'tur_sayisi' kez farkli
    komsulukla gorulur ve olasiliklar ortalanir (test-time voting).

    Returns:
        tahminler (N,) sinif indeksleri, olasiliklar (N, C)
    """
    model.eval()
    N = len(koseler)
    rng = np.random.RandomState(tohum)
    toplam = None
    parcalar = []
    for _ in range(max(1, tur_sayisi)):
        sira = rng.permutation(N)
        eksik = (-N) % nokta_sayisi
        if eksik:
            sira = np.concatenate([sira, rng.choice(N, eksik, replace=N < eksik)])
        parcalar.extend(sira.reshape(-1, nokta_sayisi))

    for b in range(0, len(parcalar), batch_boyutu):
        grup = parcalar[b:b + batch_boyutu]
        girdiler = []
        for idx in grup:
            oz = np.concatenate([noktalari_normalize_et(koseler[idx]), normaller[idx]], axis=1).T
            girdiler.append(oz)
        x = torch.from_numpy(np.stack(girdiler).astype(np.float32)).to(cihaz)
        with torch.autocast(device_type=torch.device(cihaz).type, enabled=torch.device(cihaz).type == "cuda"):
            olas = torch.softmax(model(x).float(), dim=1).cpu().numpy()  # (b, C, n)
        if toplam is None:
            toplam = np.zeros((N, olas.shape[1]), dtype=np.float32)
        for idx, o in zip(grup, olas):
            np.add.at(toplam, idx, o.T)

    toplam /= np.maximum(toplam.sum(axis=1, keepdims=True), 1e-8)
    return toplam.argmax(axis=1), toplam


def predict_dental_mesh(
    obj_path: str,
    checkpoint_path: str = None,
    num_points: int = None,
    config=None,
    no_gui: bool = False,
    output_ply: str = None,
    model_name: str = None,
    votes: int = 3,
):
    """
    Egitilmis model ile 3D dis mesh'inin her kosesini FDI sinifina atar.

    Args:
        obj_path (str): Segment edilecek 3D model yolu (.obj, .ply, .stl).
        checkpoint_path (str, optional): Model agirlik dosyasi (.pth).
        num_points (int): Modelin tek seferde gordugu nokta sayisi (checkpoint'ten okunur).
        config (ConfigNode, optional): Merkezi konfigurasyon nesnesi.
        no_gui (bool): True ise 3D pencere acilmaz.
        output_ply (str, optional): Renklendirilmis segmentasyon MESH'inin kaydedilecegi yol (.ply/.obj).
            Ayni isimle Teeth3DS formatinda etiket dosyasi (.json) da yazilir.
        model_name (str): Eski (salt state_dict) checkpoint'ler icin mimari adi.
        votes (int): Test-time oylama tur sayisi.

    Returns:
        dict: predictions (kose bazli), fdi_labels, mesh, point_cloud, classes, fdi_classes, num_points
    """
    if config is None:
        config = load_config()
    ai_cfg = config.ai_model
    if checkpoint_path is None:
        checkpoint_path = ai_cfg.checkpoint_path

    if not isinstance(obj_path, str) or not os.path.exists(obj_path):
        logger.error(f"Girdi mesh dosyasi bulunamadi: {obj_path}")
        return None

    mesh_val = validate_mesh(obj_path, min_vertices=50)
    if not mesh_val.is_valid:
        logger.error(f"Gecersiz mesh dosyasi: {mesh_val.error_message}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Yapay zeka cikarim cihazi: {device}")

    in_channels = getattr(ai_cfg, "in_channels", 6)
    num_classes = getattr(ai_cfg, "num_classes", 33)
    model_name = model_name or getattr(ai_cfg, "model", "pointnet")
    kayit = {}
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint dosyasi bulunamadi ({checkpoint_path})! Baslatilmamis agirliklarla calisilacak.")
        model = model_olustur(model_name, in_channels, num_classes).to(device)
    else:
        try:
            model, kayit = checkpoint_yukle(checkpoint_path, device, varsayilan_model=model_name,
                                            giris_kanallari=in_channels, sinif_sayisi=num_classes)
            logger.info(f"Model ({kayit.get('model_name')}) agirliklari yuklendi: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Checkpoint yuklenirken hata olustu: {e}")
            return None

    if num_points is None:
        num_points = int(kayit.get("num_points", getattr(ai_cfg, "num_points", 4096)))

    try:
        koseler, normaller, _, _ = mesh_oku(obj_path)
    except Exception as e:
        logger.error(f"Mesh dosyasi okunamadi: {e}")
        return None

    logger.info(f"{len(koseler)} kosenin tamami icin segmentasyon yapiliyor (parca={num_points}, oylama={votes})...")
    preds, _ = koseleri_tahmin_et(model, koseler, normaller, nokta_sayisi=num_points, cihaz=device, tur_sayisi=votes)

    unique_classes = np.unique(preds)
    fdi_detected = [IDX_TO_LABEL.get(int(i), 0) for i in unique_classes]
    fdi_labels = np.array([IDX_TO_LABEL[int(p)] for p in preds], dtype=np.int64)
    logger.info(f"Tespit edilen FDI Dis Numaralari: {fdi_detected}")

    palette = get_fdi_color_palette()
    colors = palette[np.clip(preds, 0, len(palette) - 1)]

    mesh = o3d_mesh_olustur(*mesh_geometri_oku(obj_path), renkler=colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(koseler.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if output_ply:
        try:
            out_dir = os.path.dirname(output_ply)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            o3d.io.write_triangle_mesh(get_safe_path(output_ply), mesh, write_vertex_colors=True)
            with open(os.path.splitext(output_ply)[0] + ".json", "w", encoding="utf-8") as f:
                json.dump({"id_patient": os.path.basename(os.path.dirname(obj_path)),
                           "labels": fdi_labels.tolist()}, f)
            logger.info(f"Segmentasyon mesh'i ve etiketleri kaydedildi: {output_ply}")
        except Exception as e:
            logger.error(f"Cikti kaydedilemedi: {e}")

    if not no_gui:
        viewer = DentalViewer(window_name="3D Dis Segmentasyonu")
        viewer.show_mesh(mesh, window_name="3D Dis Segmentasyonu (FDI)")

    return {
        "predictions": preds,
        "fdi_labels": fdi_labels,
        "mesh": mesh,
        "point_cloud": pcd,
        "classes": unique_classes.tolist(),
        "fdi_classes": fdi_detected,
        "num_points": len(koseler),
    }


if __name__ == "__main__":
    parser = get_base_parser(description="3D Dis Segmentasyon Cikarim Araci")
    parser.add_argument("--mesh", type=str, default=None, help="Segment edilecek 3D mesh dosyasi (.obj, .ply)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Egitilmis model agirlik yolu (.pth)")
    parser.add_argument("--model", type=str, default=None, choices=["pointnet", "dgcnn"], help="Eski checkpoint'ler icin mimari")
    parser.add_argument("--num-points", type=int, default=None, help="Parca basina nokta sayisi (varsayilan: checkpoint)")
    parser.add_argument("--votes", type=int, default=3, help="Test-time oylama tur sayisi")
    parser.add_argument("--output-ply", type=str, default=None, help="Renkli segmentasyon mesh cikti yolu (.ply)")
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
        output_ply=args.output_ply,
        model_name=args.model,
        votes=args.votes,
    )

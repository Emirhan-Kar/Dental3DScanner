import os
import sys
import argparse
import copy
import yaml

# Proje kok dizinini sys.path'e ekle (dogrudan calistirmalari desteklemek icin)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = os.path.join("configs", "config.yaml")

# Tum sistem icin temel varsayilan degerler (Dosya bulunamazsa bile calismayi garanti eder)
DEFAULT_CONFIG = {
    "camera": {
        "params_yaml": "calibration/camera_params.yaml",
        "fallback_matrix": [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0]
        ],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "width": 1920,
        "height": 1080,
        "laser_plane": [0.0, -1.0, 0.5, -150.0]
    },
    "laser_detector": {
        "lower_hsv": [0, 100, 100],
        "upper_hsv": [10, 255, 255],
        "intensity_threshold": 50
    },
    "icp_stitcher": {
        "voxel_size": 0.4,
        "ransac_distance_multiplier": 3.0,
        "icp_distance_multiplier": 1.5,
        "icp_max_iterations": 80,
        "ransac_max_iterations": 100000
    },
    "calibration": {
        "chessboard_size": [9, 6],
        "square_size": 25.0,
        "image_folder": "data/2DTeethData",
        "output_path": "calibration/camera_params.yaml"
    },
    "ai_model": {
        "in_channels": 6,
        "num_classes": 33,
        "num_points": 4096,
        "batch_size": 4,
        "epochs": 120,
        "learning_rate": 0.001,
        "lr_step_size": 10,
        "lr_gamma": 0.5,
        "checkpoint_path": "ai_model/best_dental_model.pth",
        "data_root": "data/3DTeethData",
        "parts": [
            "data_part_1",
            "data_part_2",
            "data_part_3",
            "data_part_4",
            "data_part_5",
            "data_part_6"
        ]
    },
    "mesh_generator": {
        "method": "poisson",
        "poisson_depth": 9,
        "density_trim_percentile": 5,
        "smooth_iterations": 3,
        "default_output": "output/dental_model.stl"
    },
    "logging": {
        "level": "INFO",
        "log_file": "logs/dental_scanner.log",
        "console": True,
        "max_bytes": 10485760,
        "backup_count": 5
    }
}


class ConfigNode(dict):
    """
    Sozluk ve nokta erisimi (dot notation) saglayan ozel konfigurasyon dugumu.
    Ornek: cfg.ai_model.batch_size veya cfg['ai_model']['batch_size']
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, ConfigNode):
                self[key] = ConfigNode(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Konfigurasyonda '{key}' alani bulunamadi.")

    def __setattr__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Konfigurasyonda '{key}' alani bulunamadi.")

    def to_dict(self):
        """Dugumu standart Python sozlugune cevirir."""
        res = {}
        for k, v in self.items():
            if isinstance(v, ConfigNode):
                res[k] = v.to_dict()
            elif isinstance(v, list):
                res[k] = [item.to_dict() if isinstance(item, ConfigNode) else item for item in v]
            else:
                res[k] = v
        return res


def _deep_update(base_dict, update_dict):
    """base_dict uzerine update_dict alanlarini yinelemeli (recursive) uygular."""
    for key, val in update_dict.items():
        if isinstance(val, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], val)
        else:
            base_dict[key] = copy.deepcopy(val)
    return base_dict


def load_config(config_path=None, overrides=None):
    """
    YAML konfigurasyon dosyasini okur ve varsayilanlarla birlestirir.

    Args:
        config_path (str, optional): YAML dosya yolu. None ise varsayilan yol kullanilir.
        overrides (dict, optional): Dinamik degistirilecek parametreler sozluk yapisi.

    Returns:
        ConfigNode: Erisilebilir konfigurasyon nesnesi.
    """
    merged = copy.deepcopy(DEFAULT_CONFIG)

    target_path = config_path if config_path else DEFAULT_CONFIG_PATH

    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                loaded_yaml = yaml.safe_load(f)
            if isinstance(loaded_yaml, dict):
                _deep_update(merged, loaded_yaml)
                logger.info(f"Konfigurasyon basariyla yuklendi: {target_path}")
            else:
                logger.warning(f"Konfigurasyon dosyasi bos veya gecersiz ({target_path}), varsayilanlar kullaniliyor.")
        except Exception as e:
            logger.error(f"Konfigurasyon dosyasi okunamadi ({target_path}): {e}. Varsayilan degerler kullanilacak.")
    else:
        if config_path:
            logger.warning(f"Belirtilen konfigurasyon dosyasi bulunamadi: {config_path}. Varsayilan degerler kullaniliyor.")
        else:
            logger.info("configs/config.yaml bulunamadi, dahili varsayilan degerler aktif edildi.")

    # Opsiyonel parametre ezme (overrides)
    if overrides and isinstance(overrides, dict):
        _deep_update(merged, overrides)

    return ConfigNode(merged)


def save_config(config, output_path):
    """
    Konfigurasyon nesnesini YAML dosyasi olarak kaydeder.

    Args:
        config (ConfigNode veya dict): Kaydedilecek konfigurasyon nesnesi.
        output_path (str): Hedef dosya yolu.

    Returns:
        bool: Basariliysa True.
    """
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        data = config.to_dict() if isinstance(config, ConfigNode) else config
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Aktif konfigurasyon basariyla kaydedildi: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Konfigurasyon kaydedilemedi ({output_path}): {e}")
        return False


def get_base_parser(description="Dental Scanner Engine"):
    """
    Temel komut satiri (CLI / argparse) ayristiricisini olusturur.
    Tüm scriptler (train, calibration, main vb.) bu parser'i genisleterek kullanabilir.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Konfigurasyon YAML dosyasinin yolu (varsayilan: configs/config.yaml)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Loglama detay seviyesi (None ise config.yaml'daki deger kullanilir)"
    )
    return parser


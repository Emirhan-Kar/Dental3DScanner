import os
import sys
import argparse
import copy
import yaml

# Proje kok dizinini sys.path'e ekle (dogrudan calistirmalari desteklemek icin)
KOK_DIZIN = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOK_DIZIN not in sys.path:
    sys.path.insert(0, KOK_DIZIN)

from utils.logger import get_logger

kayitci = get_logger(__name__)

DEFAULT_CONFIG_PATH = os.path.join("configs", "config.yaml")
VARSAYILAN_KONFIG_YOLU = DEFAULT_CONFIG_PATH

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
        "ransac_max_iterations": 100000,
        "registration_mode": "hybrid",
        "min_fitness": 0.45,
        "max_rmse_ratio": 0.55
    },
    "calibration": {
        "chessboard_size": [9, 6],
        "square_size": 25.0,
        "image_folder": "data/2DTeethData",
        "output_path": "calibration/camera_params.yaml"
    },
    "ai_model": {
        "model": "dgcnn",
        "in_channels": 6,
        "num_classes": 33,
        "num_points": 8192,
        "batch_size": 8,
        "epochs": 100,
        "num_workers": 4,
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
            "data_part_6",
            "data_part_7"
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
VARSAYILAN_KONFIG = DEFAULT_CONFIG


class ConfigNode(dict):
    """
    Sozluk ve nokta erisimi (dot notation) saglayan ozel konfigurasyon dugumu.
    Ornek: cfg.ai_model.batch_size veya cfg['ai_model']['batch_size']
    """
    def __init__(self, *argumanlar, **anahtar_argumanlar):
        super().__init__(*argumanlar, **anahtar_argumanlar)
        for anahtar, deger in list(self.items()):
            if isinstance(deger, dict) and not isinstance(deger, ConfigNode):
                self[anahtar] = ConfigNode(deger)

    def __getattr__(self, anahtar):
        try:
            return self[anahtar]
        except KeyError:
            raise AttributeError(f"Konfigurasyonda '{anahtar}' alani bulunamadi.")

    def __setattr__(self, anahtar, deger):
        if isinstance(deger, dict) and not isinstance(deger, ConfigNode):
            deger = ConfigNode(deger)
        self[anahtar] = deger

    def __delattr__(self, anahtar):
        try:
            del self[anahtar]
        except KeyError:
            raise AttributeError(f"Konfigurasyonda '{anahtar}' alani bulunamadi.")

    def to_dict(self):
        """Dugumu standart Python sozlugune cevirir."""
        sonuc = {}
        for anahtar, deger in self.items():
            if isinstance(deger, ConfigNode):
                sonuc[anahtar] = deger.to_dict()
            elif isinstance(deger, list):
                sonuc[anahtar] = [oge.to_dict() if isinstance(oge, ConfigNode) else oge for oge in deger]
            else:
                sonuc[anahtar] = deger
        return sonuc


def _deep_update(temel_sozluk, guncelleme_sozlugu):
    """temel_sozluk uzerine guncelleme_sozlugu alanlarini yinelemeli (recursive) uygular."""
    for anahtar, deger in guncelleme_sozlugu.items():
        if isinstance(deger, dict) and anahtar in temel_sozluk and isinstance(temel_sozluk[anahtar], dict):
            _deep_update(temel_sozluk[anahtar], deger)
        else:
            temel_sozluk[anahtar] = copy.deepcopy(deger)
    return temel_sozluk


def load_config(konfig_yolu=None, degisiklikler=None, **ek_parametreler):
    """
    YAML konfigurasyon dosyasini okur ve varsayilanlarla birlestirir.

    Args:
        konfig_yolu (str, optional): YAML dosya yolu.
        degisiklikler (dict, optional): Dinamik degistirilecek parametreler.

    Returns:
        ConfigNode: Erisilebilir konfigurasyon nesnesi.
    """
    konfig_yolu = ek_parametreler.get("config_path", konfig_yolu)
    degisiklikler = ek_parametreler.get("overrides", degisiklikler)

    birlestirilmis = copy.deepcopy(VARSAYILAN_KONFIG)
    hedef_yol = konfig_yolu if konfig_yolu else VARSAYILAN_KONFIG_YOLU

    if os.path.exists(hedef_yol):
        try:
            with open(hedef_yol, "r", encoding="utf-8") as dosya:
                yuklenen_yaml = yaml.safe_load(dosya)
            if isinstance(yuklenen_yaml, dict):
                _deep_update(birlestirilmis, yuklenen_yaml)
                kayitci.info(f"Konfigurasyon basariyla yuklendi: {hedef_yol}")
            else:
                kayitci.warning(f"Konfigurasyon dosyasi bos veya gecersiz ({hedef_yol}), varsayilanlar kullaniliyor.")
        except Exception as hata:
            kayitci.error(f"Konfigurasyon dosyasi okunamadi ({hedef_yol}): {hata}. Varsayilan degerler kullanilacak.")
    else:
        if konfig_yolu:
            kayitci.warning(f"Belirtilen konfigurasyon dosyasi bulunamadi: {konfig_yolu}. Varsayilan degerler kullaniliyor.")
        else:
            kayitci.info("configs/config.yaml bulunamadi, dahili varsayilan degerler aktif edildi.")

    # Opsiyonel parametre ezme
    if degisiklikler and isinstance(degisiklikler, dict):
        _deep_update(birlestirilmis, degisiklikler)

    return ConfigNode(birlestirilmis)


def save_config(konfig, cikis_yolu, **ek_parametreler):
    """
    Konfigurasyon nesnesini YAML dosyasi olarak kaydeder.

    Args:
        konfig (ConfigNode veya dict): Kaydedilecek konfigurasyon nesnesi.
        cikis_yolu (str): Hedef dosya yolu.

    Returns:
        bool: Basariliysa True.
    """
    konfig = ek_parametreler.get("config", konfig)
    cikis_yolu = ek_parametreler.get("output_path", cikis_yolu)

    try:
        cikis_dizini = os.path.dirname(cikis_yolu)
        if cikis_dizini:
            os.makedirs(cikis_dizini, exist_ok=True)

        veri = konfig.to_dict() if isinstance(konfig, ConfigNode) else konfig
        with open(cikis_yolu, "w", encoding="utf-8") as dosya:
            yaml.dump(veri, dosya, default_flow_style=False, sort_keys=False)

        kayitci.info(f"Aktif konfigurasyon basariyla kaydedildi: {cikis_yolu}")
        return True
    except Exception as hata:
        kayitci.error(f"Konfigurasyon kaydedilemedi ({cikis_yolu}): {hata}")
        return False


def get_base_parser(aciklama="Dental Tarayici Motoru", **ek_parametreler):
    """
    Temel komut satiri (CLI / argparse) ayristiricisini olusturur.
    Tum betikler (train, calib_script vb.) bu ayristiriciyi genisleterek kullanabilir.
    """
    aciklama = ek_parametreler.get("description", aciklama)
    ayristirici = argparse.ArgumentParser(description=aciklama)
    ayristirici.add_argument(
        "--config",
        type=str,
        default=VARSAYILAN_KONFIG_YOLU,
        help="Konfigurasyon YAML dosyasinin yolu (varsayilan: configs/config.yaml)"
    )
    ayristirici.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Loglama detay seviyesi (None ise config.yaml'daki deger kullanilir)"
    )
    return ayristirici

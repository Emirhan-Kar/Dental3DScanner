import os
import pytest
from utils.logger import setup_logger, get_logger
from utils.config import load_config, save_config, ConfigNode


class TestLogger:
    def test_setup_logger(self, tmp_path):
        """Logger kurulumu ve log dosyasina yazma testi."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("TestUnit", log_file=str(log_file), console=False)

        assert logger.name == "TestUnit"
        logger.info("Test log mesaji")

        # Dosya yazildi mi kontrol et
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "Test log mesaji" in content

    def test_get_logger_hierarchical(self):
        """Hiyerarsik modul logger nesnesi dondurme."""
        logger = get_logger("my_module")
        assert "my_module" in logger.name


class TestConfig:
    def test_load_default_config(self):
        """Varsayilan konfigurasyon yukleme ve ana bolumlerin kontrolu."""
        cfg = load_config()
        assert hasattr(cfg, "camera")
        assert hasattr(cfg, "laser_detector")
        assert hasattr(cfg, "icp_stitcher")
        assert hasattr(cfg, "calibration")
        assert hasattr(cfg, "ai_model")
        assert hasattr(cfg, "logging")

    def test_dot_notation_and_dict_sync(self):
        """Dot notation erisimi ve sozluk eslesmesi."""
        cfg = ConfigNode({"section": {"value": 42}})
        assert cfg.section.value == 42
        assert cfg["section"]["value"] == 42

        # Deger degistirme
        cfg.section.value = 100
        assert cfg["section"]["value"] == 100

    def test_missing_attribute_raises(self):
        """Olmayan alan icin AttributeError firlatma."""
        cfg = ConfigNode({"key": "val"})
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent_field

    def test_save_and_reload(self, tmp_path):
        """Konfigurasyonu kaydetme ve tekrar yukleme testi."""
        cfg = load_config()
        cfg.ai_model.batch_size = 16

        saved_path = tmp_path / "saved_config.yaml"
        success = save_config(cfg, str(saved_path))
        assert success is True
        assert saved_path.exists()

        reloaded = load_config(str(saved_path))
        assert reloaded.ai_model.batch_size == 16

    def test_load_config_with_overrides(self):
        """Dinamik sozluk parametre ezme testi."""
        overrides = {"ai_model": {"epochs": 999}}
        cfg = load_config(overrides=overrides)
        assert cfg.ai_model.epochs == 999

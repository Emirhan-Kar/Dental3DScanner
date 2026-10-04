import os
import pytest
from calibration.calib_script import CameraCalibrator


class TestCameraCalibrator:
    def test_init_defaults(self, default_config):
        """Varsayilan kalibrator baslatma."""
        calibrator = CameraCalibrator(config=default_config)
        assert calibrator.chessboard_size == (9, 6)
        assert calibrator.square_size == 25.0
        assert calibrator.output_path == "calibration/camera_params.yaml"

    def test_init_invalid_chessboard_size(self):
        """3x3'ten kucuk satranc tahtasi gecersiz olmali."""
        with pytest.raises(ValueError):
            CameraCalibrator(chessboard_size=(2, 2))
        with pytest.raises(ValueError):
            CameraCalibrator(chessboard_size=(9,))

    def test_init_invalid_square_size(self):
        """Sifir veya negatif kare kenari hatali olmali."""
        with pytest.raises(ValueError):
            CameraCalibrator(square_size=0.0)
        with pytest.raises(ValueError):
            CameraCalibrator(square_size=-10.0)

    def test_process_images_nonexistent_folder(self):
        """Mevcut olmayan klasor FileNotFoundError firlatmali."""
        calibrator = CameraCalibrator()
        with pytest.raises(FileNotFoundError):
            calibrator.process_images("olmayan_kalibrasyon_klasoru")

    def test_process_images_empty_folder(self, tmp_path):
        """Gorsel icermeyen bos klasor ValueError firlatmali."""
        empty_dir = tmp_path / "empty_calib"
        empty_dir.mkdir()
        calibrator = CameraCalibrator()
        with pytest.raises(ValueError):
            calibrator.process_images(str(empty_dir))

import os
import pytest
import numpy as np
import open3d as o3d

from main import DentalScannerPipeline
from ui.viewer import DentalViewer, get_fdi_color_palette
from ai_model.inference import predict_dental_mesh


class TestPipeline:
    def test_pipeline_init(self, default_config):
        """Pipeline baslatma testi."""
        pipeline = DentalScannerPipeline.from_config(default_config)
        assert pipeline.detector is not None
        assert pipeline.triangulator is not None
        assert pipeline.stitcher is not None
        assert pipeline.processed_frames == 0

    def test_pipeline_process_frame_simulation(self, default_config, synthetic_laser_frame):
        """Tekil cerceve isleme testi."""
        pipeline = DentalScannerPipeline.from_config(default_config)
        result = pipeline.process_frame(synthetic_laser_frame)

        assert result["status"] == "success"
        assert len(result["peaks"]) > 0
        assert result["points_3d"].shape[1] == 3
        assert pipeline.processed_frames == 1

    def test_pipeline_run_simulation(self, default_config, tmp_path):
        """Simulasyon taramasi ve PLY cikti kaydi testi."""
        pipeline = DentalScannerPipeline.from_config(default_config)
        out_ply = str(tmp_path / "sim_out.ply")

        pcd = pipeline.run_simulation(num_slices=5, show_2d=False, save_output=out_ply)

        assert pcd is not None
        assert len(pcd.points) > 0
        assert os.path.exists(out_ply)
        assert os.path.getsize(out_ply) > 0


class TestViewer:
    def test_color_palette(self):
        """FDI 33 sinif renk paleti dogrulamasi."""
        palette = get_fdi_color_palette()
        assert palette.shape == (33, 3)
        # Sinif 0 dis eti (pembe tonu)
        assert palette[0, 0] > 0.7

    def test_viewer_init(self):
        """Viewer baslatma."""
        viewer = DentalViewer()
        assert viewer.width == 1280
        assert viewer.height == 720

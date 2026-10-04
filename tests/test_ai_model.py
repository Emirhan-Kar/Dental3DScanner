import pytest
import torch
from ai_model.network import DentalPointNetSeg
from ai_model.dataset import DentalMeshDataset, FDI_CLASSES, LABEL_TO_IDX, IDX_TO_LABEL


class TestDentalPointNetSeg:
    def test_init_valid(self):
        """Model baslatma ve varsayilan parametreler."""
        model = DentalPointNetSeg(in_channels=6, num_classes=33)
        assert model.in_channels == 6
        assert model.num_classes == 33

    def test_init_invalid(self):
        """Gecersiz kanal ve sinif sayilari."""
        with pytest.raises(ValueError):
            DentalPointNetSeg(in_channels=0)
        with pytest.raises(ValueError):
            DentalPointNetSeg(num_classes=-5)

    def test_forward_pass_cpu(self):
        """Forward pass boyut ve cikti tipi dogrulamasi."""
        model = DentalPointNetSeg(in_channels=6, num_classes=33)
        model.eval()

        batch_size = 2
        num_points = 512
        dummy_x = torch.randn(batch_size, 6, num_points)

        with torch.no_grad():
            out = model(dummy_x)

        assert out.shape == (batch_size, 33, num_points)
        assert not torch.isnan(out).any()

    def test_forward_invalid_tensor(self):
        """Gecersiz girdi tensörleri."""
        model = DentalPointNetSeg(in_channels=6, num_classes=33)

        with pytest.raises(TypeError):
            model(None)

        with pytest.raises(ValueError):
            model(torch.randn(2, 3, 100))  # 3 kanal yerine 6 bekleniyor

        with pytest.raises(ValueError):
            model(torch.randn(6, 100))  # 2D tensör


class TestDentalDataset:
    def test_fdi_constants(self):
        """FDI standart numaralama ve sozluk eslesmeleri."""
        assert len(FDI_CLASSES) == 33
        assert FDI_CLASSES[0] == 0  # Dis eti
        assert len(LABEL_TO_IDX) == 33
        assert len(IDX_TO_LABEL) == 33

        for fdi, idx in LABEL_TO_IDX.items():
            assert IDX_TO_LABEL[idx] == fdi

    def test_dataset_init_invalid_root(self):
        """Olmayan dizin FileNotFoundError firlatmali."""
        with pytest.raises(FileNotFoundError):
            DentalMeshDataset(data_root="olmayan_klasor_123")

    def test_dataset_init_invalid_points(self):
        """Sifir veya negatif num_points degeri."""
        with pytest.raises(ValueError):
            DentalMeshDataset(num_points=0)
        with pytest.raises(ValueError):
            DentalMeshDataset(num_points=-100)


class TestCheckpointManagement:
    def test_save_and_load_rich_checkpoint(self, tmp_path):
        """Model, optimizer ve scheduler durumunun zengin checkpoint ile saklanmasi."""
        model = DentalPointNetSeg(in_channels=6, num_classes=33)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        ckpt_path = str(tmp_path / "test_checkpoint.pth")
        ckpt_data = {
            "epoch": 7,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_miou": 0.825,
            "best_val_acc": 0.910,
            "num_classes": 33,
            "num_points": 2048
        }
        torch.save(ckpt_data, ckpt_path)

        # Yeni model ile yukle
        restored_model = DentalPointNetSeg(in_channels=6, num_classes=33)
        loaded_ckpt = torch.load(ckpt_path, map_location="cpu")

        assert "model_state_dict" in loaded_ckpt
        assert loaded_ckpt["epoch"] == 7
        assert loaded_ckpt["best_val_miou"] == 0.825

        restored_model.load_state_dict(loaded_ckpt["model_state_dict"])

        # Parametrelerin esit oldugunu dogrula
        for p1, p2 in zip(model.parameters(), restored_model.parameters()):
            assert torch.equal(p1, p2)

import os
import json
import numpy as np
import pytest
import torch
import open3d as o3d

from ai_model.dataset import (hasta_bazli_bol, hasta_kimligi, sinif_agirliklari_hesapla, bolumu_kaydet,
                              bolumu_yukle, DentalMeshDataset, ornekleri_tara)
from ai_model.network import DentalDGCNNSeg, DentalPointNetSeg, model_olustur, checkpoint_yukle
from ai_model.inference import koseleri_tahmin_et
from ai_model.evaluate import karisiklik_matrisi, matristen_metrikler


def _sahte_vaka(kok, parca, cene, hasta, etiketler=(0, 11, 21)):
    d = os.path.join(kok, parca, cene, hasta)
    os.makedirs(d, exist_ok=True)
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=10.0, resolution=12)
    obj = os.path.join(d, f"{hasta}_{cene}.obj")
    o3d.io.write_triangle_mesh(obj, mesh)
    n = len(mesh.vertices)
    lab = [etiketler[i % len(etiketler)] for i in range(n)]
    with open(os.path.join(d, f"{hasta}_{cene}.json"), "w") as f:
        json.dump({"id_patient": hasta, "jaw": cene, "labels": lab}, f)
    return obj


@pytest.fixture
def sahte_veri(tmp_path):
    kok = str(tmp_path / "teeth")
    for i in range(10):
        _sahte_vaka(kok, "data_part_1", "upper", f"H{i:02d}")
        _sahte_vaka(kok, "data_part_2", "lower", f"H{i:02d}", etiketler=(0, 31, 41))
    return kok


class TestHastaBazliBolme:
    def test_hasta_sizintisi_yok(self, sahte_veri):
        ornekler = ornekleri_tara(sahte_veri, ["data_part_1", "data_part_2"])
        assert len(ornekler) == 20
        bolum = hasta_bazli_bol(ornekler, 0.2, 0.2, tohum=1)
        setler = {k: {hasta_kimligi(o) for o, _ in v} for k, v in bolum.items()}
        assert not (setler["train"] & setler["val"])
        assert not (setler["train"] & setler["test"])
        assert not (setler["val"] & setler["test"])
        assert sum(len(v) for v in bolum.values()) == 20
        # Alt ve ust cene ayni sette
        for v in bolum.values():
            for h in {hasta_kimligi(o) for o, _ in v}:
                assert sum(hasta_kimligi(o) == h for o, _ in v) == 2

    def test_tekrar_uretilebilir(self, sahte_veri):
        ornekler = ornekleri_tara(sahte_veri, ["data_part_1", "data_part_2"])
        assert hasta_bazli_bol(ornekler, tohum=7) == hasta_bazli_bol(ornekler, tohum=7)

    def test_kaydet_yukle(self, sahte_veri, tmp_path):
        ornekler = ornekleri_tara(sahte_veri, ["data_part_1", "data_part_2"])
        bolum = hasta_bazli_bol(ornekler, 0.2, 0.2)
        yol = str(tmp_path / "split.json")
        bolumu_kaydet(bolum, yol, sahte_veri)
        geri = bolumu_yukle(yol, sahte_veri)
        for k in ("train", "val", "test"):
            assert [os.path.normpath(o) for o, _ in geri[k]] == [os.path.normpath(o) for o, _ in bolum[k]]

    def test_gecersiz_oran(self):
        with pytest.raises(ValueError):
            hasta_bazli_bol([], 0.6, 0.6)


class TestVeriSeti:
    def test_sinif_agirliklari(self, sahte_veri):
        ornekler = ornekleri_tara(sahte_veri, ["data_part_1"])
        w = sinif_agirliklari_hesapla(ornekler, 33)
        assert w.shape == (33,)
        assert w[0] > 0 and w[1] > 0       # 0 ve 11 mevcut
        assert w[2] == 0                   # 12 hic yok

    def test_onbellek_ve_sabit_ornekleme(self, sahte_veri, tmp_path):
        ornekler = ornekleri_tara(sahte_veri, ["data_part_1"])
        ds = DentalMeshDataset(ornek_listesi=ornekler, nokta_sayisi=64, sabit_ornekleme=True,
                               onbellek_dizini=str(tmp_path / "cache"))
        a, b = ds[0], ds[0]   # ikincisi onbellekten
        assert len(os.listdir(tmp_path / "cache")) == 1
        assert a["points"].shape == (6, 64)
        assert torch.allclose(a["points"], b["points"], atol=1e-3)
        assert torch.equal(a["labels"], b["labels"])
        assert set(a["labels"].tolist()) <= {0, 1, 9}


class TestModeller:
    def test_dgcnn_forward(self):
        m = DentalDGCNNSeg(in_channels=6, num_classes=33, k=8).eval()
        with torch.no_grad():
            y = m(torch.randn(2, 6, 256))
        assert y.shape == (2, 33, 256)

    def test_dgcnn_gecersiz(self):
        with pytest.raises(ValueError):
            DentalDGCNNSeg(k=0)
        with pytest.raises(ValueError):
            DentalDGCNNSeg(in_channels=6)(torch.randn(1, 3, 32))

    def test_model_olustur(self):
        assert isinstance(model_olustur("dgcnn"), DentalDGCNNSeg)
        assert isinstance(model_olustur("pointnet"), DentalPointNetSeg)
        with pytest.raises(ValueError):
            model_olustur("resnet")

    def test_checkpoint_yukle_eski_ve_yeni(self, tmp_path):
        eski = str(tmp_path / "eski.pth")
        torch.save(DentalPointNetSeg().state_dict(), eski)
        m, k = checkpoint_yukle(eski, varsayilan_model="dgcnn")
        assert isinstance(m, DentalPointNetSeg) and k["model_name"] == "pointnet"

        yeni = str(tmp_path / "yeni.pth")
        torch.save({"model_name": "dgcnn", "num_points": 512,
                    "model_state_dict": DentalDGCNNSeg().state_dict()}, yeni)
        m, k = checkpoint_yukle(yeni)
        assert isinstance(m, DentalDGCNNSeg) and k["num_points"] == 512


class TestCikarimVeMetrik:
    def test_tum_koseler_tahmin_edilir(self):
        model = DentalPointNetSeg().eval()
        v = np.random.rand(1000, 3).astype(np.float32)
        n = np.random.rand(1000, 3).astype(np.float32)
        tahmin, olas = koseleri_tahmin_et(model, v, n, nokta_sayisi=256, tur_sayisi=2)
        assert tahmin.shape == (1000,)
        assert olas.shape == (1000, 33)
        np.testing.assert_allclose(olas.sum(axis=1), 1.0, atol=1e-4)

    def test_matris_metrikleri(self):
        hedef = np.array([0, 0, 1, 1, 2, 2])
        tahmin = np.array([0, 0, 1, 2, 2, 2])
        m = matristen_metrikler(karisiklik_matrisi(tahmin, hedef, 3))
        assert m["accuracy"] == pytest.approx(5 / 6)
        assert m["iou_per_class"][0] == pytest.approx(1.0)
        assert m["iou_per_class"][1] == pytest.approx(0.5)
        assert m["iou_per_class"][2] == pytest.approx(2 / 3)
        assert m["mIoU_teeth"] == pytest.approx((0.5 + 2 / 3) / 2)


class TestKoseSirasi:
    def test_obj_kose_sirasi_korunur(self, tmp_path):
        """Teeth3DS etiketleri OBJ 'v' sirasina gore verilir; okuyucu sirayi bozmamali.
        (open3d.io.read_triangle_mesh sirayi degistirdigi icin etiketler kayiyordu.)"""
        from ai_model.dataset import obj_oku, mesh_oku
        rng = np.random.RandomState(0)
        v = rng.rand(300, 3).astype(np.float32) * 50
        f = rng.randint(0, 300, size=(500, 3))
        yol = tmp_path / "m.obj"
        with open(yol, "w") as fh:
            for p in v:
                fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0.5 0.5 0.5\n")
            for t in f:
                fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
        okunan, yuzler = obj_oku(str(yol))
        np.testing.assert_allclose(okunan, v, atol=1e-4)
        np.testing.assert_array_equal(yuzler, f)
        k, n, _, _ = mesh_oku(str(yol))
        np.testing.assert_allclose(k, v, atol=1e-4)
        assert n.shape == v.shape

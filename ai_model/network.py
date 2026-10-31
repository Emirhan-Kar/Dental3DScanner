import torch
import torch.nn as nn
import torch.nn.functional as F


class DentalPointNetSeg(nn.Module):
    def __init__(self, giris_kanallari=6, sinif_sayisi=33, **ek_parametreler):
        """
        3D Nokta Bulutu Dis Segmentasyon Agi.

        Args:
            giris_kanallari (int): Girdi kanal sayisi (varsayilan 6: X, Y, Z + Normaller Nx, Ny, Nz)
            sinif_sayisi (int): Sinif sayisi (varsayilan 33: 32 Dis + 1 Dis Eti)

        Raises:
            TypeError: Argumanlar tamsayi degilse
            ValueError: giris_kanallari veya sinif_sayisi pozitif degilse
        """
        super().__init__()

        giris_kanallari = ek_parametreler.get("in_channels", giris_kanallari)
        sinif_sayisi = ek_parametreler.get("num_classes", sinif_sayisi)

        if not isinstance(giris_kanallari, int) or giris_kanallari <= 0:
            raise ValueError(f"giris_kanallari pozitif bir tamsayi olmalidir, alinan: {giris_kanallari}")
        if not isinstance(sinif_sayisi, int) or sinif_sayisi <= 0:
            raise ValueError(f"sinif_sayisi pozitif bir tamsayi olmalidir, alinan: {sinif_sayisi}")

        self.giris_kanallari = giris_kanallari
        self.sinif_sayisi = sinif_sayisi
        self.in_channels = giris_kanallari
        self.num_classes = sinif_sayisi

        # NOT: Katman isimleri (conv1, bn1 ...) kayitli model agirliklariyla (.pth)
        # uyumluluk icin degistirilmemistir.

        # 1. Asama: Noktasal Oznitelik Cikarimi (Yerel Oznitelikler)
        self.conv1 = nn.Conv1d(giris_kanallari, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

        # 2. Asama: Global Cene Geometrisi Temsili
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        # 3. Asama: Segmentasyon Kafasi (Global + Yerel Oznitelik Birlesimi)
        # 1024 (global) + 128 (yerel 3. katman) = 1152
        self.conv6 = nn.Conv1d(1024 + 128, 512, 1)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, sinif_sayisi, 1)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, girdi):
        """
        Ileri gecis fonksiyonu.

        Args:
            girdi (torch.Tensor): Girdi tensoru, boyut: (B, C, N) -> (Batch, giris_kanallari, Nokta Sayisi)

        Returns:
            torch.Tensor: Ham sinif skorlari (logits) tensoru, boyut: (B, sinif_sayisi, N)

        Raises:
            TypeError: girdi torch.Tensor degilse
            ValueError: girdi'nin boyutlari veya kanal sayisi uyumsuz ise
        """
        if not torch.is_tensor(girdi):
            raise TypeError(f"Girdi bir torch.Tensor olmalidir, alinan: {type(girdi).__name__}")

        if girdi.dim() != 3:
            raise ValueError(f"Girdi tensoru 3 boyutlu olmalidir (B, C, N), alinan: {girdi.dim()}D (sekil: {tuple(girdi.shape)})")

        B, C, N = girdi.size()

        if C != self.giris_kanallari:
            raise ValueError(f"Girdi kanal sayisi ({C}), modelin bekledigi ({self.giris_kanallari}) ile uyusmuyor.")

        if N == 0:
            raise ValueError("Girdi tensorundeki nokta sayisi (N) 0 olamaz.")

        # Yerel oznitelikler
        ara1 = F.relu(self.bn1(self.conv1(girdi)))
        ara2 = F.relu(self.bn2(self.conv2(ara1)))
        nokta_oznitelik = F.relu(self.bn3(self.conv3(ara2)))  # (B, 128, N)

        # Global oznitelikler
        ara3 = F.relu(self.bn4(self.conv4(nokta_oznitelik)))
        ara4 = self.bn5(self.conv5(ara3))                      # (B, 1024, N)
        global_oznitelik = torch.max(ara4, 2, keepdim=True)[0]  # (B, 1024, 1)

        # Global oznitelik tum noktalara yayilir ve birlestirilir
        genisletilmis_global = global_oznitelik.repeat(1, 1, N)  # (B, 1024, N)
        birlesik_oznitelik = torch.cat([nokta_oznitelik, genisletilmis_global], dim=1)  # (B, 1152, N)

        # Noktasal siniflandirma
        cikti = F.relu(self.bn6(self.conv6(birlesik_oznitelik)))
        cikti = F.relu(self.bn7(self.conv7(cikti)))
        cikti = self.dropout(cikti)
        cikti = F.relu(self.bn8(self.conv8(cikti)))
        skorlar = self.conv9(cikti)  # (B, sinif_sayisi, N)

        return skorlar

# ─────────────────────────────────────────────────────────────────────
# DGCNN (Dynamic Graph CNN) - Wang ve ark., 2019
# PointNet her noktayi tek basina isler; DGCNN ise her noktanin k en yakin
# komsusuyla kurdugu "kenar" (edge) ozniteliklerini ogrenir. Bu yerel geometri
# bilgisi, komsu disler arasindaki sinirlarin dogru ayrilmasi icin kritiktir.
# ─────────────────────────────────────────────────────────────────────

def _knn(x, k):
    """x: (B, C, N) -> en yakin k komsunun indeksleri (B, N, k). Hassasiyet icin float32."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        ic = -2 * torch.matmul(x.transpose(2, 1), x)
        kare = torch.sum(x ** 2, dim=1, keepdim=True)
        negatif_uzaklik = -kare - ic - kare.transpose(2, 1)
        return negatif_uzaklik.topk(k=k, dim=-1)[1]


def _kenar_oznitelikleri(x, k, idx=None):
    """EdgeConv girdisi: [x_j - x_i, x_i] -> (B, 2C, N, k)."""
    B, C, N = x.size()
    if idx is None:
        idx = _knn(x, k)
    taban = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + taban).view(-1)
    x_t = x.transpose(2, 1).contiguous()                    # (B, N, C)
    komsu = x_t.view(B * N, -1)[idx, :].view(B, N, k, C)
    merkez = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)
    return torch.cat((komsu - merkez, merkez), dim=3).permute(0, 3, 1, 2).contiguous()


class DentalDGCNNSeg(nn.Module):
    def __init__(self, giris_kanallari=6, sinif_sayisi=33, k=20, **ek_parametreler):
        """
        DGCNN tabanli 3D dis segmentasyon agi.

        Args:
            giris_kanallari (int): 6 (X, Y, Z, Nx, Ny, Nz)
            sinif_sayisi (int): 33 (32 dis + dis eti)
            k (int): Komsu sayisi (varsayilan 20)
        """
        super().__init__()
        giris_kanallari = ek_parametreler.get("in_channels", giris_kanallari)
        sinif_sayisi = ek_parametreler.get("num_classes", sinif_sayisi)
        if not isinstance(giris_kanallari, int) or giris_kanallari <= 0:
            raise ValueError(f"giris_kanallari pozitif bir tamsayi olmalidir, alinan: {giris_kanallari}")
        if not isinstance(sinif_sayisi, int) or sinif_sayisi <= 0:
            raise ValueError(f"sinif_sayisi pozitif bir tamsayi olmalidir, alinan: {sinif_sayisi}")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k pozitif bir tamsayi olmalidir, alinan: {k}")

        self.giris_kanallari = self.in_channels = giris_kanallari
        self.sinif_sayisi = self.num_classes = sinif_sayisi
        self.k = k

        def edge(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))

        def nokta(cin, cout):
            return nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout), nn.LeakyReLU(0.2))

        self.edge1 = nn.Sequential(edge(giris_kanallari * 2, 64), edge(64, 64))
        self.edge2 = nn.Sequential(edge(64 * 2, 64), edge(64, 64))
        self.edge3 = edge(64 * 2, 64)
        self.global_katman = nokta(192, 1024)
        self.kafa = nn.Sequential(
            nokta(1024 + 192, 512),
            nokta(512, 256),
            nn.Dropout(p=0.5),
            nokta(256, 128),
            nn.Conv1d(128, sinif_sayisi, 1),
        )

    def forward(self, girdi):
        if not torch.is_tensor(girdi):
            raise TypeError(f"Girdi bir torch.Tensor olmalidir, alinan: {type(girdi).__name__}")
        if girdi.dim() != 3:
            raise ValueError(f"Girdi tensoru 3 boyutlu olmalidir (B, C, N), alinan: {girdi.dim()}D")
        B, C, N = girdi.size()
        if C != self.giris_kanallari:
            raise ValueError(f"Girdi kanal sayisi ({C}), modelin bekledigi ({self.giris_kanallari}) ile uyusmuyor.")
        if N == 0:
            raise ValueError("Girdi tensorundeki nokta sayisi (N) 0 olamaz.")
        k = min(self.k, N)

        # Ilk graf sadece XYZ uzerinden kurulur (normaller komsuluk icin yaniltici olabilir)
        idx = _knn(girdi[:, :3, :], k)
        x1 = self.edge1(_kenar_oznitelikleri(girdi, k, idx)).max(dim=-1)[0]
        x2 = self.edge2(_kenar_oznitelikleri(x1, k)).max(dim=-1)[0]
        x3 = self.edge3(_kenar_oznitelikleri(x2, k)).max(dim=-1)[0]
        yerel = torch.cat((x1, x2, x3), dim=1)                      # (B, 192, N)
        global_oz = self.global_katman(yerel).max(dim=-1, keepdim=True)[0].expand(-1, -1, N)
        return self.kafa(torch.cat((global_oz, yerel), dim=1))      # (B, sinif, N)


MODELLER = {
    "pointnet": DentalPointNetSeg,
    "dgcnn": DentalDGCNNSeg,
}


def model_olustur(ad="pointnet", giris_kanallari=6, sinif_sayisi=33, **ek):
    """Ada gore segmentasyon modeli olusturur ('pointnet' veya 'dgcnn')."""
    ad = (ad or "pointnet").lower()
    if ad not in MODELLER:
        raise ValueError(f"Bilinmeyen model: '{ad}'. Secenekler: {list(MODELLER)}")
    if ad == "dgcnn":
        return DentalDGCNNSeg(giris_kanallari=giris_kanallari, sinif_sayisi=sinif_sayisi, k=ek.get("k", 20))
    return DentalPointNetSeg(giris_kanallari=giris_kanallari, sinif_sayisi=sinif_sayisi)


def checkpoint_yukle(yol, cihaz="cpu", varsayilan_model="pointnet", giris_kanallari=6, sinif_sayisi=33):
    """Hem yeni (sozluk + 'model_name') hem eski (salt state_dict) checkpoint'leri yukler.

    Returns:
        (model, checkpoint_sozlugu)
    """
    kayit = torch.load(yol, map_location=cihaz, weights_only=False)
    if isinstance(kayit, dict) and "model_state_dict" in kayit:
        ad = kayit.get("model_name", varsayilan_model)
        model = model_olustur(ad, kayit.get("in_channels", giris_kanallari),
                              kayit.get("num_classes", sinif_sayisi), k=kayit.get("k", 20))
        model.load_state_dict(kayit["model_state_dict"])
    else:
        # Salt state_dict: mimariyi katman isimlerinden anla
        ad = "dgcnn" if any(k.startswith("edge1") for k in kayit) else "pointnet"
        model = model_olustur(ad, giris_kanallari, sinif_sayisi)
        model.load_state_dict(kayit)
        kayit = {"model_name": ad}
    return model.to(cihaz), kayit

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
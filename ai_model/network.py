import torch
import torch.nn as nn
import torch.nn.functional as F


class DentalPointNetSeg(nn.Module):
    def __init__(self, in_channels=6, num_classes=33):
        """
        3D Nokta Bulutu Diş Segmentasyon Ağı.

        Args:
            in_channels (int): Girdi kanal sayısı (varsayılan 6: X, Y, Z + Normaller Nx, Ny, Nz)
            num_classes (int): Sınıf sayısı (varsayılan 33: 32 Diş + 1 Diş Eti)

        Raises:
            TypeError: Argümanlar tamsayı değilse
            ValueError: in_channels veya num_classes pozitif değilse
        """
        super().__init__()

        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(f"in_channels pozitif bir tamsayı olmalıdır, alınan: {in_channels}")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes pozitif bir tamsayı olmalıdır, alınan: {num_classes}")

        self.in_channels = in_channels
        self.num_classes = num_classes

        # 1. Aşama: Noktasal Öznitelik Çıkarımı (Yerel Öznitelikler)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

        # 2. Aşama: Global Çene Geometrisi Temsili
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        # 3. Aşama: Segmentasyon Kafası (Global + Yerel Öznitelik Birleşimi)
        # 1024 (global) + 128 (yerel 3. katman) = 1152
        self.conv6 = nn.Conv1d(1024 + 128, 512, 1)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, num_classes, 1)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        İleri geçiş fonksiyonu.

        Args:
            x (torch.Tensor): Girdi tensörü, boyut: (B, C, N) -> (Batch, in_channels, Nokta Sayısı)

        Returns:
            torch.Tensor: Logits tensörü, boyut: (B, num_classes, N)

        Raises:
            TypeError: x torch.Tensor değilse
            ValueError: x'in boyutları veya kanal sayısı uyumsuz ise
        """
        if not torch.is_tensor(x):
            raise TypeError(f"Girdi bir torch.Tensor olmalıdır, alınan: {type(x).__name__}")

        if x.dim() != 3:
            raise ValueError(f"Girdi tensörü 3 boyutlu olmalıdır (B, C, N), alınan: {x.dim()}D (şekil: {tuple(x.shape)})")

        B, C, N = x.size()

        if C != self.in_channels:
            raise ValueError(f"Girdi kanal sayısı ({C}), modelin beklediği ({self.in_channels}) ile uyuşmuyor.")

        if N == 0:
            raise ValueError("Girdi tensöründeki nokta sayısı (N) 0 olamaz.")

        # Yerel öznitelikler
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        point_feat = F.relu(self.bn3(self.conv3(x2)))  # (B, 128, N)

        # Global öznitelikler
        x3 = F.relu(self.bn4(self.conv4(point_feat)))
        x4 = self.bn5(self.conv5(x3))                   # (B, 1024, N)
        global_feat = torch.max(x4, 2, keepdim=True)[0] # (B, 1024, 1)

        # Global özniteliği tüm noktalara yay ve birleştir
        global_feat_expanded = global_feat.repeat(1, 1, N)  # (B, 1024, N)
        concat_feat = torch.cat([point_feat, global_feat_expanded], dim=1) # (B, 1152, N)

        # Noktasal sınıflandırma
        out = F.relu(self.bn6(self.conv6(concat_feat)))
        out = F.relu(self.bn7(self.conv7(out)))
        out = self.dropout(out)
        out = F.relu(self.bn8(self.conv8(out)))
        logits = self.conv9(out)  # (B, num_classes, N)

        return logits
# models/correction/unet.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise Linear Modulation
    cond_dim -> (gamma, beta) for a given channel size (C)
    """
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(z).chunk(2, dim=1)  # [B, C], [B, C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta  = beta .unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class DoubleConv(nn.Module):
    """Conv -> BN -> ReLU -> FiLM -> Conv -> BN -> ReLU -> FiLM"""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.film1 = FiLM(cond_dim, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.film2 = FiLM(cond_dim, out_ch)

        # He init
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.film1(x, z)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.film2(x, z)
        return x


class Down(nn.Module):
    """Downscale then DoubleConv"""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = DoubleConv(in_ch, out_ch, cond_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.block(x, z)


class Up(nn.Module):
    """Upscale then concat skip, then DoubleConv"""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch, cond_dim)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, cond_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x, z)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """UNet with FiLM conditioning at every block.
    Args:
        n_channels: 入力画像チャネル数（例: 3）
        n_classes: 出力チャネル数（例: 3）
        cond_dim:   ゼルニケ係数など条件ベクトルの次元
        base_ch:    ベースチャネル幅
        bilinear:   Upサンプリング方式
    """
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 3,
        cond_dim: int = 20,
        base_ch: int = 32,
        bilinear: bool = True
    ):
        super().__init__()
        self.inc   = DoubleConv(n_channels, base_ch, cond_dim)
        self.down1 = Down(base_ch, base_ch*2, cond_dim)
        self.down2 = Down(base_ch*2, base_ch*4, cond_dim)
        self.up1   = Up(base_ch*4 + base_ch*2, base_ch*2, cond_dim, bilinear)
        self.up2   = Up(base_ch*2 + base_ch,   base_ch,   cond_dim, bilinear)
        self.outc  = OutConv(base_ch, n_classes)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z: [B, cond_dim]
        x1 = self.inc(x, z)
        x2 = self.down1(x1, z)
        x3 = self.down2(x2, z)

        x  = self.up1(x3, x2, z)
        x  = self.up2(x,  x1, z)
        logits = self.outc(x)
        return logits

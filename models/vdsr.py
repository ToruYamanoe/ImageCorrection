import torch
from torch import nn


class VDSR(nn.Module):
    def __init__(self, img_shape=(3, 500, 500), init_weights=True):
        super().__init__()
        self.img_shape = img_shape

        self.conv_1 = nn.Conv2d(self.img_shape[0], 64, 3, padding='same', padding_mode='replicate', bias=False)

        conv_block = []
        for i in range(18):
            conv_block.append(nn.Conv2d(64, 64, 3, padding='same', padding_mode='replicate', bias=False))
            conv_block.append(nn.ReLU(True))
        self.conv_block = nn.Sequential(*conv_block)
        self.conv_2 = nn.Conv2d(64, self.img_shape[0], 3, padding='same', padding_mode='replicate', bias=False)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        residual = x
        out = self.conv_1(x)
        out = self.conv_block(out)
        out = self.conv_2(out)
        return out + residual

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
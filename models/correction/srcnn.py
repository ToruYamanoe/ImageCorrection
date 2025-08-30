import torch
from torch import nn


class SRCNN(nn.Module):
    def __init__(self, img_shape=(3, 500, 500)):
        super().__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Conv2d(self.img_shape[0], 64, 9, padding='same', padding_mode='replicate'),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 5, padding='same', padding_mode='replicate'),
            nn.ReLU(True),
            nn.Conv2d(32, self.img_shape[0], 1, padding='same', padding_mode='replicate')
        )

    def forward(self, x):
        x = self.model(x)
        return x
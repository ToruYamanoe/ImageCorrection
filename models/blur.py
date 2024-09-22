import torch
from torch import nn
import numpy as np


class BlurModel(nn.Module):
    def __init__(self, img_shape=[3, 500, 500], S: float = 1.0, C: float = 0.0, A: int = 0, R: float = 1.5, sp=49):
        super().__init__()
        self.img_shape = img_shape
        self.sphere = S
        self.cylinder = C
        self.axis = A
        self.R = R
        self.sp = sp
        self.psf = self._calculate_psf(*self._convert_diopter_to_Zernike())

        # self.conv = nn.Conv2d(self.img_shape[0], self.img_shape[0], self.psf.shape[1:], padding='same', padding_mode='replicate', groups=self.img_shape[0])
        # self.conv.weight = torch.nn.parameter.Parameter(self.psf.float())
        self.conv = nn.Conv2d(self.img_shape[0], self.img_shape[0], self.psf.shape[2:], bias=False, padding='same', padding_mode='replicate', groups=self.img_shape[0])
        self.conv.weight = torch.nn.parameter.Parameter(torch.Tensor(self.psf).float())
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.conv(x)
        return x

    def _convert_diopter_to_Zernike(self):
        A = self.axis * np.pi / 180

        Z4 = self.R ** 2 * (self.sphere + self.cylinder / 2) / (4 * np.sqrt(3))
        Z5 = self.R ** 2 * self.cylinder * np.sin(2 * A) / (4 * np.sqrt(6))
        Z6 = self.R ** 2 * self.cylinder * np.cos(2 * A) / (4 * np.sqrt(6))
        return Z4, Z5, Z6

    def _zernikecartesian(self, coefficient, x, y):
        Z = [0] + coefficient
        r = np.sqrt(x**2 + y**2)
        Z1 = Z[1] * 1
        Z2 = Z[2] * 2 * x
        Z3 = Z[3] * 2 * y
        Z4 = Z[4] * np.sqrt(3) * (2 * r**2 - 1)
        Z5 = Z[5] * 2 * np.sqrt(6) * x * y
        Z6 = Z[6] * np.sqrt(6) * (x**2 - y**2)
        Z7 = Z[7] * np.sqrt(8) * y * (3 * r**2 - 2)
        Z8 = Z[8] * np.sqrt(8) * x * (3 * r**2 - 2)
        Z9 = Z[9] * np.sqrt(8) * y * (3 * x**2 - y**2)
        Z10 = Z[10] * np.sqrt(8) * x * (x**2 - 3 * y**2)
        Z11 = Z[11] * np.sqrt(5) * (6 * r**4 - 6 * r**2 + 1)
        Z12 = Z[12] * np.sqrt(10) * (x**2 - y**2) * (4 * r**2 - 3)
        Z13 = Z[13] * 2 * np.sqrt(10) * x * y * (4 * r**2 - 3)
        Z14 = Z[14] * np.sqrt(10) * (r**4 - 8 * x**2 * y**2)
        Z15 = Z[15] * 4 * np.sqrt(10) * x * y * (x**2 - y**2)
        Z16 = Z[16] * np.sqrt(12) * x * (10 * r**4 - 12 * r**2 + 3)
        Z17 = Z[17] * np.sqrt(12) * y * (10 * r**4 - 12 * r**2 + 3)
        Z18 = Z[18] * np.sqrt(12) * x * (x**2 - 3 * y**2) * (5 * r**2 - 4)
        Z19 = Z[19] * np.sqrt(12) * y * (3 * x**2 - y**2) * (5 * r**2 - 4)
        Z20 = Z[20] * np.sqrt(12) * x * (16 * x**4 - 20 * x**2 * r**2 + 5 * r**4)
        Z21 = Z[21] * np.sqrt(12) * y * (16 * y**4 - 20 * y**2 * r**2 + 5 * r**4)
        Z22 = Z[22] * np.sqrt(7) * (20 * r**6 - 30 * r**4 + 12 * r**2 - 1)
        Z23 = Z[23] * 2 * np.sqrt(14) * x * y * (15 * r**4 - 20 * r**2 + 6)
        Z24 = Z[24] * np.sqrt(14) * (x**2 - y**2) * (15 * r**4 - 20 * r**2 + 6)
        Z25 = Z[25] * 4 * np.sqrt(14) * x * y * (x**2 - y**2) * (6 * r**2 - 5)
        Z26 = Z[26] * np.sqrt(14) * (8 * x**4 - 8 * x**2 * r**2 + r**4) * (6 * r**2 - 5)
        Z27 = Z[27] * np.sqrt(14) * x * y * (32 * x**4 - 32 * x**2 * r**2 + 6 * r**4)
        Z28 = Z[28] * np.sqrt(14) * (32 * x**6 - 48 * x**4 * r**2 + 18 * x**2 * r**4 - r**6)
        Z29 = Z[29] * 4 * y * (35 * r**6 - 60 * r**4 + 30 * r**2 + 10)
        Z30 = Z[30] * 4 * x * (35 * r**6 - 60 * r**4 + 30 * r**2 + 10)
        Z31 = Z[31] * 4 * y * (3 * x**2 - y**2) * (21 * r**4 - 30 * r**2 + 10)
        Z32 = Z[32] * 4 * x * (x**2 - 3 * y**2) * (21 * r**4 - 30 * r**2 + 10)
        Z33 = Z[33] * 4 * (7 * r**2 - 6) * (4 * x**2 * y * (x**2 - y**2) + y * (r**4 - 8 * x**2 * y**2))
        Z34 = Z[34] * (4 * (7 * r**2 - 6) * (x * (r**4 - 8 * x**2 * y**2) - 4 * x * y**2 * (x**2 - y**2)))
        Z35 = Z[35] * (8 * x**2 * y * (3 * r**4 - 16 * x**2 * y**2) + 4 * y * (x**2 - y**2) * (r**4 - 16 * x**2 * y**2))
        Z36 = Z[36] * (4 * x * (x**2 - y**2) * (r**4 - 16 * x**2 * y**2) - 8 * x * y**2 * (3 * r**4 - 16 * x**2 * y**2))
        Z37 = Z[37] * 3 * (70 * r**8 - 140 * r**6 + 90 * r**4 - 20 * r**2 + 1)
        ZW = Z1 + Z2 + Z3 + Z4 + Z5 + Z6 + Z7 + Z8 + Z9 + Z10 + Z11 + Z12 + Z13 + Z14 + Z15 + Z16 + Z17 + Z18 + Z19 + \
            Z20 + Z21 + Z22 + Z23 + Z24 + Z25 + Z26 + Z27 + Z28 + Z29 + Z30 + Z31 + Z32 + Z33 + Z34 + Z35 + Z36 + Z37
        return ZW

    def _calculate_psf(self, Z4, Z5, Z6):
        Z1 = Z2 = Z3 = Z7 = Z8 = Z9 = Z10 = Z11 = Z12 = Z13 = Z14 = Z15 = Z16 = Z17 = Z18 = Z19 = Z20 = Z21 = Z22 = Z23 = Z24 = Z25 = Z26 = Z27 = Z28 = Z29 = Z30 = Z31 = Z32 = Z33 = Z34 = Z35 = Z36 = Z37 = 0
        coefficients = [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11, Z12, Z13, Z14, Z15, Z16, Z17, Z18,
                        Z19, Z20, Z21, Z22, Z23, Z24, Z25, Z26, Z27, Z28, Z29, Z30, Z31, Z32, Z33, Z34, Z35, Z36, Z37]

        pupil = l1 = self.sp  # exit pupil sample points
        r = self.R
        x = np.linspace(-r, r, l1)
        [X, Y] = np.meshgrid(x, x)
        Z = self._zernikecartesian(coefficients, X, Y)
        for i in range(len(Z)):
            for j in range(len(Z)):
                if x[i]**2 + x[j]**2 > r**2:
                    Z[i][j] = 0

        d = pupil + 1  # background
        A = np.zeros([d, d])
        A[int(d / 2 - l1 / 2 + 1):int(d / 2 + l1 / 2 + 1), int(d / 2 - l1 / 2 + 1):int(d / 2 + l1 / 2 + 1)] = Z
        axis_1 = d / pupil * r
        # Aは波面収差を表す（wavefront function）

        # exp(2iπ/λ)
        abbe = np.exp(-1j * 2 * np.pi * A)
        for i in range(len(abbe)):
            for j in range(len(abbe)):
                if abbe[i][j] == 1:
                    abbe[i][j] = 0

        PSF = abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(abbe))))**2
        PSF = PSF / PSF.max()
        PSF = PSF / sum(sum(PSF))

        PSF = np.expand_dims(PSF, axis=0)
        PSF = np.expand_dims(PSF, axis=0)
        if self.img_shape[0] == 3:
            PSF = np.repeat(PSF, 3, axis=0)

        # print(PSF.shape)
        return PSF

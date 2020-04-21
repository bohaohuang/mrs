"""
OCR module from paper: https://arxiv.org/abs/1909.11065
This code comes from: https://github.com/rosinality/ocr-pytorch
"""

# Built-in

# Libs
import torch

# Pytorch
from torch import nn
from torch.nn import functional as F

# Own modules


def conv2d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv2d(
            in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)


def conv1d(in_channel, out_channel):
    layers = [
        nn.Conv1d(in_channel, out_channel, 1, bias=False),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)


class OCRModule(nn.Module):
    def __init__(self, n_class, region_chan, pixel_chan, chan_out_num=512):
        super(OCRModule, self).__init__()
        self.L = nn.Conv2d(region_chan, n_class, 1)
        self.X = conv2d(pixel_chan, chan_out_num, 3)

        self.phi = conv1d(chan_out_num, chan_out_num // 2)
        self.psi = conv1d(chan_out_num, chan_out_num // 2)
        self.delta = conv1d(chan_out_num, chan_out_num // 2)
        self.rho = conv1d(chan_out_num // 2, chan_out_num)
        self.g = conv2d(chan_out_num + chan_out_num, chan_out_num, 1)

    def forward(self, region, pixel):
        pixel_size = pixel.shape[2:]
        X = self.X(pixel)
        L = self.L(region)
        L = F.interpolate(L, size=pixel_size, mode='bilinear', align_corners=False)

        batch, n_class, height, width = L.shape
        l_flat = L.view(batch, n_class, -1)

        # M: NKL
        M = torch.softmax(l_flat, -1)
        channel = X.shape[1]
        X_flat = X.view(batch, channel, -1)
        # f_k: NCK
        f_k = (M @ X_flat.transpose(1, 2)).transpose(1, 2)

        # query: NKD
        query = self.phi(f_k).transpose(1, 2)
        # key: NDL
        key = self.psi(X_flat)
        logit = query @ key
        # attn: NKL
        attn = torch.softmax(logit, 1)

        # delta: NDK
        delta = self.delta(f_k)
        # attn_sum: NDL
        attn_sum = delta @ attn
        # x_obj = NCHW
        X_obj = self.rho(attn_sum).view(batch, -1, height, width)

        concat = torch.cat([X, X_obj], 1)
        X_bar = self.g(concat)
        return L, X_bar  # region pred, pixel pred

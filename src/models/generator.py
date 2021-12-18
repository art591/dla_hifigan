import torch
from torch import nn


class ResBlockConv(nn.Module):
    def __init__(self, channels, k_n, D_r_n_m):
        super().__init__()
        self.conv = nn.Sequential(*[nn.Sequential(
                                            nn.LeakyReLU(0.1),
                                            nn.Conv1d(channels,
                                                      channels,
                                                      kernel_size=k_n,
                                                      dilation=dilation,
                                                      padding='same')
                                        )
                                   for dilation in D_r_n_m])

    def forward(self, x):
        return self.conv(x)


class MRF(nn.Module):
    def __init__(self, channels, k_r, D_r):
        super().__init__()
        self.resblocks = nn.ModuleList([ResBlockConv(channels, k_r[i], D_r[i]) for i in range(len(D_r))])

    def forward(self, x):
        for i in range(len(self.resblocks)):
            x = x + self.resblocks[i](x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, channels, k_u_l, k_r, D_r):
        super().__init__()
        self.block = nn.Sequential(nn.LeakyReLU(0.1),
                                   nn.ConvTranspose1d(channels,
                                                      channels // 2,
                                                      kernel_size=k_u_l,
                                                      stride=k_u_l // 2,
                                                      padding=k_u_l // 4),
                                   MRF(channels // 2, k_r, D_r))

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels, h_u, k_u, k_r, D_r):
        super().__init__()
        self.start_conv = nn.Conv1d(in_channels, h_u, kernel_size=7, padding='same')
        self.middle = nn.Sequential(*[MiddleBlock(h_u // (2 ** i), k_u[i], k_r, D_r) for i in range(len(k_u))])
        self.last = nn.Sequential(nn.LeakyReLU(0.1), nn.Conv1d(h_u // (2 ** (len(k_u))),
                                                               1,
                                                               kernel_size=7,
                                                               padding='same'), nn.Tanh())

    def forward(self, x):
        x = self.start_conv(x)
        x = self.middle(x)
        x = self.last(x)
        return x

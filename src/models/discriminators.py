import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn.functional import pad


class PeriodDicriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        cs = [64, 128, 256, 512, 1024]
        self.layers = nn.ModuleList([
            nn.Sequential(spectral_norm(nn.Conv2d(1, cs[0], kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))), nn.LeakyReLU(0.1)),
            nn.Sequential(spectral_norm(nn.Conv2d(cs[0], cs[1], kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))), nn.LeakyReLU(0.1)),
            nn.Sequential(spectral_norm(nn.Conv2d(cs[1], cs[2], kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))), nn.LeakyReLU(0.1)),
            nn.Sequential(spectral_norm(nn.Conv2d(cs[2], cs[3], kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))), nn.LeakyReLU(0.1)),
            nn.Sequential(spectral_norm(nn.Conv2d(cs[3], cs[4], kernel_size=(5, 1), stride=1, padding=(2, 0))), nn.LeakyReLU(0.1)),
            spectral_norm(nn.Conv2d(cs[4], 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))
                                    ])

    def forward(self, x):
        to_pad = (x.shape[1] // self.period) * self.period - x.shape[1]
        bs, seq_len = x.shape[:2]
        x_padded = pad(x, (0, to_pad), "reflect").reshape(-1, seq_len // self.period, self.period).unsqueeze(1)
        features = []
        for i in range(len(self.layers)):
            x_padded = self.layers[i](x_padded)
            if i + 1 != len(self.layers):
                features.append(x_padded)
        return x_padded.flatten(1), features


class MultiPeriodDicriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.periods = periods
        self.ds = nn.ModuleList([PeriodDicriminator(p) for p in periods])

    def forward(self, x):
        features = []
        results = []
        for i in range(len(self.ds)):
            res, fs = self.ds[i](x)
            results.append(res)
            features += fs
        return results, features


class ScaleDicriminator(nn.Module):
    def __init__(self, downsample_rate):
        super().__init__()
        self.downsample = nn.Sequential(*[nn.AvgPool1d(4, 2, padding=2) for i in range(downsample_rate // 2)])
        cs = [64, 128, 256, 512, 1024]
        self.layers = nn.ModuleList([
            nn.Sequential(spectral_norm(nn.Conv1d(1, 128, 15, 1, padding=7)), nn.LeakyReLU(0.1)),
            nn.Sequential(spectral_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)), nn.LeakyReLU(0.1)),
            nn.Sequential(spectral_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)), nn.LeakyReLU(0.1)),
            nn.Sequential(spectral_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)), nn.LeakyReLU(0.1)),
            nn.Sequential(spectral_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), nn.LeakyReLU(0.1)),
            spectral_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2))
                                    ])

    def forward(self, x):
        x_down = self.downsample(x).unsqueeze(1)
        features = []
        for i in range(len(self.layers)):
            x_down = self.layers[i](x_down)
            if i + 1 != len(self.layers):
                features.append(x_down)
        return x_down.flatten(1), features


class MultiScaleDicriminator(nn.Module):
    def __init__(self, downsample_rates=[1, 2, 4]):
        super().__init__()
        self.ds = nn.ModuleList([ScaleDicriminator(dr) for dr in downsample_rates])

    def forward(self, x):
        features = []
        results = []
        for i in range(len(self.ds)):
            res, fs = self.ds[i](x)
            results.append(res)
            features += fs
        return results, features


class Discriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11], downsample_rates=[1, 2, 4]):
        super().__init__()
        self.mpd = MultiPeriodDicriminator(periods)
        self.msd = MultiScaleDicriminator(downsample_rates)

    def forward(self, x):
        results, features = self.mpd(x)
        results_, features_ = self.msd(x)
        results += results_
        features += features_
        return results, features

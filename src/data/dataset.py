import torch
from torch import nn
import torchaudio
import numpy as np


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root):
        super().__init__(root=root)

    def __getitem__(self, index: int):
        waveform, _, _, _ = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()
        return waveform, waveforn_length

import torch
from torch import nn
import torchaudio
import numpy as np


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, segment_size):
        super().__init__(root=root)
        self.segment_size = segment_size

    def __getitem__(self, index: int):
        waveform, _, _, _ = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()
        if waveforn_length <= self.segment_size:
            return waveform, waveforn_length
        start = np.random.randint(0, waveforn_length - self.segment_size - 1)[0]
        waveform = waveform[:, start:start+self.segment_size]
        waveforn_length = waveforn_length - self.segment_size
        return waveform, waveforn_length

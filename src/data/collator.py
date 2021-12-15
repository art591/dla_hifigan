from typing import Tuple, Dict, Optional, List, Union
from itertools import islice
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from src.data.melspectrogram import MelSpectrogram, MelSpectrogramConfig


class LJSpeechCollator:
    def __init__(self, device='cpu'):
        self.device = device
        self.featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
        self.hop_length = MelSpectrogramConfig().hop_length

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveform_length = list(
            zip(*instances)
        )
        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1).to(self.device)

        waveform_length = torch.cat(waveform_length)
        waveform = waveform[:, :-(waveform_length.max() % 2)]
        melspec = self.featurizer(waveform)
        melspec_length = waveform_length // self.hop_length + (waveform_length % self.hop_length != 0).int()
            
        return {"waveform" : waveform,
                "waveform_length" : waveform_length.to(self.device),
                "melspec" : melspec,
                "melspec_length" : melspec_length.to(self.device)}
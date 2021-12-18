import os
import torch
from torch import nn
import torchaudio
from src.data.melspectrogram import MelSpectrogram, MelSpectrogramConfig
from src.models.generator import Generator


TEST_TRUE_AUDIO_DIR = 'test/true/'
PRED_AUDIO_DIR = 'test/pred/'
SAMPLE_RATE = 22050
MODEL_PATH = 'first_try/generator.pth'
device = 'cuda'
featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)


generator = Generator(80, 512, [16,16,4,4], [3,7,11], [[1, 3, 5, 1, 1, 1], [1, 3, 5, 1, 1, 1], [1, 3, 5, 1, 1, 1]]).to(device)
generator.load_state_dict(torch.load(MODEL_PATH))


if __name__ == '__main__':
    test_files = os.listdir(TEST_TRUE_AUDIO_DIR)
    for filename in test_files:
        print(filename)
        filepath = TEST_TRUE_AUDIO_DIR + filename
        audio_tensor, sr = torchaudio.load(filepath)
        if sr != SAMPLE_RATE:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, SAMPLE_RATE)
        audio_tensor = audio_tensor.to(device)
        melspec = featurizer(audio_tensor)
        predicted_audio = generator(melspec).squeeze(1)
        torchaudio.save(PRED_AUDIO_DIR + 'pred_' + filename, predicted_audio.detach().cpu(), SAMPLE_RATE)

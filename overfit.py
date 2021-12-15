import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.data.dataset import LJSpeechDataset
from src.data.collator import LJSpeechCollator
from src.models.generator import Generator
import wandb
from tqdm import tqdm

device = 'cpu'

collator = LJSpeechCollator(device)
featurizer = collator.featurizer
dataloader = DataLoader(LJSpeechDataset('.'), batch_size=10, collate_fn=collator)
#generator = Generator(80, 128, [16,16,4,4], [3,7,11], [[1,3,5], [1,3,5], [1,3,5]]).to(device)
generator = Generator(80, 32, [16,16,4,4], [3], [[1]]).to(device)

l1loss = nn.L1Loss()
optimizer = optim.Adam(generator.parameters(), lr=0.001)

log_audio_every = 250
log_loss_every = 10

# with wandb.init(project="tts_2_vocoder", name="overfit") as run:
batch = next(iter(dataloader))
    for i in tqdm(range(1, 5000)):
        melspec = batch['melspec']
        pred_wave = generator(melspec).squeeze(1)
        pred_wave = pred_wave[:, :batch['waveform'].shape[1]]
        pred_mel = featurizer(pred_wave)

        mask = (torch.arange(pred_mel.shape[1])[None, :].to(device)  <= batch['melspec_length'][:, None]).float()
        loss_mel = l1loss(pred_mel * mask[:, :, None], melspec * mask[:, :, None])
        generator_loss = loss_mel

    #     if i % log_loss_every == 0:
    #         run.log({"loss" : loss})
    #     if i % log_audio_every == 0:
    #         print("Logging audio")
    #         mel_to_log = result[0]
    #         melspec_to_log  = result[0][:, :batch['melspec_length'][0]].unsqueeze(0)
    #         reconstructed_wav = vocoder.inference(melspec_to_log).squeeze().detach().cpu().numpy()
    #         run.log({"Audio" : wandb.Audio(reconstructed_wav, 22050)})
        optimizer.zero_grad()
        generator_loss.backward()
        optimizer.step()

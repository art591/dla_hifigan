import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.data.dataset import LJSpeechDataset
from src.data.collator import LJSpeechCollator
from src.models.generator import Generator
from src.models.discriminators import Discriminator
import wandb
from tqdm import tqdm


def get_discriminators_results(disc, tensor, is_fake):
    targets = []
    results, features = disc(tensor)
    for res in results:
        if is_fake:
            targets.append(torch.zeros(res.shape).to(device))
        else:
            targets.append(torch.ones(res.shape).to(device))
    return results, features, targets

def calc_disc_adv_loss(fake_disc_output, fake_targets, real_disc_output, real_targets):
    loss = 0
    for fo, ft, ro, rt in zip(fake_disc_output, fake_targets, real_disc_output, real_targets):
        loss += 0.5 * (AdvLoss(fo, ft) + AdvLoss(ro, rt))
    return loss / len(fake_disc_output)

def calc_gen_adv_loss(gen_disc_output, gen_disc_targets, gen_disc_features, real_disc_features):
    adv_loss = 0
    for go, gt in zip(gen_disc_output, gen_disc_targets):
        adv_loss += AdvLoss(go, gt)
    adv_loss = adv_loss / len(gen_disc_output)
    feature_loss = 0
    for gf, rf in zip(gen_disc_features, real_disc_features):
        feature_loss += FeatureLoss(gf, rf)
    feature_loss = feature_loss / len(gen_disc_features)
    return adv_loss, feature_loss


def train(run,
          generator,
          discriminator,
          gen_optimizer,
          disc_optimizer,
          train_dataloader,
          epoch,
          update_disc_every,
          log_audio_every,
          log_loss_every):
    generator.train()
    discriminator.train()
    i = 0
    for batch in tqdm(train_dataloader):
        melspec = batch['melspec']
        real_wave = batch['waveform']
        pred_wave = generator(melspec).squeeze(1)
        pred_wave_detached = pred_wave.detach()
        pred_wave = pred_wave[:, :real_wave.shape[1]]
        pred_mel = featurizer(pred_wave)

        mask = (torch.arange(pred_mel.shape[1])[None, :].to(device)  <= batch['melspec_length'][:, None]).float()
        gen_mel_loss = MelLoss(pred_mel * mask[:, :, None], melspec * mask[:, :, None])

        gen_disc_output, gen_disc_features, gen_disc_targets = get_discriminators_results(discriminator, pred_wave, False)
        _, real_disc_features, _ = get_discriminators_results(discriminator, real_wave, False)

        gen_adv_loss, gen_feature_loss = calc_gen_adv_loss(gen_disc_output, gen_disc_targets, gen_disc_features, real_disc_features)
        gen_loss = 45 * gen_mel_loss + 2 * gen_feature_loss + gen_adv_loss

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()
        if i % update_disc_every == 0:
            fake_disc_output, _, fake_targets = get_discriminators_results(discriminator, pred_wave_detached, True)
            real_disc_output, _, real_targets = get_discriminators_results(discriminator, real_wave, False)
            disc_loss = calc_disc_adv_loss(fake_disc_output, fake_targets, real_disc_output, real_targets)

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
        if i % log_loss_every == 0 and i != 0:
            run.log({"Train Gen loss" : gen_loss}, step=epoch * len(train_dataloader) + i)
            run.log({"Train Gen Mel Loss" : gen_mel_loss}, step=epoch * len(train_dataloader) + i)
            run.log({"Train Gen Adv loss" : gen_adv_loss}, step=epoch * len(train_dataloader) + i)
            run.log({"Train Gen Features Loss" : gen_feature_loss}, step=epoch * len(train_dataloader) + i)
            run.log({"Learning rate" : gen_optimizer.param_groups[0]['lr']}, step=epoch * len(train_dataloader) + i)
            if i % update_disc_every == 0:
                run.log({"Train Discr Loss" : disc_loss}, step=epoch * len(train_dataloader) + i)
        if i % log_audio_every == 0 and i != 0:
            pred_audios = []
            real_audios = []
            for k in range(min(3, batch_size)):
                reconstructed_wav = pred_wave[k].detach().cpu().numpy()
                pred_audios.append(wandb.Audio(reconstructed_wav, 22050, caption=f"pred_{k}"))
                reconstructed_wav = real_wave[k].detach().cpu().numpy()
                real_audios.append(wandb.Audio(reconstructed_wav, 22050, caption=f"real_{k}"))
            
            run.log({"Train Predicted Audio" : pred_audios}, step=epoch * len(train_dataloader) + i)
            run.log({"Train Real Audio" : real_audios}, step=epoch * len(train_dataloader) + i)
        i += 1

def validation(run, iteration, model):
    model.eval()
    pass


if __name__ == '__main__':
    project_name = 'tts_2'
    name = 'first_try'
    experiment_path = name
    log_audio_every = 100
    log_loss_every = 5
    update_disc_every = 1
    save_every = 1
    n_epochs = 100
    batch_size = 16
    segment_size = 8192
    device = 'cuda'

    collator = LJSpeechCollator(device)
    featurizer = collator.featurizer
    train_dataloader = DataLoader(LJSpeechDataset('.', segment_size), batch_size=batch_size, collate_fn=collator)
    generator = Generator(80, 512, [16,16,4,4], [3,7,11], [[1, 3, 5, 1, 1, 1], [1, 3, 5, 1, 1, 1], [1, 3, 5, 1, 1, 1]]).to(device)
    discriminator = Discriminator().to(device)

    gen_optimizer = optim.AdamW(generator.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01)
    disc_optimizer = optim.AdamW(discriminator.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01)
    gen_scheduler = optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma=0.999)
    disc_scheduler = optim.lr_scheduler.ExponentialLR(disc_optimizer, gamma=0.999)


    AdvLoss = nn.MSELoss()
    FeatureLoss = nn.L1Loss()
    MelLoss = nn.MSELoss()
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    with wandb.init(project=project_name, name=name) as run:
        for i in range(n_epochs):
            print(f'Start Epoch {i}')
            train(run,
                  generator,
                  discriminator,
                  gen_optimizer,
                  disc_optimizer,
                  train_dataloader,
                  i,
                  update_disc_every,
                  log_audio_every,
                  log_loss_every)
            if i % save_every == 0:
                torch.save(generator.state_dict(), f"{experiment_path}/generator.pth")
            #validation(run, (i + 1) * len(train_dataloader), model)
            gen_scheduler.step()
            disc_scheduler.step()

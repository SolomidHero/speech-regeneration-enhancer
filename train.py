#!/usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import datetime
from tqdm.auto import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from torch.optim import AdamW
from modules import (
  Generator, Discriminator,
  MultiResolutionSTFTLoss, adversarial_loss, discriminator_loss
)
from utils import (
  FeatureDataset, load_dataset_filelist,
  scan_checkpoint, load_checkpoint, save_checkpoint
)

from omegaconf import DictConfig, OmegaConf
import hydra

torch.backends.cudnn.benchmark = True


def train(rank: int, cfg: DictConfig):
  print(OmegaConf.to_yaml(cfg))

  if cfg.train.n_gpu > 1:
    init_process_group(
      backend=cfg.train.dist_config['dist_backend'], init_method=cfg.train.dist_config['dist_url'],
      world_size=cfg.train.dist_config['world_size'] * cfg.train.n_gpu, rank=rank
    )

  device = torch.device('cuda:{:d}'.format(rank) if torch.cuda.is_available() else 'cpu')
  criterion = MultiResolutionSTFTLoss(device=device)

  # Defining model (generator and discriminator)
  generator = Generator(
    sum(cfg.model.feature_dims),
    *cfg.model.cond_dims,
    **cfg.model.generator
  ).to(device)
  discriminator = Discriminator(**cfg.model.discriminator).to(device)

  if rank == 0:
    print(generator)
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    print("checkpoints directory : ", cfg.train.ckpt_dir)

  # Loading checkpoints (if exist)
  if os.path.isdir(cfg.train.ckpt_dir):
    cp_g = scan_checkpoint(cfg.train.ckpt_dir, 'g_')
    cp_do = scan_checkpoint(cfg.train.ckpt_dir, 'd_')

  steps = 1
  if cp_g is None or cp_do is None:
    state_dict_do = None
    last_epoch = -1
  else:
    state_dict_g = load_checkpoint(cp_g, device)
    state_dict_do = load_checkpoint(cp_do, device)
    generator.load_state_dict(state_dict_g['generator'])
    discriminator.load_state_dict(state_dict_do['discriminator'])
    steps = state_dict_do['steps'] + 1
    last_epoch = state_dict_do['epoch']

  if cfg.train.n_gpu > 1:
    generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
    discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

  # Defining optimizers
  optim_g = AdamW(generator.parameters(), cfg.opt.lr, betas=cfg.opt.betas)
  optim_d = AdamW(discriminator.parameters(), cfg.opt.lr, betas=cfg.opt.betas)

  if state_dict_do is not None:
    optim_g.load_state_dict(state_dict_do['optim_g'])
    optim_d.load_state_dict(state_dict_do['optim_d'])

  # Defining schedulers
  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=cfg.opt.lr_decay, last_epoch=last_epoch)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=cfg.opt.lr_decay, last_epoch=last_epoch)

  # Data preparation
  print("Preparing train data...")
  train_filelist = load_dataset_filelist(cfg.dataset.train_list)
  trainset = FeatureDataset(
    cfg.dataset, train_filelist, cfg.data, segmented=True, preload_gt=True, seed=cfg.seed
  )
  train_sampler = DistributedSampler(trainset) if cfg.train.n_gpu > 1 else None
  train_loader = DataLoader(
    trainset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True,
    sampler=train_sampler, pin_memory=True, drop_last=True
  )

  if rank == 0:
    print("Preparing validation data...")
    val_filelist = load_dataset_filelist(cfg.dataset.test_list)
    valset = FeatureDataset(
      cfg.dataset, val_filelist, cfg.data, segmented=True, preload_gt=True,
      segment_size=cfg.data.segment_size * cfg.train.batch_size
    )
    val_loader = DataLoader(
      valset, batch_size=1, num_workers=cfg.train.num_workers, shuffle=False,
      sampler=train_sampler, pin_memory=True
    )

    log_dir = f'{cfg.train.logs_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    sw = SummaryWriter(log_dir)

  # Train loop
  generator.train()
  discriminator.train()
  for epoch in range(max(0, last_epoch), cfg.train.epochs):
    if rank == 0:
      start = time.time()
      print("Epoch: {}".format(epoch+1))

    if cfg.train.n_gpu > 1:
      train_sampler.set_epoch(epoch)

    for y, x_noised_features, x_noised_cond in train_loader:
      if rank == 0:
        start_b = time.time()

      y = y.to(device, non_blocking=True)
      x_noised_features = x_noised_features.transpose(1, 2).to(device, non_blocking=True)
      x_noised_cond = x_noised_cond.to(device, non_blocking=True)
      z1 = torch.randn(cfg.train.batch_size, cfg.model.cond_dims[1], device=device)
      z2 = torch.randn(cfg.train.batch_size, cfg.model.cond_dims[1], device=device)

      y_hat1 = generator(x_noised_features, x_noised_cond, z=z1)
      y_hat2 = generator(x_noised_features, x_noised_cond, z=z2)


      # Discriminator
      real_scores, fake_scores = discriminator(y), discriminator(y_hat1.detach())
      d_loss = discriminator_loss(real_scores, fake_scores)

      optim_d.zero_grad()
      d_loss.backward()
      d_grad_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.train.grad_norm_clip_value)
      optim_d.step()

      # Generator
      fake_scores = discriminator(y_hat1)
      g_stft_loss = criterion(y, y_hat1) + criterion(y, y_hat2) - criterion(y_hat1, y_hat2)
      g_adv_loss = adversarial_loss(fake_scores)
      g_loss = g_adv_loss + cfg.train.lambda_adv * g_stft_loss

      optim_g.zero_grad()
      g_loss.backward()
      g_grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.train.grad_norm_clip_value)
      optim_g.step()

      if rank == 0:
        # stdout logging
        if steps % cfg.train.stdout_interval == 0:
          with torch.no_grad():
            print('Steps : {:d}, Gen Loss Total : {:4.3f}, STFT Error : {:4.3f}, s/b : {:4.3f}'.
              format(steps, g_loss, g_stft_loss, time.time() - start_b))

        # checkpointing
        if steps % cfg.train.checkpoint_interval == 0:
          ckpt_dir = "{}/g_{:08d}".format(cfg.train.ckpt_dir, steps)
          save_checkpoint(
            ckpt_dir,
            { 'generator': (generator.module if cfg.train.n_gpu > 1 else generator).state_dict() }
          )
          ckpt_dir = "{}/do_{:08d}".format(cfg.train.ckpt_dir, steps)
          save_checkpoint(
            ckpt_dir, {
            'discriminator': (discriminator.module if cfg.train.n_gpu > 1 else discriminator).state_dict(),
            'optim_g': optim_g.state_dict(),
            'optim_d': optim_d.state_dict(),
            'steps': steps,
            'epoch': epoch
          })

        # Tensorboard summary logging
        if steps % cfg.train.summary_interval == 0:
          sw.add_scalar("loss/g_loss_total", g_loss, steps)
          sw.add_scalar("loss/g_adv_plus_d_total", g_adv_loss.item() + d_loss.item(), steps)

          sw.add_scalar("loss_component/g_stft_error", g_stft_loss, steps)
          sw.add_scalar("loss_component/g_adv_loss", g_adv_loss, steps)
          sw.add_scalar("loss_component/d_loss", d_loss, steps)

          sw.add_scalar("grad/d_grad_norm", d_grad_norm, steps)
          sw.add_scalar("grad/g_grad_norm", g_grad_norm, steps)

        # Validation
        if steps % cfg.train.validation_interval == 0:
          generator.eval()
          torch.cuda.empty_cache()

          val_err_tot = 0
          val_pbar = tqdm(total=len(valset), ncols=0, desc="Valid", unit=" uttr")

          for j, (y, x_noised_features, x_noised_cond) in enumerate(val_loader):
            with torch.no_grad():
              y_hat = generator(x_noised_features.transpose(1, 2).to(device), x_noised_cond.to(device))
              val_err_tot += criterion(y.to(device), y_hat).item()

            if j <= 4:
              sw.add_audio('generated/y_hat_{}'.format(j), y_hat[0], steps, cfg.data.target_sample_rate)
              sw.add_audio('gt/y_{}'.format(j), y[0], steps, cfg.data.target_sample_rate)

            val_pbar.update(val_loader.batch_size)
            val_pbar.set_postfix(loss=f"{val_err_tot / (j + 1):.2f}")

          val_err = val_err_tot / (j + 1)
          sw.add_scalar("validation/stft_error", val_err, steps)

          val_pbar.close()
          generator.train()

      steps += 1

    scheduler_g.step()
    scheduler_d.step()

    if rank == 0:
      print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


@hydra.main(config_name="config")
def main(cfg: DictConfig):
  torch.manual_seed(cfg.seed)

  if torch.cuda.is_available():
    torch.cuda.manual_seed(cfg.seed)
    cfg.train.n_gpu = torch.cuda.device_count()
    cfg.train.batch_size = int(cfg.train.batch_size / cfg.train.n_gpu)
    print('Batch size per GPU :', cfg.train.batch_size)
  else:
    cfg.train.n_gpu = 0
    print('No GPU registered for training!')

  if cfg.train.n_gpu > 1:
    mp.spawn(train, nprocs=cfg.train.n_gpu, args=(cfg,))
  else:
    train(0, cfg)


if __name__ == '__main__':
  main()

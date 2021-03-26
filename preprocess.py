#!/usr/bin/env python3
"""Precompute Wav2Vec2, f0, speaker embedding features."""

import os
import json
from pathlib import Path
from multiprocessing import cpu_count

import tqdm
import torch
from torch.utils.data import DataLoader
from jsonargparse import ArgumentParser, ActionConfigFile

from features import get_ppg, get_f0, get_loudness, get_speaker_embed, denoise
from utils import (
  PreprocessDataset,
  get_datafolder_files, define_train_list, train_test_split, save_dataset_filelist
)


import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name="config")
def main(cfg: DictConfig):
  """Preprocessing function for DAPS-like dataset (https://archive.org/details/daps_dataset).
  - Extracts features (PPG, f0, loudness, spk embedding) for every wav in datafolder
  - Builds train and test list
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print(OmegaConf.to_yaml(cfg.dataset))
  print(OmegaConf.to_yaml(cfg.data))
  print('device:', device)

  # folders preparation
  for out_dir_path in [cfg.dataset.f0_dir, cfg.dataset.ppg_dir, cfg.dataset.loudness_dir]:
    out_dir_path = Path(out_dir_path)
    if out_dir_path.exists():
      assert out_dir_path.is_dir()
    else:
      out_dir_path.mkdir(parents=True)

  # preprocess dataset and loader
  filepathes = get_datafolder_files(cfg.dataset.wav_dir)
  dataset = PreprocessDataset(filepathes, cfg.data)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=cfg.train.num_workers)
  spk_embs = dict()


  # feature extraction
  pbar = tqdm.tqdm(total=len(dataset), ncols=0)
  for wav, filename in dataloader:
    # batch size is 1
    wav = denoise(wav[0].numpy(), device=device)
    filename = Path(filename[0])

    with torch.no_grad():
      ppg = torch.from_numpy(get_ppg(wav, cfg.data.sample_rate, device=device))
      f0 = torch.from_numpy(get_f0(
        wav, cfg.data.sample_rate,
        hop_ms=cfg.data.hop_length * 1000 / cfg.data.sample_rate, f_min=cfg.data.f_min, f_max=cfg.data.f_max
      ))
      loudness = torch.from_numpy(get_loudness(
        wav, cfg.data.sample_rate, n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length, win_length=cfg.data.win_length
      ))
      spk_emb = torch.from_numpy(get_speaker_embed(wav, cfg.data.sample_rate, device=device)).cpu()


    torch.save(ppg, os.path.join(cfg.dataset.ppg_dir, filename.with_suffix('.pt')))
    torch.save(f0, os.path.join(cfg.dataset.f0_dir, filename.with_suffix('.pt')))
    torch.save(loudness, os.path.join(cfg.dataset.loudness_dir, filename.with_suffix('.pt')))
    spk_embs[filename.stem] = spk_emb

    pbar.update(dataloader.batch_size)

  torch.save(spk_embs, cfg.dataset.spk_embs_file)

  # generation of train and test files
  train_list = define_train_list(filepathes)
  train_list, test_list = train_test_split(train_list)
  save_dataset_filelist(train_list, cfg.dataset.train_list)
  save_dataset_filelist(test_list, cfg.dataset.test_list)


if __name__ == "__main__":
  main()
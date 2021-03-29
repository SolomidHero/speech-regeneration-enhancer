# this test uses 
# - tests/test_dataset_config.yaml config for cfg.dataset
# - tests/test_train_config.yaml config for cfg.train
# instead of default in root directory


from preprocess import main as preprocess_main
from train import main as train_main

import os
import pytest
import torch
from pathlib import Path
from hydra.experimental import compose, initialize


@pytest.fixture(scope="module")
def n_files(cfg):
  n_files = 0
  for _, _, filenames in os.walk(cfg.dataset.wav_dir):
    for name in filenames:
      if Path(name).suffix == '.wav':
        n_files += 1

  return n_files


def test_config_consistency(cfg):
  with initialize(config_path="../"):
    train_config = compose(config_name="config")

  assert set(train_config.dataset.keys()) == set(cfg.dataset.keys())
  assert set(train_config.train.keys()) == set(cfg.train.keys())


def test_preprocess(cfg, n_files):
  preprocess_main(cfg)

  assert len(os.listdir(cfg.dataset.ppg_dir)) == n_files
  assert len(os.listdir(cfg.dataset.f0_dir)) == n_files
  assert len(os.listdir(cfg.dataset.loudness_dir)) == n_files

  spk_embs = torch.load(cfg.dataset.spk_embs_file)
  assert len(spk_embs) == n_files


def test_train(cfg):
  cfg.train.n_gpu = 0
  train_main(cfg)

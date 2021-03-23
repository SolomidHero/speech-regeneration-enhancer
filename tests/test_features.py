import pytest
import numpy as np

def test_config_feature_dim_values(cfg):
  assert len(cfg.model.feature_dims) == 3
  assert len(cfg.model.cond_dims) == 2


def test_denoise(example_wav):
  from features import denoise

  denoised = denoise(example_wav['wav'])

  assert len(denoised.shape) == 1
  assert denoised.shape == example_wav['wav'].shape
  assert denoised.dtype == np.float32


@pytest.fixture(scope='module')
def speaker_embed(example_wav):
  from features import get_speaker_embed
  speaker_embed = get_speaker_embed(example_wav['wav'], example_wav['sr'])
  return speaker_embed

def test_speaker_embed(speaker_embed):
  assert len(speaker_embed.shape) == 1


@pytest.fixture(scope='module')
def loudness(example_wav, n_fft, hop_length, win_length, n_frames):
  from features import get_loudness
  loudness = get_loudness(
    example_wav['wav'],
    example_wav['sr'],
    n_fft=n_fft, hop_length=hop_length, win_length=win_length
  )
  return loudness

def test_loudness(loudness, n_frames):
  assert len(loudness.shape) == 2
  assert len(loudness) == n_frames
  assert loudness.dtype == np.float32


@pytest.fixture(scope='module')
def f0(example_wav, hop_ms, f_min, f_max, n_frames):
  from features import get_f0
  f0 = get_f0(example_wav['wav'], example_wav['sr'], hop_ms, f_min, f_max)
  return f0

def test_f0(f0, n_frames):
  assert len(f0.shape) == 2
  assert len(f0) == n_frames
  assert f0.dtype == np.float32


@pytest.fixture(scope='module')
def ppg(example_wav, hop_ms):
  from features import get_ppg
  ppg = get_ppg(example_wav['wav'], example_wav['sr'], backend='wav2vec2')
  return ppg

def test_ppg(ppg, example_wav):
  # wav2vec2 has 320 stride, receptive field (window) 400
  desired_len = example_wav['wav'].shape[-1] // 320 + 1
  assert len(ppg.shape) == 2
  assert ppg.shape[0] == desired_len
  assert ppg.dtype == np.float32


def test_features_alignment(speaker_embed, loudness, f0, ppg):
  assert ppg.shape[0] == f0.shape[0]
  assert ppg.shape[0] == loudness.shape[0]


def test_feature_dims_with_config(cfg, ppg, f0, loudness, speaker_embed):
  # source features
  assert ppg.shape[1] == cfg.model.feature_dims[0], f"extracted PPG feature dim ({ppg.shape[1]}) must match in config ({cfg.model.feature_dims[0]})"
  assert f0.shape[1] == cfg.model.feature_dims[1], f"extracted f0 feature dim ({f0.shape[1]}) must match in config ({cfg.model.feature_dims[1]})"
  assert loudness.shape[1] == cfg.model.feature_dims[2], f"extracted loudness feature dim ({loudness.shape[1]}) must match in config ({cfg.model.feature_dims[2]})"

  # condition features
  assert speaker_embed.shape[0] == cfg.model.cond_dims[0], f"extracted speaker embedding feature dim ({speaker_embed.shape[0]}) must match in config ({cfg.model.cond_dims[0]})"

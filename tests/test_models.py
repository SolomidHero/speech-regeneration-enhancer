import torch
from functools import reduce
import pytest
from modules import (
  Generator, Discriminator,
  MultiResolutionSTFTLoss, adversarial_loss, discriminator_loss
)

@pytest.fixture(autouse=True, scope="module")
def n_frames():
  return 13

@pytest.fixture(autouse=True, scope="module")
def origin_wav_len(n_frames, cfg):
  return n_frames * cfg.data.hop_length

@pytest.fixture(autouse=True, scope="module")
def n_features(cfg):
  return sum(cfg.model.feature_dims)

@pytest.fixture(autouse=True, scope="module")
def cond_dim(cfg):
  return cfg.model.cond_dims[0]

@pytest.fixture(autouse=True, scope="module")
def z_dim(cfg):
  return cfg.model.cond_dims[1]

@pytest.fixture(autouse=True, scope="module")
def generator(cfg, n_features, cond_dim, z_dim):
  generator = Generator(n_features, cond_dim, z_dim, **cfg.model.generator)
  return generator

@pytest.fixture(autouse=True, scope="module")
def discriminator(cfg):
  discriminator = Discriminator(**cfg.model.discriminator)
  return discriminator


def test_generator(cfg, generator, n_features, cond_dim, z_dim, n_frames):
  total_upsample = reduce(lambda x, y: x * y, cfg.model.generator.upsamples)

  # assert total_upsample == cfg.data.hop_length, \
  #   "upsampling factor and hop_length should match"

  features = torch.randn(1, n_features, n_frames)
  cond = torch.randn(1, cond_dim)
  z = torch.randn(1, z_dim)

  generator.eval()
  with torch.no_grad():
    output = generator(features, cond)
    output_with_z = generator(features, cond, z=z)

  assert output.shape == output_with_z.shape
  assert output.size(1) == n_frames * total_upsample, \
    f"generator output shape {output.shape} expected to have {n_frames} samples"


def test_generator_with_grad(cfg, generator, n_features, cond_dim, z_dim, n_frames):
  features = torch.randn(1, n_features, n_frames)
  cond = torch.randn(1, cond_dim)
  z = torch.randn(1, z_dim)

  generator.train()
  output = generator(features, cond, z=z)
  MultiResolutionSTFTLoss()(output, output.detach()).backward()


def test_discriminator(cfg, discriminator, origin_wav_len):
  total_downsample = reduce(lambda x, y: x * y, cfg.model.discriminator.downsamples)

  wav = torch.randn(1, origin_wav_len)
  with torch.no_grad():
    scores = discriminator(wav)

  assert scores.size(1) <= origin_wav_len // total_downsample, \
    f"discriminator output shape {scores.shape} expected to have {origin_wav_len // total_downsample} or less samples"


def test_discriminator_with_grad(cfg, discriminator, origin_wav_len):
  wav = torch.randn(1, origin_wav_len)
  scores = discriminator(wav)
  adversarial_loss(scores).backward()


def test_discriminator_loss_with_grad(cfg, generator, discriminator, n_features, cond_dim, z_dim, n_frames, origin_wav_len):
  features = torch.randn(1, n_features, n_frames)
  cond = torch.randn(1, cond_dim)
  z = torch.randn(1, z_dim)
  wav = torch.randn(1, origin_wav_len)

  generator.train()
  discriminator.train()

  fake_wav = generator(features, cond, z=z)
  real_scores, fake_scores = discriminator(wav), discriminator(fake_wav.detach())

  d_loss = discriminator_loss(fake_scores, real_scores)
  d_loss.backward()


def test_generator_loss_with_grad(cfg, generator, discriminator, n_features, cond_dim, z_dim, n_frames, origin_wav_len):
  total_upsample = reduce(lambda x, y: x * y, cfg.model.generator.upsamples)
  features = torch.randn(1, n_features, n_frames)
  cond = torch.randn(1, cond_dim)
  z = torch.randn(1, z_dim)
  wav = torch.randn(1, n_frames * total_upsample)

  generator.train()
  discriminator.train()

  fake_wav = generator(features, cond, z=z)
  fake_scores = discriminator(fake_wav)

  g_loss = adversarial_loss(fake_scores) + MultiResolutionSTFTLoss()(fake_wav, wav)
  g_loss.backward()
# models.py
# Implementation of necessary modules and enhancer model itself.

import torch
from .commons import Conv1d
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ConditionalBatchNorm1d(nn.Module):
  def __init__(self, n_features, cond_dim, eps=1e-5, momentum=0.1):
    super().__init__()
    self.n_features = n_features
    self.bn = nn.BatchNorm1d(n_features, affine=False, eps=eps, momentum=momentum,)

    # linear layers
    self.gamma_embed = spectral_norm(Conv1d(cond_dim, n_features, 1, bias=False))
    self.beta_embed = spectral_norm(Conv1d(cond_dim, n_features, 1, bias=False))

  def forward(self, x, y):
    out = self.bn(x)
    gamma = self.gamma_embed(y) + 1
    beta = self.beta_embed(y)
    out = gamma.view(-1, self.n_features, 1) * out + beta.view(-1, self.n_features, 1)
    return out


class ResBlock(nn.Module):
  """
    1-dimensional ResBlock (with or w/o upsample only)
    This is half of GBlock presented in reference.
    Convolution in skip-connection and upsample here is an option

    Reference: https://arxiv.org/pdf/1909.11646.pdf
  """
  def __init__(self, in_channel, out_channel, cond_dim,
    kernel_size=3, padding=1, stride=1, dilations=(1, 2),
    bn=True, activation=F.relu, upsample=1,
  ):
    super().__init__()


    self.upsample = upsample
    self.activation = activation

    self.conv0 = spectral_norm(
      Conv1d(in_channel, out_channel, kernel_size, stride, padding, bias=not bn)
    )
    self.conv1 = spectral_norm(
      Conv1d(out_channel, out_channel, kernel_size, stride, padding, bias=not bn)
    )

    self.skip_proj = False
    if in_channel != out_channel or upsample != 1:
      self.conv_skip = spectral_norm(Conv1d(in_channel, out_channel, 1, 1, 0))
      self.skip_proj = True

    self.bn = bn
    if bn:
      self.cbn0 = ConditionalBatchNorm1d(in_channel, cond_dim)
      self.cbn1 = ConditionalBatchNorm1d(out_channel, cond_dim)

  def forward(self, x, cond):
    skip = x

    # first conv layers
    if self.bn:
      x = self.cbn0(x, cond)
    x = self.activation(x)
    if self.upsample != 1:
      x = F.interpolate(x, scale_factor=self.upsample)
    x = self.conv0(x)

    # second conv layers
    if self.bn:
      x = self.cbn1(x, cond)
    x = self.activation(x)
    x = self.conv1(x)

    # skip connection
    if self.upsample != 1:
      skip = F.interpolate(skip, scale_factor=self.upsample)
    if self.skip_proj:
      skip = self.conv_skip(skip)
    return x + skip


class GBlock(nn.Module):
  def __init__(self, in_channel, out_channel, cond_dim,
    upsample=1, dilations=(1, 2, 4, 8)
  ):
    super().__init__()

    self.gblock = nn.ModuleList([
      ResBlock(in_channel, out_channel, cond_dim, dilations=dilations[0:2], upsample=upsample),
      ResBlock(out_channel, out_channel, cond_dim, dilations=dilations[2:4]),
    ])

  def forward(self, x, cond):
    for resblock in self.gblock:
      x = resblock(x, cond)

    return x


class Generator(nn.Module):
  def __init__(self, in_channel, cond_dim, z_dim, hidden_dim=768,
    n_blocks=7, upsamples=[2, 2, 2, 3, 4, 5], channel_divs=[1, 1, 2, 1, 1, 2, 2],
  ):
    super().__init__()

    self.z_dim = z_dim
    upsamples = list(upsamples)
    channel_divs = list(channel_divs)

    assert len(upsamples) <= n_blocks, \
      f"Number of blocks with upsample ({len(upsamples)} must be <= total number of blocks ({n_blocks})"
    assert len(channel_divs) == n_blocks, \
      f"Number of channel divisions {channel_divs} for blocks != n_blocks ({n_blocks})"

    upsamples = [1] * (n_blocks - len(upsamples)) + upsamples

    from itertools import accumulate
    channel_divs = list(accumulate([1] + channel_divs, lambda x, y: x * y))

    self.input_conv = spectral_norm(Conv1d(in_channel, hidden_dim, 3, 1, 1))
    self.gblocks = nn.ModuleList([
      GBlock(hidden_dim // channel_divs[i], hidden_dim // channel_divs[i + 1], cond_dim + z_dim, upsamples[i])
      for i in range(len(upsamples))
    ])
    self.output_conv = spectral_norm(Conv1d(hidden_dim // channel_divs[-1], 1, 1, 1, 0))
    self.tanh = nn.Tanh()

  def forward(self, x, cond, z=None):
    if z is None:
      z = torch.randn(x.shape[0], self.z_dim, device=cond.device)

    z = z.unsqueeze(2) if len(z.shape) == 2 else z
    cond = cond.unsqueeze(2) if len(cond.shape) == 2 else cond
    cond = torch.cat([cond, z], dim=1)

    x = self.input_conv(x)
    for block in self.gblocks:
      x = block(x, cond)
    x = self.output_conv(x).flatten(1, -1)

    x = self.tanh(x)
    return x


class Discriminator(nn.Module):
  def __init__(self,
    out_channels=[16, 64, 256, 1024, 1024, 1024, 1],
    kernels=[15, 41, 41, 41, 41, 5, 3],
    downsamples=[1, 2, 2, 4, 4, 1, 1],
    lrelu_slope=0.2,
  ):
    super().__init__()

    assert out_channels[-1] == 1, "out_channels last value must be 1"
    assert len(out_channels) == len(kernels) and len(kernels) == len(downsamples), \
      "out_channels, kernel sizes, downsamples numbers must match"

    self.lrelu = nn.LeakyReLU(lrelu_slope)
    self.convs = nn.ModuleList([
      spectral_norm(Conv1d(in_c, out_c, kernel_size, stride=down, padding=0))
      for in_c, out_c, kernel_size, down in zip([1] + out_channels[:-1], out_channels, kernels, downsamples)
    ])

  def forward(self, x):
    if len(x.shape) < 3:
      x = x.unsqueeze(1)

    for i, l in enumerate(self.convs):
      x = l(x)
      if i + 1 == len(self.convs):
        break
      x = self.lrelu(x)

    x = torch.flatten(x, 1, -1)

    return x

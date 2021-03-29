import torch
import torch.nn as nn
import torch.nn.functional as F


def adversarial_loss(scores, as_real=True):
  if as_real:
    return torch.mean((1 - scores) ** 2)
  return torch.mean(scores ** 2)


def discriminator_loss(fake_scores, real_scores):
  loss = adversarial_loss(fake_scores, as_real=False) + adversarial_loss(real_scores, as_real=True)
  return loss


def stft(x, n_fft, hop_length, win_length, window, eps=1e-6):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
      x: Input signal tensor (B, T).
    Returns:
      Tensor: Magnitude spectrogram (B, T, n_fft // 2 + 1).
    """
    x_stft = torch.stft(x,
      n_fft, hop_length, win_length, window,
      center=False, return_complex=True
    ).abs().clamp(min=eps)

    return x_stft


class SpectralConvergence(nn.Module):
  def __init__(self):
    """Initilize spectral convergence loss module."""
    super().__init__()

  def forward(self, predicts_mag, targets_mag):
    """Calculate norm of difference operator.
    Args:
      predicts_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
      targets_mag  (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
    Returns:
      Tensor: Spectral convergence loss value.
    """

    return torch.mean(
      torch.norm(targets_mag - predicts_mag, dim=(1, 2), p='fro') / torch.norm(targets_mag, dim=(1, 2), p='fro')
    )


class LogSTFTMagnitude(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, predicts_mag, targets_mag):
    log_predicts_mag = torch.log(predicts_mag)
    log_targets_mag = torch.log(targets_mag)

    outputs = F.l1_loss(log_predicts_mag, log_targets_mag)

    return outputs


class STFTLoss(nn.Module):
  def __init__(self, n_fft, hop_length, win_length, device='cpu'):
    super().__init__()

    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    self.window = torch.hann_window(win_length).to(device)
    self.sc_loss = SpectralConvergence()
    self.mag_loss = LogSTFTMagnitude()

  def forward(self, predicts, targets):
    """
    Args:
        x: predicted signal (B, T).
        y: truth signal (B, T).
    Returns:
        Tensor: STFT loss values.
    """

    predicts_mag = stft(predicts, self.n_fft, self.hop_length, self.win_length, self.window)
    targets_mag = stft(targets, self.n_fft, self.hop_length, self.win_length, self.window)

    sc_loss = self.sc_loss(predicts_mag, targets_mag)
    mag_loss = self.mag_loss(predicts_mag, targets_mag)

    return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
  def __init__(self,
    fft_sizes=[2048, 1024, 512, 256, 128, 64],
    win_sizes=[2048, 1024, 512, 256, 128, 64],
    hop_sizes=[512, 256, 128, 64, 32, 16],
    device='cpu'
  ):
    super().__init__()

    self.loss_layers = torch.nn.ModuleList([
      STFTLoss(n_fft, hop_length, win_length, device=device)
      for n_fft, win_length, hop_length in zip(fft_sizes, win_sizes, hop_sizes)
    ])

  def forward(self, fake_signals, true_signals):
    res_losses = []
    for layer in self.loss_layers:
      sc_loss, mag_loss = layer(fake_signals, true_signals)
      res_losses.append(sc_loss + mag_loss)

    loss = sum(res_losses) / len(res_losses)

    return loss
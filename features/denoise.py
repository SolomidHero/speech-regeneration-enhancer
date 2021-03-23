# denoising.py
# Preprocessing of audio with high-quality denoising.
# facebookresearch/denoiser based on demucs architecture
# is used as denoise model

import torch

def load_denoiser(model_name="dns64"):
  return torch.hub.load("facebookresearch/denoiser", "dns64", force_reload=False).eval()

def denoise(wav, sr=None, device='cpu'):
  """
  Denoise .wav audio data
  Args:
    wav    - waveform (numpy array)
    device - (defaul 'cpu')
  Returns:
    wav    - same wav, denoised
  """
  model = load_denoiser().to(device)
  with torch.no_grad():
    res = model(torch.from_numpy(wav).unsqueeze(0).to(device))
  return res.squeeze().cpu().numpy()
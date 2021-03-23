# speaker_embed.py
# Speaker Embedding feature extraction utils.
# Dense vector of constant dim, which contains representation 
# of speaker prosody and style

from pyannote.audio import Inference
import torch

def load_pyannote_audio(ckpt_path='hbredin/SpeakerEmbedding-XVectorMFCC-VoxCeleb', device='cpu'):
  """Load speaker embedding model from pyannote.audio"""
  model = Inference(ckpt_path, device=device, window='sliding')
  return model

def normalize(embed):
  return embed / (embed ** 2).sum(-1, keepdims=True) ** 0.5

def get_speaker_embed(wav, sr, device='cpu', backend='pyannote'):
  if backend == 'pyannote':
    model = load_pyannote_audio(device=device)
    if len(wav.shape) == 1:
      wav = wav[None]
    spk_emb = model({
      'waveform': torch.from_numpy(wav).to(device),
      'sample_rate': sr,
    }).data.mean(0)
    spk_emb = normalize(spk_emb)

  return spk_emb
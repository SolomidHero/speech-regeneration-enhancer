# ppg.py
# PPG feature extraction utils.
# Phonetic Posteriorgrams are fully linguistic features
# and are often represented by bottleneck feature in ASR network

from transformers import Wav2Vec2Model
import torch


def load_wav2vec2(ckpt_path='facebook/wav2vec2-base-960h'):
  """Load pretrained Wav2Vec2 model."""
  def extract_features(self, wav, mask):
    # wav2vec has window of 400, so we pad to center windows
    wav = torch.nn.functional.pad(wav.unsqueeze(1), (200, 200), mode='reflect').squeeze(1)
    return [self(wav).last_hidden_state]

  Wav2Vec2Model.extract_features = extract_features # for same behaviour as fairseq.Wav2Vec2Model
  model = Wav2Vec2Model.from_pretrained(ckpt_path)
  return model


def get_ppg(wav, sr, device='cpu', backend='wav2vec2'):
  if backend == 'wav2vec2':
    model = load_wav2vec2().to(device)

    with torch.no_grad():
      ppg = model.extract_features(torch.from_numpy(wav).unsqueeze(0).to(device), None)[0]

  return ppg.squeeze(0).cpu().numpy()
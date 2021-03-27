# ppg.py
# PPG feature extraction utils.
# Phonetic Posteriorgrams are fully linguistic features
# and are often represented by bottleneck feature in ASR network

from transformers import Wav2Vec2Model
import torch


def load_wav2vec2(ckpt_path='facebook/wav2vec2-base-960h'):
  """Load pretrained Wav2Vec2 model."""
  def extract_features(self, wav, mask):
    return [self(wav).last_hidden_state]

  Wav2Vec2Model.extract_features = extract_features # for same behaviour as fairseq.Wav2Vec2Model
  model = Wav2Vec2Model.from_pretrained(ckpt_path).eval()
  return model


def get_ppg(wav, sr, device='cpu', backend='wav2vec2', max_window=20.0, overlap=2.0):
  wav = torch.from_numpy(wav).unsqueeze(0).to(device)

  if backend == 'wav2vec2':
    # wav2vec has window of 400 and hop of 320,
    # so we pad to center windows
    hop_length = 320
    win_length = 400

    n_frames = wav.shape[1] // hop_length + 1
    wav = torch.nn.functional.pad(wav.unsqueeze(1), (win_length // 2, win_length // 2), mode='reflect').squeeze(1)
    model = load_wav2vec2().to(device)

    with torch.no_grad():
      if wav.shape[-1] / sr > max_window:
        segment_n_frames = int(max_window * sr) // hop_length
        segment_n_points = segment_n_frames * hop_length
        overlap_n_frames = int(overlap * sr) // hop_length
        overlap_n_points = overlap_n_frames * hop_length
        hop_segment_len = segment_n_points - overlap_n_points

        n_segments = (wav.shape[-1] - segment_n_points) // hop_segment_len + 1
        ppgs = []

        # process ppg for every window, except last
        for i in range(n_segments):
          sub_wav = wav[:, i * hop_segment_len:i * hop_segment_len + segment_n_points + win_length - hop_length]
          cur_ppg = model.extract_features(sub_wav, None)[0].squeeze(0).cpu()
          cur_ppg = cur_ppg[overlap_n_points // hop_length:] if i > 0 else cur_ppg
          ppgs.append(cur_ppg)

        # add last window
        n_frames_calced = (n_segments - 1) * hop_segment_len // hop_length + segment_n_frames
        n_frames_left = n_frames - n_frames_calced

        sub_wav = wav[:, (n_frames - segment_n_frames) * hop_length:]
        cur_ppg = model.extract_features(sub_wav, None)[0].squeeze(0).cpu()
        cur_ppg = cur_ppg[-n_frames_left:]
        ppgs.append(cur_ppg)

        # cat into one ppg
        ppg = torch.cat(ppgs, dim=0)
      else:
        ppg = model.extract_features(wav, None)[0].squeeze(0).cpu()

  return ppg.numpy()
# loudness.py
# loudness feature extraction utils.
# Basically it is a percieved mesure of signal energy.
# Here we use weighting of power spectrogram frequencies,
# then mean and log for each frame.

import librosa
import numpy as np


def get_loudness(wav, sr, n_fft=1280, hop_length=320, win_length=None, ref=1.0, min_db=-80.0):
  """
  Extract the loudness measurement of the signal.
  Feature is extracted using A-weighting of the signal frequencies.

  Args:
    wav          - waveform (numpy array)
    sr           - sampling rate
    n_fft        - number of points for fft
    hop_length   - stride of stft
    win_length   - size of window of stft
    ref          - reference for amplitude log-scale
    min_db       - floor for db difference
  Returns:
    loudness     - loudness of signal, shape (n_frames,) 
  """

  A_weighting = librosa.A_weighting(librosa.fft_frequencies(sr, n_fft=n_fft)+1e-6, min_db=min_db)
  weighting = 10 ** (A_weighting / 10)

  power_spec = abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)) ** 2
  loudness = np.mean(power_spec * weighting[:, None], axis=0)
  loudness = librosa.power_to_db(loudness, ref=ref) # in db

  return loudness[:, np.newaxis].astype(np.float32)
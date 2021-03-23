# f0.py
# f0 (main signal frequency) extraction utils.
# f0 contour (a.k.a. pitch contour) shows the current
# dominate note/frequency for each frame

import numpy as np
import scipy.interpolate
import pyworld as pw

# best practice is to make f0 continuous and logarithmed
def convert_continuos_f0(f0):
  """CONVERT F0 TO CONTINUOUS F0
  Reference:
  https://github.com/bigpon/vcc20_baseline_cyclevae/blob/master/baseline/src/bin/feature_extract.py

  Args:
      f0 (ndarray): original f0 sequence with the shape (T)
  Return:
      (ndarray): continuous f0 with the shape (T)
  """
  # get uv information as binary
  uv = np.float32(f0 != 0)

  # get start and end of f0
  start_f0 = f0[f0 != 0][0]
  end_f0 = f0[f0 != 0][-1]

  # padding start and end of f0 sequence
  start_idx = np.where(f0 == start_f0)[0][0]
  end_idx = np.where(f0 == end_f0)[0][-1]
  f0[:start_idx] = start_f0
  f0[end_idx:] = end_f0

  # get non-zero frame index
  nz_frames = np.where(f0 != 0)[0]

  # perform linear interpolation
  f = scipy.interpolate.interp1d(nz_frames, f0[nz_frames])
  cont_f0 = f(np.arange(0, f0.shape[0]))

  return np.log(cont_f0)
  # return uv, np.log(cont_f0)


def get_f0(wav, sr, hop_ms, f_min=0, f_max=None):
  """
  Extract f0 (1d-array of frame values) from wav (1d-array of point values).
  Args:
    wav    - waveform (numpy array)
    sr     - sampling rate
    hop_ms - stride (in milliseconds) for frames
    f_min  - f0 floor frequency
    f_max  - f0 ceil frequency
  Returns:
    f0     - interpolated main frequency, shape (n_frames,) 
  """
  if f_max is None:
    f_max = sr / 2

  _f0, t = pw.dio(wav.astype(np.float64), sr, frame_period=hop_ms, f0_floor=f_min, f0_ceil=f_max) # raw pitch extractor
  f0 = pw.stonemask(wav.astype(np.float64), _f0, t, sr)  # pitch refinement

  return convert_continuos_f0(f0)[:, np.newaxis].astype(np.float32)
import math
import os
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.utils.data
import torchaudio

import numpy as np
import librosa
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn


def load_wav(path, sr=16000):
  """Load audio from path and resample it
  Return: 1d numpy.array of audio data
  """
  wav = librosa.load(path, sr=sr, dtype=np.float32, mono=True)[0]
  return wav


def get_datafolder_files(datafolder_path, pattern='.wav'):
  """Get all files with specified extension in directory tree
  Return: list of file pathes
  """
  filelist = []

  for root, _, filenames in os.walk(datafolder_path):
    for filename in filenames:
      if Path(filename).suffix == pattern:
        filelist.append(os.path.join(root, filename))

  return filelist


def define_train_list(filepathes, clean_suffix='clean', n_utterance_tokens=2):
  """Return dict in following format:
  { utterance_name : [filepath, filename1, filename2...] },
  where first value of list is groundtruth file's path
  for all other files (with same utterance)
  """
  assert clean_suffix in ['cleanraw', 'clean', 'produced']
  train_list = defaultdict(list)

  for filepath in filepathes:
    p = Path(filepath)
    tokens = p.stem.split('_')
    utterance = '_'.join(tokens[:n_utterance_tokens])

    if tokens[-1] == clean_suffix:
      train_list[utterance] = [filepath] + train_list[utterance]
    else:
      train_list[utterance].append(p.stem)

  return train_list


def train_test_split(filelist, p=0.85, seed=17):
  """Return train and test set of filenames
  This function follows `define_train_list` and uses its output
  """
  random.seed(seed)
  train_list, test_list = dict(), dict()

  for utterance, files in filelist.items():
    gt_filepath, filenames = files[0], files[1:]
    random.shuffle(filenames)

    val_len = int((1 - p) * len(filenames))

    train_list[utterance] = [gt_filepath] + filenames[val_len:]
    test_list[utterance] = [gt_filepath] + filenames[:val_len]

  return train_list, test_list


def save_dataset_filelist(filelist, filelist_path, delim='|'):
  with open(filelist_path, 'w', encoding='utf-8') as f:
    for utterance, files in filelist.items():
      print(utterance + delim + delim.join(files), file=f)


def load_dataset_filelist(filelist_path, delim='|'):
  filelist = dict()
  with open(filelist_path, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if len(line) <= 0:
        continue
      tokens = line.split(delim)
      utterance, files = tokens[0], tokens[1:]
      filelist[utterance] = files

  return filelist


class PreprocessDataset(torch.utils.data.Dataset):
  """
    Torch Dataset class for wav files for their following feature extraction.
    Assumes that wav files (all names are different)
    can be in some subdirectories of root dir.
  """
  def __init__(self, filepathes, data_cfg):
    self.data_cfg = data_cfg
    self.audio_pathes = filepathes
    self.filenames = list(map(lambda p: Path(p).name, filepathes))

  def __getitem__(self, index):
    file_path = self.audio_pathes[index]
    filename = self.filenames[index]
    wav = load_wav(file_path, sr=self.data_cfg.sample_rate)

    return wav, filename

  def __len__(self):
    return len(self.audio_pathes)


class FeatureDataset(torch.utils.data.Dataset):
  """
    Torch Dataset class for wav files and their features.
    Assumes that wav files (all names are different)
    can be in some subdirectories of root dir,
    but feature files in corresponding feature directories alone.
  """
  def __init__(self, dataset_cfg, filelist, data_cfg, preload_gt=True, segmented=True, segment_size=None, seed=17):
    self.data_cfg = data_cfg
    self.dataset_cfg = dataset_cfg
    self.filelist = filelist
    self.segmented = segmented
    self.upsampling_rate = self.data_cfg.hop_length * self.data_cfg.target_sample_rate // self.data_cfg.sample_rate
    if self.segmented:
      segment_size = self.data_cfg.segment_size if segment_size is None else segment_size

      assert segment_size % (self.data_cfg.hop_length * self.data_cfg.target_sample_rate // self.data_cfg.sample_rate) == 0

      self.n_points = segment_size
      self.n_frames = self.n_points // self.upsampling_rate

    self.noise_to_gt_dict = dict()
    for files in filelist.values():
      # first is ground truth
      self.noise_to_gt_dict[Path(files[0]).stem] = files[0]
      for filename in files[1:]:
        self.noise_to_gt_dict[filename] = files[0]

    self.filenames = list(map(lambda p: Path(p).stem, self.noise_to_gt_dict.keys()))
    self.gt_list = set(self.noise_to_gt_dict.values())

    self.preload_gt = preload_gt
    if preload_gt:
      self.gt_data = {
        gt_path : load_wav(gt_path, sr=self.data_cfg.target_sample_rate)
        for gt_path in self.gt_list
      }

    self.spk_embs = torch.load(self.dataset_cfg.spk_embs_file)

    random.seed(seed)

  def __getitem__(self, index):
    filename = self.filenames[index]
    gt_path = self.noise_to_gt_dict[filename]

    if self.preload_gt:
      gt_wav = self.gt_data[gt_path]
    else:
      gt_wav = load_wav(gt_path, sr=self.data_cfg.target_sample_rate)

    # features: [ppg, f0, loudness]
    features = []
    for feature_dir in (
      self.dataset_cfg.ppg_dir, self.dataset_cfg.f0_dir, self.dataset_cfg.loudness_dir
    ):
      feat = torch.load(os.path.join(feature_dir, Path(filename).with_suffix('.pt')))
      features.append(feat)
    features = torch.cat(features, dim=1)

    # condition: [embed]
    cond = self.spk_embs[filename]

    # because center=True for feature extraction
    n_pad = self.upsampling_rate // 2
    gt_wav = np.pad(gt_wav, (n_pad, n_pad), mode='reflect')[:self.upsampling_rate * features.shape[0]]

    if self.segmented:
      start_frame = random.randint(0, features.shape[0] - self.n_frames)
      start_point = start_frame * self.data_cfg.hop_length

      gt_wav = gt_wav[start_point:start_point + self.n_points]
      features = features[start_frame:start_frame + self.n_frames]

    return gt_wav, features, cond

  def __len__(self):
    return len(self.filenames)
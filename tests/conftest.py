# here we make fixtures of toy data
# real parameters are stored and accessed from config


import pytest
import librosa
import os
import numpy as np

from hydra.experimental import compose, initialize


@pytest.fixture(scope="session")
def cfg():
  with initialize(config_path="../", job_name="test_app"):
    config = compose(config_name="config")
    config.dataset = compose(config_name="tests/test_dataset_config")
    config.train = compose(config_name="tests/test_train_config")

    return config

@pytest.fixture(scope="session")
def sample_rate(cfg):
  return cfg.data.sample_rate

@pytest.fixture(scope="session")
def example_wav(sample_rate):
  wav, sr = librosa.load(
    os.path.dirname(__file__) + "/data/example.mp3",
    sr=sample_rate, dtype=np.float32,
  )
  return { 'wav': wav, 'sr': sr }

@pytest.fixture(scope="session")
def n_fft(cfg):
  return cfg.data.n_fft

@pytest.fixture(scope="session")
def hop_length(cfg):
  return cfg.data.hop_length

@pytest.fixture(scope="session")
def win_length(cfg):
  return cfg.data.win_length

@pytest.fixture(scope="session")
def f_min(cfg):
  return cfg.data.f_min

@pytest.fixture(scope="session")
def f_max(cfg):
  return cfg.data.f_max

@pytest.fixture(scope="session")
def hop_ms(example_wav, hop_length):
  return 1e3 * hop_length / example_wav['sr']

@pytest.fixture(scope="session")
def n_frames(example_wav, hop_length):
  return (example_wav['wav'].shape[-1] - 1) // hop_length + 1

# It is not clear if we should cleanup the test directories
# or leave them for debugging
# https://github.com/pytest-dev/pytest/issues/3051
@pytest.fixture(autouse=True, scope='session')
def clear_files_teardown():
  yield None
  os.system("rm -r tests/test_dataset tests/test_experiment")
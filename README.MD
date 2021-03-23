# Regeneration Enhancer ![python app workflow](https://github.com/github/docs/actions/workflows/python-app.yml/badge.svg)

This repository provides speech enhancement via regeneration implementation with Pytorch. Algorithm is based on [paper](https://arxiv.org/abs/2102.00429), but several changes were made in feature extraction and therefore model parameters.

**TODO list:**
- add inference scripts
- implement streaming model and its inference
- provide multilingual enhancement models (and adapt feature extraction too)
- make pypi package

# Requirements

This repository is tested on Ubuntu 16.04 with a GPU 1080 Ti.

- Python 3.6+
- Cuda 10.0+
<!-- CuDNN 7+
NCCL 2+ (for distributed multi-gpu training) -->
- libsndfile (you can install via `sudo apt install libsndfile-dev` in ubuntu)
<!-- - sox (you can install via `sudo apt install sox` in ubuntu) -->
- pip requirements (defined in `requirements.txt`, install via `pip install -r requirements.txt`):
  - hydra-core 2.0+
  - pytorch 1.7+
  - torchaudio 0.7.2+
  - librosa 0.8.0+
  - pytest 6.2.0+
  - transformers 4.3.0+, pyworld 0.2.12+, pyannote.audio 2.0+ (for feature extraction)

# Installation

```bash
git clone https://github.com/SolomidHero/speech-regeneration-enhancer
pip install -e ./speech-regeneration-enhancer
```

# Training

For training you should use [DAPS dataset](https://archive.org/details/daps_dataset), or dataset with similar file namings (folder structure doesn't matter):

```
data_folder/
  wav_1_clean.wav
  dirty/
    wav_1_recoder_bathroom.wav
    wav_2_microphone_street.wav
  some_sub_tree/
    wav_2_clean.wav
```

In this repository we use hydra configuration ([read more](https://hydra.cc/)), thus for training and inference you can only change `config.yaml` file. Also defining through parameters in bash is available.

When changes to config are made, you can check yourself if your parameters are acceptable by any of these commands:
```bash
pytest                       # to check if everything is working
pytest tests/test_scripts.py # to check if training process can be done
```

1. After data downloading and config changes, run preprocessing script (feature extraction made here):

```bash
preprocess.py dataset.wav_dir=/path/to/wavs # parameters can be added into config directly
```

2. Finally we are able to train model:

```bash
train.py train.epochs=50 train.ckpt_dir=/path/to/ckpts # parameters can be added into config directly
```

In `/path/to/ckpt` checkpoints for generator and other stuff (discriminator, optimizers) will appear from now.


# Reference

- paper: *["HIGH FIDELITY SPEECH REGENERATION WITH APPLICATION TO SPEECH ENHANCEMENT"](https://arxiv.org/abs/2102.00429)*
- data: [Device and Produced Speech Dataset](https://archive.org/details/daps_dataset)
- [wav2vec2](https://huggingface.co/transformers/model_doc/wav2vec2.html)
- [pyannote-audio](https://github.com/pyannote/pyannote-audio/)
- [python wrapper for world](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)
- [demucs denoiser](https://github.com/facebookresearch/denoiser/)

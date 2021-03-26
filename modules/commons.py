# commons.py
# Implementation of some torch.nn modules

import torch.nn as nn


class Conv1d(nn.Conv1d):
  "Conv1d with orthogonal initialisation"
  def reset_parameters(self) -> None:
    super().reset_parameters()
    nn.init.orthogonal_(self.weight)

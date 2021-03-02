import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.optim import Variable

class LottoNetwork(nn.Module):
  def __init__(self, in_size, out_size):
    super(LottoNetwork, self).__init__()
    self.network = self.build_network(in_size, out_size)

  def build_network(self, in_size, out_size):
    return nn.Sequential(
          nn.Linear(in_size, 1024),
          nn.ReLU(),
          nn.BatchNorm1d(1024),
          nn.Linear(1024, 256),
          nn.ReLU(),
          nn.BatchNorm1d(256),
          nn.Linear(256, out_size))

  def forward(self, x):
    return torch.sigmoid(self.network(x))        

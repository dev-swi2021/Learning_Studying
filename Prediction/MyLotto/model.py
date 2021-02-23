import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.optim import Variable

class LottoNetwork(nn.Module):
  def __init__(self, in_size, mid_size, out_size):
    super(LottoNetwork, self).__init__()
    self.front_network = self.build_front_network(in_size, mid_size)
    self.branch_network = self.build_branch_network(mid_size, out_size)
    
  def build_front_network(self, in_size, out_size):
    return nn.Sequential(
      nn.Linear(in_size, 1024),
      nn.ReLU(),
      nn.BatchNorm1d(1024),
      nn.Linear(1024, 256),
      nn.ReLU(),
      nn.BatchNorm1d(256),
      nn.Linear(256, out_size))
   
  def build_branch_network(self, in_size, out_size):
    h_size = in_size // 4
    layers = nn.Sequential(nn.Linear(in_size, h_size), nn.ReLU(),nn.BatchNorm1d(h)size), nn.Linear(h_size, out_size))
    return nn.ModuleList([layers for _ in range(6)]
  
  def forward(self, x):
    x = self.front_network(x)
    return torch.dstack([torch.sigmoid(network(x)) for network in self.branch_network]).transpose(2,1)

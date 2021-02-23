import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils

class CustomDataset(Dataset):
  def __init__(self, data, k, train=False):
    super(CustomDataset, self).__init__()
    """
        :params data: 학습 혹은 테스트에 사용될 데이터
        :params k: 주차(ex: k=7, 하나의 데이터는 7주치)
        :params train: 학습 데이터인지, 테스트 데이터인지(True/False)
    """
    self.data = data
    self.k = k
    self.train = train
    self.feature, self.target = self._process()
    
  def __len__(self):
    return len(self.target)
  
  def __getitem__(self, idx):
    feature = torch.from_numpy(self.feature[idx]).float()
    target = torch.from_numpy(self.target[idx]).float()
    return feature, target
  
  def _process(self):
    """
      Feature 추가 해야함
    """
    
    last = self.data.shape[0] - (self.k - 1)
    one_hot_encode = 45
    
    target = self.data[self.data.columns[pd.Series(self.data.columns).str(startswith('다음번호')]]
    feature = self.data.drop(columns=self.data.columns[pd.Series(self.data.columns).str.startswith('다음번호')])
    
    # transform k data --> one data
    feature_k_lst = [feature[i : i+self.k].reset_index(drop=True).to_numpy().reshape(-1,self.k*feature.shape[1]) for i in range(last_)]
    target_k_lst = [np.eye(one_hot_encode)[target.iloc[i].to_numpy()] for i in range(self.k-1, last_)] if self.train else None  # 각 데이터마다 (6,46) 크기
    # target_k_lst = [target.iloc[i].to_numpy() for i in range(self.k-1, last_)] if self.train else None  # 각 데이터마다 (6,46) 크기
    return feature_k_lst, target_k_lst
    

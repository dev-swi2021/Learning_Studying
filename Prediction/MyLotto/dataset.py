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
    target = torch.from_numpy(self.target[idx]).float() if self.train else None
    return feature, target
  
  def _transform_one_hot_encode(self, numbers):
      number_lst = [0]*45 # 0->1, 44->45
      for n in numbers:
        number_lst[n-1] = 1
      return np.array(number_lst)
    
  def _process(self):
    """
      Feature 추가 해야함
    """
    last_ = self.data.shape[0] - self.k
    tmp_data = self.data.drop(columns=['회차'])
    feature_k_lst = [tmp_data.iloc[idx:idx+self.k].to_numpy().reshape(-1, self.k*tmp_data.shape[1]) for idx in range(last_)]
    target_k_lst = [self._transform_one_hot_encode(tmp_data.loc[idx+self.k, '번호_1':'번호_6'].to_numpy()) for idx in range(last_)] if self.train else None
    for idx in range(last_):
      feature_k_lst.append(tmp_data.iloc[idx:idx+self.k].to_numpy().reshape(-1,self.k*tmp_data.shape[1]))
      if self.train:
        # 정답 데이터 one-hot으로 변경
        target_k_lst.append(self._transform_one_hot_encode(tmp_data.loc[idx+self.k, '번호_1':'번호_6'].to_numpy()))  # 0->1, 44->45
    return feature_k_lst, target_k_lst    

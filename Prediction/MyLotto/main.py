import os
import sys
import time
import random
import numpy as np
import pandas as pd

from torch.utils.data improt Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils

import torch
import torch.nn as nn
from get_lotto_number import GetLottoDataset
from model import LottoNetwork

from torchsummary import summary
from tqdm.notebook import tqdm

USE_CUDA = True if torch.cuda.is_available() else False
data_path = 'path/to/your/dataset'

# HyperParameter Settings
EPOCHS = 300
INPUT_SIZE = 49
MID_SIZE = 256
OUTPUT_SIZE = 45
LEARNING_RATE = 0.05
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED) if USE_CUDA else torch.manual_seed(RANDOM_SEED)

def train(epoch, model, optimizer, loss_fn, batch_loader, cuda):
  model.train()
  loss_lst = []
  for idx, (features, targets) in enumerate(batch_loader):
    optimizer.zero_grad()
    if cuda:
      features = features.cuda()
      targets = targets.cuda()
    
    with torch.set_grad_enabled(True):
      preds = model(features.squeeze(1))
      # print(preds.shape, targets.shape)
      _loss = loss_fn(targets, preds)
      _loss.backward()
      optimizer.step()
      loss_lst.append(_loss.cpu().detach() if cuda else _loss.detach())
      return np.mean(loss_lst)

def eval(model, batch_loader):
  model.eval()
  with torch.no_grad():
    return model(batch_loader)
  



# 정답 데이터는 그 다음 주 당첨 번호이므로 합쳐서 관리
lotto_df = pd.read_csv(data_path)
_lotto_df = pd.merge(lotto_df, lotto_df.loc[1:,:].reset_index(), how='left', left_index=True, right_index=True).drop(columns=['회차_x','index', '회차_y','번호_보너스_y'])
_lotto_df.rename(columns={'번호_1_x': '번호_1', '번호_2_x': '번호_2', '번호_3_x': '번호_3', '번호_4_x': '번호_4', '번호_5_x': '번호_5', '번호_6_x': '번호_6','번호_보너스_x': '번호_보너스', 
                          '번호_1_y': '다음번호_1', '번호_2_y': '다음번호_2', '번호_3_y': '다음번호_3', '번호_4_y': '다음번호_4', '번호_5_y': '다음번호_5', '번호_6_y': '다음번호_6'},inplace=True)
_lotto_df.fillna(0, inplace=True)
_lotto_df = _lotto_df.astype({'다음번호_1':'int', '다음번호_2': 'int', '다음번호_3':'int', '다음번호_4':'int', '다음번호_5':'int', '다음번호_6':'int'})

# 과적합 방지를 위해 train set, validation set 분리하기
split_size = int(_lotto_df.shape[0]*0.85)
train, valid = _lotto_df[:split_size], _lotto_df[split_size:].reset_index(drop=True)

train_dataset = CustomDataset(data=train, k=7, train=True)
valid_dataset = CustomDataset(data=valid, k=7, train=True)

# data loader 만들기
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

print("Train DataFrame shape : {}, Validation DataFrame shape : {}".format(train.shape, valid.shape))
print("Train Dataset Length : {}, Validation Dataset Length : {}".format(train_dataset.__len__(), valid_dataset.__len__()))

network = LottoNetwork(INPUT_SIZE, MID_SIZE, OUTPUT_SIZE).cuda() if USE_CUDA else LottoNetwork(INPUT_SIZE, MID_SIZE, OUTPUT_SIZE)
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
loss_fn = nn.MultiLabelSoftMarginLoss()

aggregated_losses = []

for e in range(1, epochs+1):
  e_loss = train(e, network, optimizer, loss_fn, train_dataloader, USE_CUDA)
  aggregated_losses.append(e_loss.item())
  
  if e % 50 == 1:
    print(f'epoch: {e:3} loss: {e_loss.item():10.8f')
    lr_schuduler.step()

print(f'epoch: {e:3} loss: {np.mean(aggregated_losses):10.10f}')


for features, targets in valid_dataloader:
  res = eval(model, features.squeze(1))

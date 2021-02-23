import os
import sys
import time
import random
import numpy as np
import pandas as pd

from torch.utils.data improt Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils

from get_lotto_number import GetLottoDataset
from model import LottoNetwork

from torchsummary import summary
from tqdm.notebook import tqdm

USE_CUDA = True if torch.cuda.is_available() else False
data_path = 'path/to/your/dataset'

lotto_df = pd.read_csv(data_path)

# HyperParameter Settings
EPOCHS = 300
INPUT_SIZE = 49
MID_SIZE = 256
OUTPUT_SIZE = 45
LEARNING_RATE = 0.05



import torch
import numpy as np

SEED = 42

def set_seed(seed = SEED):
  torch.manual_seed(seed)
  np.random.seed(seed)
  
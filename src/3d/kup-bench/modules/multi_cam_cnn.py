from typing import Optional
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from rlbench.demo import Demo

from tqdm import tqdm as progress
from lib.cam_type import CamType

from modules.demo_obs_dataset import DemoObsDataset

class CNNEncoder(nn.Module):
  def __init__(self):
    super(CNNEncoder, self).__init__()
    self.conv_encode = nn.Sequential(
      # 3 64 64
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      # 32 31 31 
      nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      # 48 14 14
      nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      # 64 6 6
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      # 128 2 2
    )
    
    
  ## given an image batch with multi cams (batch_size, num_cams, c, w, h)
  def forward(self, image):
    return self.conv_encode(image)
    
class MultiCamCnn(nn.Module):
  def __init__(self, cam_type: CamType):
    ## here we want a combination of cam_types, single is also possible
    super(MultiCamCnn, self).__init__()
    
    cnns = {}
    if cam_type & CamType.WRIST:
      cnns[f"{CamType.WRIST}"] = CNNEncoder()
    if cam_type & CamType.LEFT_SHOULDER:
      cnns[f"{CamType.LEFT_SHOULDER}"] = CNNEncoder()
    if cam_type & CamType.RIGHT_SHOULDER:
      cnns[f"{CamType.RIGHT_SHOULDER}"] = CNNEncoder()
    
    self.out_shape = (128, 2, 2) ## this is per CNNEncoder
    
    ## ModuleDict indexing must be done with strings
    self.conv_encodes = nn.ModuleDict(cnns)
    
  def forward(self, image, cam_type: CamType):
    
    assert cam_type.is_single_type(), f"[multi_cam_cnn] The camera type given '{cam_type}' has multiple cameras in it"
    
    return self.conv_encodes[f"{cam_type}"](image) #type: ignore lets see if this works
    


        
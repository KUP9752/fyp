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
from modules.cam_type import CamType

from modules.demo_obs_dataset import DemoObsDataset


class PolicyHead(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super(PolicyHead, self).__init__()
        self.policy_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, fused_feature):
        return self.policy_mlp(fused_feature)  # (batch_size, action_dim)

class CameraAttention(nn.Module):
  def __init__(self, feature_dim, hidden_dim):
    super(CameraAttention, self).__init__()
    self.attention_mlp = nn.Sequential(
      nn.Linear(feature_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, 1)  # Output 1 score per camera
    )

  def forward(self, features):  
    # features: (batch_size, num_cameras, feature_dim)
    scores: torch.Tensor = self.attention_mlp(features)  # (batch_size, num_cameras, 1)
    scores = scores.squeeze(-1)            # (batch_size, num_cameras)

    return F.softmax(scores, dim=1)  # (batch_size, num_cameras)


class CamAttentionPolicy(nn.Module):
  def __init__(
    self, 
    action_shape: int, 
    cam_type: CamType,
    feat_dim = 128,
    cam_att_hidden_dim = 64,
  ):
    self.cam_type = cam_type
    super(CamAttentionPolicy, self).__init__()
    print(f"[cam_attention_policy] - Policy] Using {self.cam_type} as camera type")
    
    self.num_cams = 0
    if cam_type & CamType.WRIST:
      self.num_cams += 1
    if cam_type & CamType.LEFT_SHOULDER:
      self.num_cams += 1
    if cam_type & CamType.RIGHT_SHOULDER:
      self.num_cams += 1
    
    if self.num_cams == 0:
      raise ValueError("[cam_attention_policy] - Policy] No cameras selected!")
    
    self.conv_encode = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
    )
    
    self.cam_attention = CameraAttention(feat_dim, cam_att_hidden_dim)
    
    self.policy_head = PolicyHead(feat_dim, action_shape)
  
  def forward(self, images: torch.Tensor):
    batch_size, num_cams, c, h, w = images.shape
    ## ensure same number of cams given
    assert num_cams == self.num_cams, f"[cam_attention_policy] Model Creation time num cams {self.num_cams} does not match the inference time tensor shape num cams: {num_cams}"
    
    images = images.view(batch_size * num_cams, c, h, w) ## so I dont have to use lists which are cpu-side
    
    feats: torch.Tensor = self.conv_encode(images)
    feats = feats.mean(dim=[-2, -1]) ## Global Average Pooling (batch_size * num_cams, feats)
    feats = feats.view(batch_size, num_cams, -1)
    
    attention_weights: torch.Tensor = self.cam_attention(feats)
    fused_feats = (attention_weights.unsqueeze(-1) * feats).sum(dim=1)
    actions = self.policy_head(fused_feats)
    
    return actions # , attention_weights //NOTE: might need attention weights later on
    

  def train_policy(self, 
    demos: list[Demo],
    epochs: int = 1000,
    minibatch_size: int = 64, ## size of the observations currently being used
    lr: float = 1e-2,
    lr_eta_min = 1e-4,
    shuffle_data = False, 
    shuffle_obs_in_demo = False,
    model_path: Optional[str] = None
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Camera: {self.cam_type}")
    
    print(f"Training Params: \n\t{epochs = }, \n\t{minibatch_size = }, \n\t{lr = }, \n\t{lr_eta_min =}, \n\t{model_path = }, \n\t{shuffle_data = },\n\t{shuffle_obs_in_demo = }, \n\t{device}\n")
    
    
    model = self.to(device)
    
    loss_fn = nn.MSELoss()
    ## NOTE: suggested nn.Smooth1Loss()
    optimiser = optim.Adam(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max= epochs, eta_min=lr_eta_min )
    
    ## 'stack' makes sure to return all the images fuxed together (batch_size, 3, num_cam, W, H)
    dataset = DemoObsDataset(demos, self.cam_type, shuffle_obs=shuffle_obs_in_demo, get_type="stack")
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=shuffle_data) ## shuffling makes it worse
    # print(f"Dataset Size: {len(dataset)}")
    
    writer = SummaryWriter()
    
    model.train()
    
    self.losses = torch.empty(epochs, dtype=torch.float32, device = device)
    for epoch in progress(range(epochs)):
      running_loss = 0
      for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimiser.zero_grad()
        pred_actions = model(inputs)
        action_loss = loss_fn(pred_actions, labels)
        action_loss.backward()
        optimiser.step()
        scheduler.step()
        running_loss += action_loss.item()
        
      epoch_loss = running_loss / len(loader)
      self.losses[epoch] = epoch_loss

      writer.add_scalar("Loss/train", epoch_loss, epoch)
      
    print(f"Done Training Policy on {len(demos)} Demos") 
    
    if model_path:
      torch.save(self.state_dict(), f"{model_path}")

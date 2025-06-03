from typing import Literal, Optional
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
from lib.utils import params_string

from modules.dataset.demo_obs_dataset import DemoObsDataset
from modules.cnns.multi_cam_cnn import MultiCamCnn
from modules.cnns.cnn_encoder import CNNEncoder


class PolicyHead(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
      super(PolicyHead, self).__init__()
      self.policy_mlp = nn.Sequential(
          nn.Linear(feature_dim, hidden_dim),
          nn.ReLU(inplace=False),
          nn.Dropout(0.2),
          nn.Linear(hidden_dim, hidden_dim),
          nn.ReLU(inplace=False),
          nn.Dropout(0.2),
          nn.Linear(hidden_dim, hidden_dim // 4),
          nn.ReLU(inplace=False),
          nn.Linear(hidden_dim // 4, action_dim)
      )

    def forward(self, fused_feature):
        return self.policy_mlp(fused_feature)  # (batch_size, action_dim)

class CameraAttention(nn.Module):
  def __init__(self, feature_dim, hidden_dim):
    super(CameraAttention, self).__init__()
    self.attention_mlp = nn.Sequential(
      nn.Linear(feature_dim, hidden_dim),
      nn.ReLU(inplace=False),
      nn.LayerNorm(hidden_dim), ## NOTE: did not help attention collapse, still extremes but flipped the other way [0, 1] -> [0.99, 0.01] 
      nn.Linear(hidden_dim, 1)  # Output 1 score per camera
    )

  def forward(self, features, temperature: float | None = None):  
    # features: (batch_size, num_cameras, feature_dim)
    scores: torch.Tensor = self.attention_mlp(features)  # (batch_size, num_cameras, 1)
    scores = scores.squeeze(-1)            # (batch_size, num_cameras)

    ## //NOTE: normalisation, att_weights were always [0, 1] or leaning towards one cam, need normalisation
    scores = scores - scores.max(dim=1, keepdim=True)[0]
    
    if temperature:
      return F.softmax(scores / temperature, dim=1)      
    else:
      return F.softmax(scores, dim=1)  # (batch_size, num_cameras), do softmax over the different camera inputs
  
class CamAttentionOld(nn.Module):
  def __init__(
    self, 
    action_shape: int, 
    cam_type: CamType,
    feat_dim = 128,
    cam_att_hidden_dim = 64,
    is_multi_cnn: bool = True,
    target_rgb: torch.Tensor | None = None,
    colour_score_pooling: Literal["mean", "max"] = "mean"
  ):
    self.cam_type = cam_type
    super(CamAttentionOld, self).__init__()
    print(f"[cam_attention_policy] - Policy] Using {self.cam_type} as camera type")
    
    self.num_cams = 0
    if cam_type & CamType.WRIST:
      self.num_cams += 1
    if cam_type & CamType.LEFT_SHOULDER:
      self.num_cams += 1
    if cam_type & CamType.RIGHT_SHOULDER:
      self.num_cams += 1
    
    if self.num_cams == 0:
      raise ValueError("[cam_attention_policy] - CamAttentionPolicy] No cameras selected!")
    
    self.is_multi_cnn = is_multi_cnn
    
    if is_multi_cnn:
      self.conv_encode = MultiCamCnn(cam_type)
    else:
      self.conv_encode = CNNEncoder(in_channels = 3)
    
    self.cam_attention = CameraAttention(feat_dim, cam_att_hidden_dim)
    self.policy_head = PolicyHead(feat_dim, action_shape)
    
    self.target_rgb = target_rgb
    self._colour_score_pooling = colour_score_pooling
  
  def set_target_rgb(self, target_rgb: torch.Tensor):
    assert target_rgb.shape == (3), f"[cam_attention_policy - set_target_rgb] Wrong RGB format given ({target_rgb})"
    self.target_rgb = target_rgb.to(next(self.parameters()).device)
  
  ## tolerance: colour match threshold, softness: distinguishing factor "inside"/"outside" threshold
  ## greater softness -> harder thrreshold, less soft ->  colours moderately close are considered the same
  ## NOTE: should be differentiable, because of norm and sigmoid
  def _colour_score(self, img: torch.Tensor, tolerance = 0.2, softness = 50, pool: Literal["mean", "max"] | None = None) -> torch.Tensor:
    
    if self.target_rgb is None:
      raise ValueError(f"[cam_attention_policy - _differentiable_colour_score] Target RGB is not set ")
    
    ## might be set at __init__ which means it willbe on differnet device
    device = next(self.parameters()).device
    self.target_rgb = self.target_rgb.to(device)
    
    img = img / 255 ## normalise pixel values
    
    ## img; Tensor (batch_size, 3, W, H) 
    diff = img - self.target_rgb.view(1, 3, 1, 1)
    dist = torch.norm(diff, dim = 1) # euclidian distance per pixel  
    soft_mask = torch.sigmoid((tolerance - dist) * softness)
    
    ## max turns this into a binary, do i see red or not, will try that first
    
    if pool is None:
      pool = self._colour_score_pooling #type: ignore[assignment]
    
    if pool == "max":
      return soft_mask.max(dim=1)[0].max(dim=1)[0]
    elif pool == "mean":
      return soft_mask.mean(dim=[1, 2]) + 1e2#TODO: These values need normalising, they are really small usualy like 1e-4/5 small
    else:
      raise ValueError(f"[cam_attention_policy - _differentiable_colour_score] Pooling type '{pool}' not supported")
  
  
  def forward(self, images: torch.Tensor):
    batch_size, num_cams, c, w, h = images.shape
    ## ensure same number of cams given
    assert num_cams == self.num_cams, f"[cam_attention_policy] Model Creation time num cams {self.num_cams} does not match the inference time tensor shape num cams: {num_cams}"
    
    # MultiCamCNN forward pass per camera selected
    
    ## in order wrist -> ls -> rs
    to_stack = []
    target_scores = []
    
    curr_index = 0 ## in the case earlier ones don't exist, for example only `RIGHT_SHOULDER`
    
    for ct in CamType.uniques(): ## in order of declaration
      if self.cam_type & ct:
        image = images[:, curr_index, :, :, :] # (B, idx, ch, w, h)
        feats = self.conv_encode(image, ct) if self.is_multi_cnn else self.conv_encode(image)
        
        to_stack.append(feats)  
        target_scores.append(self._colour_score(image))  
        
        curr_index += 1 ## move onto next available cam
      
    
    # print()
    feats = torch.stack(to_stack, dim = 1)  ## dim = 1 so (batch_size, num_cams, feat_size, 2, 2)
    
    t_scores = torch.stack(target_scores, dim = 1) ## (b, num_cams, 1) last float being target score
    t_scores = t_scores / (t_scores.sum(dim=1, keepdim=True) + 1e-6) ## normalisation with some epsilon, maybe can use softmax?
    # print(f"1-{feats.shape = }")
    
    assert feats.shape[1] == self.num_cams, f"[cam_attention_policy] The image dimension ({feats.shape[1]}) is not the same as the number of cams being used for the policy ({self.num_cams})"  
    
    feats = feats.mean(dim=[-2, -1]) ## Global Average Pooling the other dimensions after feat size
    feats = feats.view(batch_size, num_cams, -1) ## should be (batch_size, num_cams, feat_size) here
    
    # print(f"2-{feats.shape = }")
    # CameraWise attention
    attention_weights: torch.Tensor = self.cam_attention(feats)
    # print(f"{attention_weights.shape = }")
    
    fused_feats = (attention_weights.unsqueeze(-1) * feats).sum(dim=1)
    # print(f"{fused_feats.shape = }")
    
    # Policy Head
    actions = self.policy_head(fused_feats)
    # print(f"{actions.shape = }")
    
    return actions, {
      "attention_weights": attention_weights,
      "kl_divergence": F.kl_div(attention_weights.log(), t_scores, reduction="batchmean")
      } #//NOTE: might need attention weights later on
    

  def train_policy(self, 
    demos: list[Demo],
    epochs: int = 1000,
    minibatch_size: int = 64, ## size of the observations currently being used
    lr: float = 1e-2,
    data_label: Literal["joint_velocities", "joint_positions"] = "joint_velocities",
    lr_eta_min = 1e-4,
    shuffle_data = True, 
    shuffle_obs_in_demo = False,
    lambda_attn: float = 1e-2,
    
    model_path: Optional[str] = None
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    params_str = params_string(
      epochs = epochs,
      minibatch_size = minibatch_size,
      lr = lr,
      data_label = data_label,
      lr_eta_min = lr_eta_min,
      shuffle_data = shuffle_data,
      shuffle_obs_in_demo = shuffle_obs_in_demo,
      lambda_attn = lambda_attn,
      device = device
    )

    print(f"Training Params: {params_str}")
    
    
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
      for inputs, labels, loader_dict in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimiser.zero_grad()
        pred_actions, extras = model(inputs)
        action_loss = loss_fn(pred_actions, labels)
        att_weights = extras["attention_weights"]
        attention_loss = extras["kl_divergence"]
        # print(f"{action_loss.item() = }")
        # print(f"{attention_loss.item() = }")
        
        total_loss = action_loss + lambda_attn * attention_loss
        # print(f"{total_loss.item() = }")
        
        total_loss.backward()
        # print(f"{att_weights = }")
        
        
        optimiser.step()
        scheduler.step()
        running_loss += action_loss.item()
        
      epoch_loss = running_loss / len(loader)
      self.losses[epoch] = epoch_loss

      writer.add_scalar("Loss/train", epoch_loss, epoch)
      
    print(f"Done Training Policy on {len(demos)} Demos") 
    
    if model_path:
      torch.save(self.state_dict(), f"{model_path}")
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
from modules.multi_cam_cnn import MultiCamCnn

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
  
class CamAttentionPolicy(nn.Module):
  def __init__(
    self, 
    action_shape: int, 
    cam_type: CamType,
    feat_dim = 128,
    cam_att_hidden_dim = 64,
    target_rgb: torch.Tensor | None = None
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
    
    self.conv_encode = MultiCamCnn(cam_type)
    
    self.cam_attention = CameraAttention(feat_dim, cam_att_hidden_dim)
    
    self.policy_head = PolicyHead(feat_dim, action_shape)
    
    self.target_rgb = target_rgb
  
  def set_target_rgb(self, target_rgb: torch.Tensor):
    assert target_rgb.shape == (3), f"[cam_attention_policy - set_target_rgb] Wrong RGB format given ({target_rgb})"
    self.target_rgb = target_rgb.to(next(self.parameters()).device)
  
  ## tolerance: colour match threshold, softness: distinguishing factor "inside"/"outside" threshold
  ## greater softness -> harder thrreshold, less soft ->  colours moderately close are considered the same
  def _differentiable_colour_score(self, img: torch.Tensor, tolerance = 0.2, softness = 50) -> torch.Tensor:
    
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
    return soft_mask.max(dim=1)[0].max(dim=1)[0]
    # return soft_mask.mean(dim=[1, 2]) #TODO: These values need normalising, they are really small usualy like 1e-4/5 small
  
  
  def forward(self, images: torch.Tensor):
    batch_size, num_cams, c, w, h = images.shape
    ## ensure same number of cams given
    assert num_cams == self.num_cams, f"[cam_attention_policy] Model Creation time num cams {self.num_cams} does not match the inference time tensor shape num cams: {num_cams}"
    
    device = next(self.parameters()).device
    # MultiCamCNN forward pass per camera selected
    
    ## allocate empty tensors, if they stay empty they will not be `cat`ed
    # feats_size, x, y = self.conv_encode.out_shape
    
    # wrist_feats = torch.empty(batch_size, c, w, h, device = device)
    # lshoulder_feats = torch.empty(batch_size, c, w, h, device = device)
    # rshoulder_feats = torch.empty(batch_size, c, w, h, device = device)
    
    ## in order wrist -> ls -> rs
    to_stack = []
    target_scores = []
    
    curr_index = 0 ## in the case earlier ones don't exist, for example only `RIGHT_SHOULDER`
    
    for ct in CamType.uniques(): ## in order of declaration
      if self.cam_type & ct:
        image = images[:, curr_index, :, :, :]
        feats = self.conv_encode(image, ct)
        
        to_stack.append(feats)  
        target_scores.append(self._differentiable_colour_score(image))  
        
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
        pred_actions, att_weights = model(inputs)
        action_loss = loss_fn(pred_actions, labels)
        action_loss.backward()
        
        print(f"{att_weights = }")
        ## checking if the wrist CNN is improving
        for name, param in self.conv_encode.named_parameters():
          if f"{CamType.WRIST}" in name:
            if param.grad is None:
              print(f"{name}: No Gradient")
            else:
              print(f"{name}: grad mean = {param.grad.mean().item(): .5f}, grad std = {param.grad.std().item(): .5f}")
              
        
        optimiser.step()
        scheduler.step()
        running_loss += action_loss.item()
        
      epoch_loss = running_loss / len(loader)
      self.losses[epoch] = epoch_loss

      writer.add_scalar("Loss/train", epoch_loss, epoch)
      
    print(f"Done Training Policy on {len(demos)} Demos") 
    
    if model_path:
      torch.save(self.state_dict(), f"{model_path}")

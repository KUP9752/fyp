from typing import Optional
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from rlbench.demo import Demo
from rlbench.backend.observation import Observation

from tqdm import tqdm as progress


from enum import Flag, auto


class CamType(Flag):
  WRIST = auto()
  LEFT_SHOULDER = auto()
  RIGHT_SHOULDER = auto()
  
  def __str__(self):
    parts = []
    if self & CamType.WRIST:
      parts.append("wrist")
    if self & CamType.LEFT_SHOULDER:
      parts.append("l_shoulder")
    if self & CamType.RIGHT_SHOULDER:
      parts.append("r_shoulder")
    
    return "+".join(parts)
  
  @classmethod
  ## recreates everytime, but couldn't find a good way to cache
  def all_combinations(cls):
    all_combs = []
    for i in range(1, 2**len(CamType)):
      comb = CamType(0)
      for j in range(len(CamType)):
        if i & (1 << j):
          comb |= CamType(1 << j)
      all_combs.append(comb)
    return all_combs


class DemoObsDataset(Dataset):
  def __init__(self, demos: list[Demo], cam_type: CamType, shuffle_obs: bool):
    self.cam_type = cam_type
    self.all_data = []
    seed = 42
    rng = np.random.default_rng(seed)
    for demo in demos:
      obss = demo._observations
      # print(f"[loader] Observations len: {len(obss)}")
      if shuffle_obs:
        rng.shuffle(obss)
      
      self.all_data.extend(obss)
      
  def __len__(self):
      return len(self.all_data)

  def __getitem__(self, idx):
    obs = self.all_data[idx]
    
    images = []
    
    if self.cam_type & CamType.WRIST:
      # print(f"Using Wrist Image")
      wrist_image = torch.tensor(obs.wrist_rgb, dtype = torch.float32)
      wrist_image = torch.permute(wrist_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
      images.append(wrist_image)
    if self.cam_type & CamType.LEFT_SHOULDER:
      # print(f"Using L Shouulder Image")
      ls_image = torch.tensor(obs.left_shoulder_rgb, dtype = torch.float32)
      ls_image = torch.permute(ls_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
      images.append(ls_image)
    if self.cam_type & CamType.RIGHT_SHOULDER:
      # print(f"Using R Shoulder Image")
      rs_image = torch.tensor(obs.right_shoulder_rgb, dtype = torch.float32)
      rs_image = torch.permute(rs_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
      images.append(rs_image)
    
    if not images:
      raise ValueError("[policy - DemoObsDataSet - __getitem__] No images selected !")
      
    ## this allows multi rgb cameras   
    inputs = torch.cat(images, dim = 0)  ## cat on the colours channel
    ## inputs shape should now be (3 * num_cams, 64, 64)
    labels = np.append(obs.joint_velocities, obs.gripper_open)
    labels = torch.tensor(labels, dtype = torch.float32)
    
    return inputs, labels

      
## This is made for image sizes of 64x64 and now multi cam setups
class Policy(nn.Module):
  def __init__(self, action_shape: int, cam_type: CamType = CamType.WRIST):
    super(Policy, self).__init__()
    self.cam_type = cam_type
    print(f"[policy - Policy] Using {self.cam_type} as camera type")
    
    num_cams = 0
    if cam_type & CamType.WRIST:
      num_cams += 1
    if cam_type & CamType.LEFT_SHOULDER:
      num_cams += 1
    if cam_type & CamType.RIGHT_SHOULDER:
      num_cams += 1
    
    if num_cams == 0:
      raise ValueError("[policy - Policy] No cameras selected!")
    
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels=3 * num_cams, out_channels=32, kernel_size=3, stride=1, padding=0),
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
    flat_size = 2 * 2 * 128
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(flat_size, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 50),
      nn.ReLU(inplace=False),
      nn.Linear(50, action_shape)
    )
  
  def forward(self, image):
    feats = self.conv(image)
    return self.fc(feats)
  

  def train_policy(self, 
            demos: list[Demo],
            epochs: int = 200,
            minibatch_size: int = 32, ## size of the observations currently being used
            lr: float = 0.01,
            shuffle_data = False, 
            shuffle_obs_in_demo = False,
            model_path: Optional[str] = None
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Camera: {self.cam_type}")
    
    print(f"Training Params: \n\t{epochs = }, \n\t{minibatch_size = }, \n\t{lr = }, \n\t{model_path = }, \n\t{shuffle_data = }, \n\t{device}\n")
    
    
    model = self.to(device)
    # print(f"What is in the demos: {type(demos)} | {type(demos[0])}")
    
    loss_fn = nn.MSELoss()
    ## NOTE: suggested nn.Smooth1Loss()
    optimiser = optim.Adam(model.parameters(), lr = lr)
    
    dataset = DemoObsDataset(demos, self.cam_type, shuffle_obs=shuffle_obs_in_demo)
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=shuffle_data) ## shuffling makes it worse
    # print(f"Dataset Size: {len(dataset)}")
    
    model.train()
    self.losses = [0 for _ in range(epochs)]
    for epoch in progress(range(epochs)):
      running_loss = 0
      
      for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimiser.zero_grad()
        pred_actions = model(inputs)
        loss = loss_fn(pred_actions, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
        loss = running_loss / len(demos)
        self.losses[epoch] = loss
      
    print(f"Done Training Policy on {len(demos)} Demos") 
    
    if model_path:
      torch.save(self.state_dict(), f"{model_path}")

class Agent(object):

    def __init__(self, action_shape, cam_type = CamType.WRIST):
      self.cam_type = cam_type
      self.action_shape = action_shape
      self.policy = Policy(action_shape, cam_type)

    def ingest(self, demos: list[Demo], **training_params):
      self.policy.train_policy(demos, **training_params)
      
    def act(self, obs:  Observation) -> torch.Tensor:
      # gripper = [1.0]  # Always open
      # return np.concatenate([arm, gripper], axis=-1)
      
      self.policy.eval()
      images = []
      if self.cam_type & CamType.WRIST:
        # print(f"Using Wrist Image")
        wrist_image = torch.tensor(obs.wrist_rgb, dtype = torch.float32)
        wrist_image = torch.permute(wrist_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
        images.append(wrist_image)
      if self.cam_type & CamType.LEFT_SHOULDER:
        # print(f"Using L Shouulder Image")
        ls_image = torch.tensor(obs.left_shoulder_rgb, dtype = torch.float32)
        ls_image = torch.permute(ls_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
        images.append(ls_image)
      if self.cam_type & CamType.RIGHT_SHOULDER:
        # print(f"Using R Shoulder Image")
        rs_image = torch.tensor(obs.right_shoulder_rgb, dtype = torch.float32)
        rs_image = torch.permute(rs_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
        images.append(rs_image)
      
      if not images:
        raise ValueError("[policy - Agent - act] No images selected !")
      
      torch_obs = torch.cat(images, dim = 0)  ## cat on the colours channel
      torch_obs = torch_obs.unsqueeze(0) ## add a batch dimension 1, 3 * num_cams, 64, 64
      with torch.no_grad():
        pred = self.policy(torch_obs)
      return pred
        
    def save_model(self, model_path: str):
      torch.save(self.policy.state_dict(), f'{model_path}')
      print(f"Saved Model under '{model_path}'")
      
      
    def load_model(self, model_path: str):
      self.policy.load_state_dict(torch.load(f'{model_path}'))  

import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from rlbench.demo import Demo
from rlbench.backend.observation import Observation

from tqdm import tqdm as progress
from utils import set_seed

from enum import Flag, auto

set_seed(42)

class CamType(Flag):
  WRIST = 0
  LEFT_SHOULDER = auto()
  RIGHT_SHOULDER = auto()
  
  def __str__(self):
    match self:
      case CamType.WRIST:
        return "wrist"
      case CamType.LEFT_SHOULDER:
        return "l_shoulder"
      case CamType.RIGHT_SHOULDER:
        return "r_shoulder"


class DemoObsDataset(Dataset):
  def __init__(self, demos: list[Demo], cam_type: CamType):
    self.cam_type = cam_type
    self.all_data = []
    set_seed(42)
    rng = np.random.default_rng()
    for demo in demos:
      obss = demo._observations
      # print(f"[loader] Observations len: {len(obss)}")
      
      rng.shuffle(obss)
      self.all_data.extend(obss)
      
  def __len__(self):
      return len(self.all_data)

  def __getitem__(self, idx):
    obs = self.all_data[idx]
    match self.cam_type:
      case CamType.WRIST:
        inputs, labels = obs.wrist_rgb, np.append(obs.joint_velocities, obs.gripper_open)
        # inputs, labels = zip(
        #   *[(obs.wrist_rgb, np.append(obs.joint_velocities, 1.)) for obs in obs_batch]
        # )
      case CamType.LEFT_SHOULDER:
        inputs, labels = obs.left_shoulder_rgb, np.append(obs.joint_velocities, obs.gripper_open)
      case CamType.RIGHT_SHOULDER:
        inputs, labels = obs.right_shoulder_rgb, np.append(obs.joint_velocities, 1.)
      case _:
        raise ValueError("There are no other camtype options")
      
    inputs = torch.tensor(inputs, dtype = torch.float32)
    ## batch, 64, 64, 3  -> batch, 3, 64, 64
    inputs = torch.permute(inputs, (2, 0, 1)) 
    
    labels = torch.tensor(labels, dtype = torch.float32)
    return inputs, labels

      
## This is made for imsage sizes of 64x64 and Single Cam!
class Policy(nn.Module):
  def __init__(self, action_shape: int, cam_type: CamType = CamType.WRIST):
    super(Policy, self).__init__()
    
    self.cam_type = cam_type
    
    self.conv = nn.Sequential(
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
            lr: float = 0.001,
            model_path: str = None
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Params: \n\t{epochs = }, \n\t{minibatch_size = }, \n\t{lr = }, \n\t{model_path = }, \n\t{device}\n")
    
    
    model = self.to(device)
    # print(f"What is in the demos: {type(demos)} | {type(demos[0])}")
    
    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr = lr)
    
    dataset = DemoObsDataset(demos, self.cam_type)
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=False) ## shuffling makes it worse
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
      
    ## TODO add the number of demos here later for debugging purposes
    print(f"Done Training Policy on {len(demos)} Demos") 
    
    if model_path:
      torch.save(self.state_dict(), f"{model_path}")

class Agent(object):

    def __init__(self, action_shape, cam_type = CamType.WRIST):
      self.action_shape = action_shape
      self.policy = Policy(action_shape, cam_type)

    def ingest(self, demos: list[Demo], **training_params):
      self.policy.train_policy(demos, **training_params)
      
    def act(self, obs:  Observation):
      # gripper = [1.0]  # Always open
      # return np.concatenate([arm, gripper], axis=-1)
      
      self.policy.eval()
      torch_obs = torch.tensor(obs.wrist_rgb, dtype=torch.float32)
      torch_obs = torch_obs.permute(2, 0, 1)
      torch_obs = torch_obs.unsqueeze(0)
      with torch.no_grad():
        pred = self.policy(torch_obs)
      return pred
        
    def save_model(self, model_path: str):
      torch.save(self.policy.state_dict(), f'{model_path}')
      print(f"Saved Model under '{model_path}'")
      
      
    def load_model(self, model_path: str):
      self.policy.load_state_dict(torch.load(f'{model_path}'))  

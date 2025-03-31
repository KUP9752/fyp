import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim

from rlbench.demo import Demo
from rlbench.backend.observation import Observation

from tqdm import tqdm as progress

from enum import Flag, auto

class CamType(Flag):
  WRIST = 0
  LEFT_SHOULDER = auto()
  RIGHT_SHOULDER = auto()


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
            demos: np.ndarray[Demo],
            epochs: int = 100,
            batch_size: int = 2,
            lr: float = 0.001,
            model_path: str = None
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = self.to(device)
    
    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr = lr)
    
    model.train()
    self.losses = [0 for _ in range(epochs)]
    for epoch in progress(range(epochs)):
      running_loss = 0
      ## picks one, currently only considering one demo
      obs_batch = np.random.choice(demos, replace = False) 
      
      match self.cam_type:
        case CamType.WRIST:
          inputs, labels = zip(
            *[(obs.wrist_rgb, np.append(obs.joint_velocities, 1.)) for obs in obs_batch]
          )
        case CamType.LEFT_SHOULDER:
          inputs, labels = zip(
            *[(obs.left_shoulder_rgb, np.append(obs.joint_velocities, 1.)) for obs in obs_batch]
          )
        case CamType.RIGHT_SHOULDER:
          inputs, labels = zip(
            *[(obs.right_shoulder_rgb, np.append(obs.joint_velocities, 1.)) for obs in obs_batch]
          )
        case _:
          raise ValueError("There are no other camtype options")
      
      inputs = torch.tensor(inputs, dtype = torch.float32)
      
      ## batch, 64, 64, 3  -> batch, 3, 64, 64
      inputs = torch.permute(inputs, (0, 3, 1, 2)) 
      
      labels = torch.tensor(labels, dtype = torch.float32)
      
      
      
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
    print(f"Done Training Policy on Demos") 
    
    if model_path:
      torch.save(self.state_dict(), f"{model_path}")

class Agent(object):

    def __init__(self, action_shape, cam_type = CamType.WRIST):
      self.action_shape = action_shape
      self.policy = Policy(action_shape, cam_type)

    def ingest(self, demos: list[Demo]):
      self.policy.train_policy(demos)
      
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
        
    def save_model(self, model_name: str):
      torch.save(self.policy.state_dict(), f'{model_name}.pth')
      print(f"Saved Model under '{model_name}.pth'")
      
      
    def load_model(self, model_name: str):
      self.policy.load_state_dict(torch.load(f'{model_name}.pth'))  

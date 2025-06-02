from typing import Literal, Optional

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
transforms.Normalize
from rlbench.demo import Demo

from tqdm import tqdm as progress
from lib.cam_type import CamType

from lib.utils import params_string

from modules.dataset.demo_obs_dataset import DemoObsDataset
from modules.dataset.demo_dataset import DemoDataset

## This is made for image sizes of 64x64 and now multi cam setups
class SimplePolicy(nn.Module):
  def __str__(self) -> str:
    return "simple-policy"
  
  def __repr__(self) -> str:
    return "SimplePolicy()"

  def __init__(self, action_shape: int, cam_type: CamType = CamType.WRIST):
    super(SimplePolicy, self).__init__()
    self.cam_type = cam_type
    print(f"[simple_policy] - Policy] Using {self.cam_type} as camera type")
    
    self.num_rgb_cams = 0
    if cam_type & CamType.WRIST:
      self.num_rgb_cams += 1
    if cam_type & CamType.LEFT_SHOULDER:
      self.num_rgb_cams += 1
    if cam_type & CamType.RIGHT_SHOULDER:
      self.num_rgb_cams += 1
    
    if self.num_rgb_cams == 0:
      raise ValueError("[simple_policy] - Policy] No cameras selected!")
    
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels=3 * self.num_rgb_cams, out_channels=32, kernel_size=3, stride=1, padding=0),
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
    self.flat_size = 2 * 2 * 128
    
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.flat_size, 200),
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
    return self.fc(feats), {} ##making all policies return action, (...) so I can have multiple outputs

  def _collate_demos(self, batch) -> tuple[torch.Tensor, torch.Tensor, dict]:
    ## batch: [(tensor, tensor)] for inputs, labels
    inputs, labels, loader_dict = zip(*batch) #unzip the tuple list

    ## NOTE: handle other dict entries as well

    ## concat on the batch axis, preserve order of input to label
    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0), {"proprio": None}

  def train_policy(self, 
    demos: list[Demo],
    epochs: int = 200,
    minibatch_size: int = 32, ## size of the observations currently being used
    lr: float = 0.01,
    data_label: Literal["joint_velocities", "joint_positions"] = "joint_velocities",
    shuffle_data = False, 
    shuffle_obs_in_demo = False,
    dataset_to_use = "obs",
    model_path: Optional[str] = None,
    lock_loader_seed: Optional[int] = None,
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Camera: {self.cam_type}")
    
    params_str = params_string(
      epochs = epochs,
      minibatch_size = minibatch_size, 
      lr = lr, 
      data_label = data_label,
      model_path = model_path, 
      shuffle_data = shuffle_data, 
      shuffle_obs_in_demo = shuffle_obs_in_demo,
      dataset_to_use = dataset_to_use,
      lock_loader_seed = lock_loader_seed,
      device = device

    )

    print(f"Training Params: {params_str}")
    
    
    model = self.to(device)
    # print(f"What is in the demos: {type(demos)} | {type(demos[0])}")
    
    loss_fn = nn.MSELoss()
    ## NOTE: suggested nn.Smooth1Loss()
    optimiser = optim.Adam(model.parameters(), lr = lr)
    
    ## 'cat' makes sure to return all the images fuxed together (batch_size, 3 * num_cam, W, H)
    ## TODO: make into DemoDatset add the data_label
    if dataset_to_use == "obs":
      dataset = DemoObsDataset(demos, self.cam_type, shuffle_obs=shuffle_obs_in_demo, get_type="cat")
      loader = DataLoader(
        dataset, 
        batch_size=minibatch_size, 
        shuffle=shuffle_data,
        generator=torch.manual_seed(lock_loader_seed) if lock_loader_seed else None
      ) ## shuffling makes it worse
    else:
      dataset = DemoDataset(
        demos,
        self.cam_type,
        get_type="cat",
        label_get = "joint_velocities"
      )
      loader = DataLoader(
        dataset, 
        batch_size=minibatch_size, 
        shuffle=shuffle_data, 
        collate_fn=self._collate_demos,
        generator=torch.manual_seed(lock_loader_seed) if lock_loader_seed else None
      )
       
      

    # print(f"Dataset Size: {len(dataset)}")
    
    model.train()
    self.losses = [0 for _ in range(epochs)]
    for epoch in progress(range(epochs)):
      running_loss = 0
      
      for inputs, labels, loader_dict in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimiser.zero_grad()
        pred_actions, _ = model(inputs)
        loss = loss_fn(pred_actions, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
        loss = running_loss / len(loader)
        self.losses[epoch] = loss
      
    print(f"Done Training Policy on {len(demos)} Demos") 
    
    if model_path:
      torch.save(self.state_dict(), f"{model_path}")


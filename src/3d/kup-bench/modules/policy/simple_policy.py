from typing import Optional

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
transforms.Normalize
from rlbench.demo import Demo

from tqdm import tqdm as progress
from lib.cam_type import CamType


from modules.demo_obs_dataset import DemoObsDataset

## This is made for image sizes of 64x64 and now multi cam setups
class SimplePolicy(nn.Module):
  def __init__(self, action_shape: int, cam_type: CamType = CamType.WRIST):
    super(SimplePolicy, self).__init__()
    self.cam_type = cam_type
    print(f"[simple_policy] - Policy] Using {self.cam_type} as camera type")
    
    num_cams = 0
    if cam_type & CamType.WRIST:
      num_cams += 1
    if cam_type & CamType.LEFT_SHOULDER:
      num_cams += 1
    if cam_type & CamType.RIGHT_SHOULDER:
      num_cams += 1
    
    if num_cams == 0:
      raise ValueError("[simple_policy] - Policy] No cameras selected!")
    
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
  

  def train_policy(self, 
            demos: list[Demo],
            epochs: int = 200,
            minibatch_size: int = 32, ## size of the observations currently being used
            lr: float = 0.01,
            shuffle_data = False, 
            shuffle_obs_in_demo = True,
            model_path: Optional[str] = None,
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Camera: {self.cam_type}")
    
    print(f"Training Params: \n\t{epochs = }, \n\t{minibatch_size = }, \n\t{lr = }, \n\t{model_path = }, \n\t{shuffle_data = },\n\t{shuffle_obs_in_demo = } \n\t{device}\n")
    
    
    model = self.to(device)
    # print(f"What is in the demos: {type(demos)} | {type(demos[0])}")
    
    loss_fn = nn.MSELoss()
    ## NOTE: suggested nn.Smooth1Loss()
    optimiser = optim.Adam(model.parameters(), lr = lr)
    
    ## 'cat' makes sure to return all the images fuxed together (batch_size, 3 * num_cam, W, H)
    dataset = DemoObsDataset(demos, self.cam_type, shuffle_obs=shuffle_obs_in_demo, get_type="cat")
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=shuffle_data) ## shuffling makes it worse
    # print(f"Dataset Size: {len(dataset)}")
    
    model.train()
    self.losses = [0 for _ in range(epochs)]
    for epoch in progress(range(epochs)):
      running_loss = 0
      
      for inputs, labels in loader:
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


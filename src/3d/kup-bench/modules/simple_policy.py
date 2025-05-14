from typing import Literal, Optional
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
transforms.Normalize
from rlbench.demo import Demo
from rlbench.backend.observation import Observation

from tqdm import tqdm as progress
from lib.cam_type import CamType

from modules.demo_obs_dataset import DemoObsDataset
from modules.demo_dataset import DemoDataset

from lib.utils import GRIPPER_CLOSE, GRIPPER_OPEN

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

class SimpleGraspPolicy(SimplePolicy):
  ## override
  def __init__(self, action_shape: int, cam_type: CamType = CamType.WRIST, grasp_thresh: float = 0.5):
    super().__init__(action_shape, cam_type)
    
    self.grasp_thresh = grasp_thresh
    
    # if cam_type & CamType.WRISTDE

    self.fc = None
    
    self.action_head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.flat_size, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 50),
      nn.ReLU(inplace=False),
      nn.Linear(50, action_shape - 1) ## predicts 8 - 1 dim action, no pose predication here
    )
    
    self.grasp_head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.flat_size, 128),
      nn.ReLU(inplace=False),
      nn.Linear(128, 64),
      nn.ReLU(inplace = False),
      nn.Linear(64, 1),
      # nn.Sigmoid() ## remove for raw logits, lets see that it predicts now
    )
    
  def forward(self, image):
    feats = self.conv(image)
    pose = self.action_head(feats)
    grasp = self.grasp_head(feats)
    
    action = torch.cat([pose, grasp], dim = 1) ## get (batch_size, 8)
    return action, {}


  ## this is used whent he "demo" options is selected for dataset, so we can catch the demos randomly but process in batch size
  def _collate_demos(self, batch):
    ## batch: [(tensor, tensor)] for inputs, labels
    inputs, labels = zip(*batch) #unzip the tuple list

    ## concat on the batch axis, preserve order of input to label
    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0)


  ## override
  def train_policy(self, 
    demos: list[Demo],
    epochs: int = 200,
    minibatch_size: int = 1, ## size of the observations currently being used
    lr: float = 0.01,
    shuffle_data = False, 
    shuffle_obs_in_demo = True,
    model_path: Optional[str] = None,
    lambda_grasp_loss: float = 1.,
    lock_loader_seed: Optional[int] = None, ## NOTE: disabled, not using
    dataset_to_use: Literal["obs", "demo"] = "obs"
    
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Camera: {self.cam_type}")
    
    print(f"Training Params: \n\t{epochs = }, \n\t{minibatch_size = }, \n\t{lr = }, \n\t{model_path = }, \n\t{shuffle_data = },\n\t{shuffle_obs_in_demo = },\n\t {dataset_to_use = }, \n\t{'seeded loader' if lock_loader_seed is not None else 'random loader'} \n\t{device}\n")
    
    
    model = self.to(device)
    # print(f"What is in the demos: {type(demos)} | {type(demos[0])}")
    
    ## 'cat' makes sure to return all the images fuxed together (batch_size, 3 * num_cam, W, H)
    if dataset_to_use == "obs":
      dataset = DemoObsDataset(demos, self.cam_type, shuffle_obs=shuffle_obs_in_demo, get_type="cat")
      loader = DataLoader(dataset, batch_size = minibatch_size, shuffle=shuffle_data)
    elif dataset_to_use == "demo":
      dataset = DemoDataset(demos, self.cam_type, get_type="cat")

      ## NOTE: shuffle_data here shuffles demos but preserver obs order
      if minibatch_size > len(demos):
        raise IndexError(f"[simple_policy - SimpleGraspPolicy - train_policy] Using a minibatch_size, {minibatch_size},  greated than given demos ({len(demos)}) is this correct?")
      loader = DataLoader(dataset, batch_size= minibatch_size, shuffle = shuffle_data, collate_fn=self._collate_demos) 
      ## NOTE: Ensure batch size is interms of demos now
    else: 
      raise ValueError(f"[simple_policy - SimpleGraspPolicy - train_policy] wrong dataset to use, '{dataset_to_use}' does not exist")


    # if lock_loader_seed is not None:
    #   loader = DataLoader(dataset,
    #     batch_size=minibatch_size,
    #     shuffle=shuffle_data,
    #     generator=torch.manual_seed(lock_loader_seed)
    #    ) ## shuffling makes it worse
    # else:
    #   loader = DataLoader(dataset,
    #     batch_size=minibatch_size,
    #     shuffle=shuffle_data
    #    ) 
    
    # grasp_labels = torch.tensor([labels[-1] for _, labels in dataset], dtype = torch.float32)
    
    # num_pos = (grasp_labels == 1).sum()
    # num_neg = (grasp_labels == 0).sum()
    # if num_pos > 0:
    #   class_weight = num_neg / num_pos
    # else:
    #   class_weight = torch.tensor(1.)
    
    # print(f"{num_pos = }, {num_neg = }")
    run_pose = []
    run_grasp = []
    
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=None)
    # bce_loss =  nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    optimiser = optim.Adam(model.parameters(), lr = lr)
    
    model.train()
    self.losses = [0 for _ in range(epochs)]
    for epoch in progress(range(epochs)):
      total_pose_loss, total_grasp_loss = 0., 0.
      
      for inputs, labels in loader:

        if dataset_to_use == "demo":
          inputs, labels = inputs.squeeze(), labels.squeeze()

        inputs, labels = inputs.to(device), labels.to(device)
        # print(f"{inputs.shape =}")
        # print(f"{labels.shape =}")
        
        optimiser.zero_grad()
        
        
        pred_actions, _ = model(inputs)
        # print(f"{pred_actions.shape = }")
        
        ## [:, x] to preserve the batch shape (batch_size, X)
        pose_loss = mse_loss(pred_actions[:, :-1], labels[:, :-1]) ## only the pose not he gripper action
        grasp_loss = bce_loss(pred_actions[:, -1], labels[:, -1])
        
        loss = pose_loss +  lambda_grasp_loss * grasp_loss
        
        loss.backward()
        optimiser.step()
        total_pose_loss += pose_loss.item()
        total_grasp_loss += grasp_loss.item()
        loss = (total_pose_loss + lambda_grasp_loss * total_grasp_loss) / len(loader)
        self.losses[epoch] = loss
        N = len(loader)
        # print(f"Epoch {epoch}: PoseLoss={total_pose_loss/N:.4f}, GraspLoss={total_grasp_loss/N:.4f}")
      
    print(f"Done Training Policy on {len(demos)} Demos") 
    
    if model_path:
      torch.save(self.state_dict(), f"{model_path}")

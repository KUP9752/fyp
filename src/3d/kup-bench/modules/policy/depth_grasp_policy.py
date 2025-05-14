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

from modules.cnns.cnn_encoder import CNNEncoder
from modules.policy.simple_grasp_policy import SimpleGraspPolicy



from modules.demo_obs_dataset import DemoObsDataset
from modules.demo_dataset import DemoDataset

## Making a separate class/file here for this differnet than `SimpleGrasp` just so it is more convenient to tweak and experiment with


class DepthGraspPolicy(SimpleGraspPolicy): 
  def __init__(self,
    action_shape: int, 
    config: Literal["depth_ch", "depth_feats", "all_sep"],
    cam_type = CamType.WRIST,
    grasp_thresh = 0.5,
  ):
    super().__init__(action_shape, cam_type, grasp_thresh)
    # super(DepthGraspPolicy, self).__init__() ##if inherining nn.Module
    # self.grasp_thresh = grasp_thresh

    match config:
      case "depth_ch":
        num_ch = 0
        if cam_type & CamType.WRIST:
          num_ch += 3
        if cam_type & CamType.LEFT_SHOULDER:
          num_ch += 3
        if cam_type & CamType.RIGHT_SHOULDER:
          num_ch += 3
        if cam_type & CamType.WRIST_DEPTH:
          num_ch += 1
        
        
        self.conv = CNNEncoder(in_channels = num_ch) ## 3 * num_cams + 1 * depth_cam
        ## should need nothing else?
      case "depth_feats":
        self._1 = 1
      case "all_sep":
        self._2 = 1
      case _: 
        raise ValueError(f"[depth_grasp_policy - DepthGraspPolicy] config '{config}' is unknown!")
    self.config = config

  def forward(self, image):
    match self.config:
      case "depth_ch":
        return super().forward(image)
      case "depth_feats":
        raise NotImplementedError(f"[depth_grasp_policy - forward]")
      case "all_sep":
        raise NotImplementedError(f"[depth_grasp_policy - forward]")
      case _: 
        raise ValueError(f"[depth_grasp_policy - forward] config '{self.config}' is unknown!")

  def train_policy(self,
    demos: list[Demo],
    epochs: int = 200,
    minibatch_size: int = 1,
    lr: float = 0.01,
    shuffle_data=True,
    shuffle_obs_in_demo=False,
    model_path: Optional[str] = None,
    lambda_grasp_loss: float = 1, 
    lock_loader_seed: Optional[int] = None,
    dataset_to_use: Literal['obs'] | Literal['demo'] = "demo",
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params_str = params_string(
      epochs = epochs, 
      model_path = model_path,
      shuffle_data = shuffle_data,
      minibatch_size = minibatch_size,
      lr = lr, 
      shuffle_obs_in_demo = f'{shuffle_obs_in_demo} [not being used!]',
      lambda_grasp_loss = lambda_grasp_loss,
      dataset_to_use = f'{dataset_to_use} [not being used!]',
      lock_loader_seed = lock_loader_seed,
      device = device
    )
    
    print(f"Training params: {params_str}")

    if dataset_to_use != "demo":
      raise NameError(f"[depth_grasp_policy - train_policy] Not allowing the other dataset anymore only allow 'demo'")
    
    shuffle_obs_in_demo = None


    model = self.to(device)

    dataset = DemoDataset(demos, cam_type=self.cam_type, get_type = "cat")
    loader = DataLoader(
      dataset,
      batch_size=minibatch_size,
      shuffle=shuffle_data,
      collate_fn=self._collate_demos,
      generator=torch.manual_seed(lock_loader_seed) if lock_loader_seed else None
    )
    
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=None)
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
        
        loss = pose_loss + lambda_grasp_loss * grasp_loss
        
        loss.backward()
        optimiser.step()

        total_pose_loss += pose_loss.item()
        total_grasp_loss += grasp_loss.item()

        loss = (total_pose_loss + lambda_grasp_loss * total_grasp_loss) / len(loader)

        self.losses[epoch] = loss
        # N = len(loader)
        # print(f"Epoch {epoch}: PoseLoss={total_pose_loss/N:.4f}, GraspLoss={total_grasp_loss/N:.4f}")
      
    print(f"Done Training Policy on {len(demos)} Demos") 
    

    
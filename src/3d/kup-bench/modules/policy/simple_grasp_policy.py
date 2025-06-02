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

from modules.joint_pos_encoder import JointPosEncoder
from modules.policy.simple_policy import SimplePolicy

from lib.utils import params_string

from modules.dataset.demo_obs_dataset import DemoObsDataset
from modules.dataset.demo_dataset import DemoDataset

class SimpleGraspPolicy(SimplePolicy):
  def __str__(self):
    return f"simple_grasp_policy-use_proprio:{self.use_proprio}-proprio_opts:{self.proprio_opts}"
  
  def __repr__(self):
    return f"SimpleGraspPolicy(use_proprio={self.use_proprio}, proprio_opts={self.proprio_opts})"
  
  ## override
  def __init__(self, 
    action_shape: int, 
    cam_type: CamType = CamType.WRIST,
    use_proprio: bool = False,
    proprio_opts: dict = {}
  ):
    super().__init__(action_shape, cam_type)
    
    self.use_proprio = use_proprio
    self.proprio_opts = proprio_opts

    self.fc = None
    self.final_feat_size = self.flat_size
    if use_proprio:
      self.jpos_feats = JointPosEncoder(**proprio_opts) if proprio_opts else JointPosEncoder()
      self.final_feat_size += self.jpos_feats.output_size

    
    self.flatten = nn.Flatten()

    self.action_head = nn.Sequential(
      nn.Linear(self.final_feat_size, 200),
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
      nn.Linear(self.final_feat_size, 128),
      nn.ReLU(inplace=False),
      nn.Linear(128, 64),
      nn.ReLU(inplace = False),
      nn.Linear(64, 1),
      # nn.Sigmoid() ## remove for raw logits, lets see that it predicts now
    )
  

  def _feats_to_action(self, feats, proprio = None) -> tuple[torch.Tensor, dict]:
    if proprio is None and self.use_proprio:
      raise RuntimeError(f"[simple_grasp_policy - (feats_to_action)] Expecting proprio data but none given!")
    ret_dict = {}
    feats = self.flatten(feats)

    if proprio is not None:
      jfeats, _ = self.jpos_feats(proprio)
      feats  = torch.cat([feats, jfeats], dim = -1) ## cat on feature dimension
      ret_dict = {"proprio_feats": jfeats}
    
    pose = self.action_head(feats)
    grasp = self.grasp_head(feats)
    
    action = torch.cat([pose, grasp], dim = 1) ## get (batch_size, 8)
    return action, ret_dict
  
  def forward(self, image, proprio = None) -> tuple[torch.Tensor, dict]:
    if proprio is None and self.use_proprio:
      raise RuntimeError(f"[simple_grasp_policy - forward] Expecting proprio data but none given!")
    
    feats = self.conv(image)
    return self._feats_to_action(feats, proprio)


  ## this is used whent he "demo" options is selected for dataset, so we can catch the demos randomly but process in batch size
  def _collate_demos(self, batch)-> tuple[torch.Tensor, torch.Tensor, dict]:
    ## batch: [(tensor, tensor)] for inputs, labels
    inputs, labels, loader_dict = zip(*batch) #unzip the tuple list

    ## NOTE: handle other dict entries as well
    proprio = None
    if self.use_proprio:
      proprio = [d["proprio"] for d in loader_dict]## should always exist, might be empty
      proprio = torch.cat(proprio, dim=0)

    ## concat on the batch axis, preserve order of input to label
    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0), {"proprio": proprio}


  ## override
  def train_policy(self, 
    demos: list[Demo],
    epochs: int = 200,
    minibatch_size: int = 1, ## size of the observations currently being used
    lr: float = 0.01,
    data_label: Literal["joint_velocities", "joint_positions"] = "joint_velocities",
    shuffle_data = False, 
    shuffle_obs_in_demo = False,
    model_path: Optional[str] = None,
    lambda_grasp_loss: float = 1.,
    lock_loader_seed: Optional[int] = None, ## NOTE: disabled, not using
    dataset_to_use: Literal["obs", "demo"] = "obs"
    
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Camera: {self.cam_type}")
    
    params_str = params_string(
      epochs = epochs,
      minibatch_size = minibatch_size, 
      lr = lr, 
      data_label = data_label,
      model_path = model_path, 
      shffle_data = shuffle_data, 
      shuffle_obs_in_demo = shuffle_obs_in_demo,
      lock_loader_seed = lock_loader_seed, 
      device = device

    )

    print(f"Training Params: {params_str}")

    model = self.to(device)
    # print(f"What is in the demos: {type(demos)} | {type(demos[0])}")
    
    ## 'cat' makes sure to return all the images fuxed together (batch_size, 3 * num_cam, W, H)
    if dataset_to_use == "obs":
      dataset = DemoObsDataset(demos, self.cam_type, shuffle_obs=shuffle_obs_in_demo, get_type="cat")
      loader = DataLoader(dataset, batch_size = minibatch_size, shuffle=shuffle_data)
    elif dataset_to_use == "demo":
      dataset = DemoDataset(demos, self.cam_type, get_type="cat", label_get = data_label,  use_proprio=self.use_proprio)

      ## NOTE: shuffle_data here shuffles demos but preserver obs order
      if minibatch_size > len(demos):
        raise IndexError(f"[simple_grasp_policy - SimpleGraspPolicy - train_policy] Using a minibatch_size, {minibatch_size},  greated than given demos ({len(demos)}) is this correct?")
      loader = DataLoader(dataset, batch_size= minibatch_size, shuffle = shuffle_data, collate_fn=self._collate_demos) 
      ## NOTE: Ensure batch size is interms of demos now
    else: 
      raise ValueError(f"[simple_grasp_policy - SimpleGraspPolicy - train_policy] wrong dataset to use, '{dataset_to_use}' does not exist")


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
      
      for inputs, labels, loader_dict in loader:

        if dataset_to_use == "demo":
          inputs, labels = inputs.squeeze(), labels.squeeze()

        inputs, labels = inputs.to(device), labels.to(device)

        proprio_inputs = None
        if self.use_proprio:
          proprio_inputs = loader_dict["proprio"].squeeze()
          proprio_inputs = proprio_inputs.to(device)

        # print(f"{inputs.shape =}")
        # print(f"{labels.shape =}")
        
        optimiser.zero_grad()
        
        
        pred_actions, _ = model(inputs, proprio = proprio_inputs)
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

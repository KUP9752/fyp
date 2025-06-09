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
from modules.cnns.multi_cam_cnn import MultiCamCnn
from modules.cnns.vit_encoder import MultiViewEncoder

from modules.film_net import FilmModulator
from modules.cnns.cross_attn_feats import CrossAttentionFeatures

from modules.joint_pos_encoder import JointPosEncoder
from modules.cnns.fusing_encoder import FusingEncoder

from modules.dataset.demo_dataset import DemoDataset
from lib.fuse_config import FuseConfig

  
## Making a separate class/file here for this differnet than `SimpleGrasp` just so it is more convenient to tweak and experiment with
class FusingPolicy(nn.Module): 
  def __str__(self):
    return f"fusing_policy-fuse_config:{self.fuse_config}-is_grasp:{self.is_grasp}-use_proprio:{self.use_proprio}-fusing_opts:{self.fusing_opts}-proprio_opts:{self.proprio_opts}"
  
  def __repr__(self):
    return f"FusingPolicy(fuse_config={self.fuse_config}, is_grasp={self.is_grasp}, use_proprio={self.use_proprio}, config={self.config}, fusing_opts={self.fusing_opts}, proprio_opts={self.proprio_opts})"
  
  def __init__(self,
    action_shape: int, 
    cam_type: CamType,
    fuse_config: FuseConfig,
    is_grasp: bool,
    fusing_opts: dict = {},
    use_proprio: bool = False, 
    proprio_opts: dict = {},
  ):
    ## keeps all defaults that are not overriden in opts
    # self.opts = self.default_opts 
    super(FusingPolicy, self).__init__()
    self.action_shape = action_shape
    self.cam_type = cam_type
    self.fuse_config = fuse_config
    self.is_grasp = is_grasp
    self.use_proprio = use_proprio
    self.fusing_opts = fusing_opts
    self.proprio_opts = proprio_opts

    print(f"Policy Fuse Config: {fuse_config}")
    
    self.feats = FusingEncoder(
      cam_type=self.cam_type, 
      config = fuse_config, 
      opts = fusing_opts ## use defaults if ont he other end if needed
    )
    self.final_feat_size = self.feats.final_feat_size

    if self.use_proprio:
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
      ## NOTE: do full regression here if not grasp
      nn.Linear(50, action_shape - 1) if self.is_grasp else nn.Linear(50, action_shape)
    )

    ## conditional on setting
    self.grasp_head = nn.Sequential(
      nn.Linear(self.final_feat_size, 128),
      nn.ReLU(inplace=False),
      nn.Linear(128, 64),
      nn.ReLU(inplace = False),
      nn.Linear(64, 1),
    ) if self.is_grasp else None
  
    
  def _feats_to_action(self, feats, proprio = None) -> torch.Tensor:
    feats = self.flatten(feats)
    if self.use_proprio:
      jfeats, _ = self.jpos_feats(proprio)
      feats = torch.cat([feats, jfeats], dim = -1) ## cat on feature dimension

    pose = self.action_head(feats) ## will predict size = 8 if not grasp so this is action
    if self.is_grasp:  
      grasp = self.grasp_head(feats) #type: ignore (//NOTE: this is handled)
      action = torch.cat([pose, grasp], dim = 1) ## (B, 8)
    else:
      action = pose

    return action 
  
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
    
    
    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0), {
      "proprio": proprio, 
      "demo_lengths": torch.LongTensor([len(i) for i in inputs]) ## needed for the lambda k thing
    }
  
  def forward(self, image, proprio=None) -> tuple[torch.Tensor, dict]:
    if proprio is None and self.use_proprio:
      raise RuntimeError(f"[fusing_policy - forward] Expecting proprio data but none given!")

    feats, feats_dict = self.feats(image)
    return self._feats_to_action(feats, proprio), feats_dict | {
      "can do any kind of additions here now": None
    }

  def train_policy(self, 
    demos: list[Demo],
    epochs: int = 200,
    minibatch_size: int = 1, ## size of the observations currently being used
    lr: float = 0.01,
    data_label: Literal["joint_velocities", "joint_positions"] = "joint_velocities",
    shuffle_data = True, 
    shuffle_obs_in_demo = False,
    lr_eta_min = 1e-4,
    model_path: Optional[str] = None,
    lambda_grasp_loss: float = 1.,
    lock_loader_seed: Optional[int] = None, 
    dataset_to_use: Literal["obs", "demo"] = "demo",
    # last_k_grasp_mask: Optional[int] = None
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params_str = params_string(
      epochs = epochs, 
      model_path = model_path,
      shuffle_data = shuffle_data,
      minibatch_size = minibatch_size,
      lr = lr, 
      data_label = data_label, 
      shuffle_obs_in_demo = f'{shuffle_obs_in_demo} [not being used!]',
      lambda_grasp_loss = lambda_grasp_loss,
      dataset_to_use = f'{dataset_to_use} [not being used!]',
      lock_loader_seed = lock_loader_seed,
      lr_eta_min = lr_eta_min,
      # last_k_grasp_mask = last_k_grasp_mask,
      device = device
    )
    
    print(f"Training params: {params_str}")

    if dataset_to_use != "demo":
      raise NameError(f"[fusing_policy - train_policy] Not allowing the other dataset anymore only allow 'demo'")
    
    shuffle_obs_in_demo = None

    model = self.to(device)

    dataset = DemoDataset(demos,
      cam_type=self.cam_type,
      get_type = "cat",
      use_proprio = self.use_proprio,
      label_get=data_label
    )

    loader = DataLoader(
      dataset,
      batch_size=minibatch_size,
      shuffle=shuffle_data,
      collate_fn=self._collate_demos, ## uses parents collater, see `simple_grasp_policy._collate_demos`
      generator=torch.manual_seed(lock_loader_seed) if lock_loader_seed else None
    )

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=None)
    mse_loss = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max= epochs, eta_min=lr_eta_min )
    self.action_losses = []
    self.grasp_losses = []

    model.train()
    for epoch in progress(range(epochs)):
      running_pose_loss, running_grasp_loss = 0., 0.
      
      for inputs, labels, loader_dict in loader:
        if dataset_to_use == "demo":
          inputs, labels = inputs.squeeze(), labels.squeeze()

        proprio_inputs = None
        if self.use_proprio:
          proprio_inputs = loader_dict["proprio"].squeeze()
          proprio_inputs = proprio_inputs.to(device)

        inputs, labels = inputs.to(device), labels.to(device)
        
        optimiser.zero_grad()
        pred_actions, _ = model(inputs, proprio = proprio_inputs)

        ## [:, x] to preserve the batch shape (batch_size, X)
        pose_loss = mse_loss(pred_actions[:, :-1], labels[:, :-1]) ## only the pose not the gripper action
        grasp_loss = bce_loss(pred_actions[:, -1], labels[:, -1])

        loss = pose_loss + lambda_grasp_loss * grasp_loss
        
        loss.backward()
        optimiser.step()
        scheduler.step()

        running_pose_loss += pose_loss.item()
        running_grasp_loss += grasp_loss.item()

      loss = (running_pose_loss + lambda_grasp_loss * running_grasp_loss) / len(loader)

      self.grasp_losses.append(running_grasp_loss / len(loader))
      self.action_losses.append(running_pose_loss / len(loader))
      
    print(f"Done Training Policy on {len(demos)} Demos") 
    

    
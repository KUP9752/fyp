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

from modules.cnns.resnet import ResNetEncoder
from modules.policy.simple_grasp_policy import SimpleGraspPolicy

from torchvision.models.resnet import Bottleneck, BasicBlock

from modules.dataset.demo_obs_dataset import DemoObsDataset
from modules.dataset.demo_dataset import DemoDataset

## Making a separate class/file here for this differnet than `SimpleGrasp` just so it is more convenient to tweak and experiment with

class ResNetGraspPolicy(nn.Module): 

  def __str__(self):
    return f"resnet_grasp_policy-config:{self.config}-opts:{self.opts}"
  
  def __repr__(self):
    return f"ResNetGraspPolicy(config={self.config}, opts={self.opts})"
  
  def __init__(self,
    action_shape: int, 
    config: Literal["depth_ch", "depth_feats", "all_sep"],
    cam_type = CamType.WRIST,
    opts: dict = {
      "gated_fuse": True,
      "resnet_name": "not a valid resnet name",
      "kernel_size": 3 ## resnet default is 7
    } ## set all defaults to none so I don't have to try/catch everytime
  ):
    super(ResNetGraspPolicy, self).__init__()
    self.action_shape = action_shape
    self.cam_type = cam_type
    self.opts = opts
    print(f"{opts = }")
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
        
        
        self.conv = ResNetEncoder(in_channels = num_ch, kernel_size= opts["kernel_size"], resnet_name=opts["resnet_name"]) ## 3 * num_cams + 1 * depth_cam
        self.resnet_out_size = self._get_resnet_out_size(self.conv.resnet_name)
        self.flat_size = self.resnet_out_size * 3 * 3
      case "depth_feats":
        ## separate conv for the images w + ls + rs (like SimpleGraspPolicy)
        ## but run depth through its own encoder, then undfuse with a separate MLP (or integrate into the later heads?)
        if not (self.cam_type & CamType.WRIST_DEPTH):
          raise ValueError(f"[resnet_grasp_policy] Policy cam_type does not include '{CamType.WRIST_DEPTH}' -> current: '{self.cam_type}'")
        num_ch = 0
        if cam_type & CamType.WRIST:
          num_ch += 3
        if cam_type & CamType.LEFT_SHOULDER:
          num_ch += 3
        if cam_type & CamType.RIGHT_SHOULDER:
          num_ch += 3
        ## Depth is not included, only the rgbs
        
        self.conv = ResNetEncoder(in_channels = num_ch, kernel_size= opts["kernel_size"], resnet_name=opts["resnet_name"]) ## 3 * num_cams + 1 * depth_cam
        self.depth_conv = ResNetEncoder(in_channels=1, kernel_size= opts["kernel_size"], resnet_name=opts["resnet_name"]) ## depth has one channel
        self.resnet_out_size = self._get_resnet_out_size(self.conv.resnet_name)
        self.flat_size = self.resnet_out_size * 3 * 3 ## NOTE: these 3s are due to the kernelsize parameter, kernel in resnet = 7 by default, 3 is better for my smaller resolution cammeras
        ## don't want it too deep, will feed into the next MLPs for prediction action and gripper
        ## TODO: multiscale fusion
        if self.opts["gated_fuse"]:
          ## gate: 2 * (B, 512/2048, 2, 2) => (B, 512, 2, 2)
          self.gate = nn.Sequential(
            nn.Conv2d(self.resnet_out_size * 2, self.resnet_out_size, kernel_size=1), 
            nn.Sigmoid() ## gating
          )
        else:
          self.fuser = nn.Sequential(
            # nn.Flatten(), ## already flatttened in `forward()` call
            nn.Linear(self.flat_size * 2, self.flat_size * 2),
            nn.BatchNorm1d(self.flat_size * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(self.flat_size * 2, self.flat_size) ## same output to fit other MLPs later
          )

      case "all_sep":
        self._2 = 1
      case _: 
        raise ValueError(f"[resnet_grasp_policy - DepthGraspPolicy] config '{config}' is unknown!")

    self.move_head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.flat_size, 1028), ## flat size is 512/2048 * 4
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(1028, 256),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(256, 64),
      nn.ReLU(inplace=False),
      nn.Linear(64, action_shape - 1) ## predicts 8 - 1 dim action, no pose predication here
    )

    self.grasp_head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.flat_size, 256), ## flat size is 512/2048 * 4
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      ## NOTE: I think this was too complicated, too deep, make it simpler like the `simple_grasp_policy` => works better
      # nn.Linear(1028, 256),
      # nn.ReLU(inplace=False),
      # nn.Dropout(0.2),
      nn.Linear(256, 64),
      nn.ReLU(inplace=False),
      nn.Linear(64, 1) 
    )

    self.config = config
    
    self.opts = opts
    self.opts["resnet_name"] = self.conv.resnet_name

  def _feats_to_action(self, feats) -> torch.Tensor:
    pose = self.move_head(feats)
    grasp = self.grasp_head(feats)
    
    action = torch.cat([pose, grasp], dim = 1) ## get (batch_size, 8)
    return action

  def forward(self, image) -> tuple[torch.Tensor, dict]:
    ## image: shape = (batch_size, chs, w, h) where chs = 3 * (given cams) + 1 (depth)
    ## so depth is always the final dimension (easier to do it this way for now, might change later)
    match self.config:
      case "depth_ch":
        return self._feats_to_action(self.conv(image)), {"opts": self.opts}
      case "depth_feats":
        if not (self.cam_type & CamType.WRIST_DEPTH):
          raise ValueError(f"[resnet_grasp_policy] Policy cam_type does not include '{CamType.WRIST_DEPTH}' -> current: '{self.cam_type}'")
        
        images = image[:, :-1, :, :] ## take all rgb cams
        ## take wrist depth //NOTE: only depth cam currently
        depth = image[:, -1, :, :].unsqueeze(dim=1) # for (B, w, h) -> (B, 1, w, h)
        
        ## feats have shape (B, 512/2048, 2, 2)
        ims_feats: torch.Tensor = self.conv(images) 
        depth_feats: torch.Tensor = self.depth_conv(depth)

        if self.opts["gated_fuse"]:
          cated = torch.cat([ims_feats, depth_feats], dim = 1) # (B, 1024/4096, 2, 2)
          gate = self.gate(cated) 
          fused_feats = gate * ims_feats + (1 - gate) * depth_feats
        else:
          ##  flatten them before concat -> (B, 512)
          ims_feats = ims_feats.view(ims_feats.shape[0], -1)
          depth_feats = depth_feats.view(depth_feats.shape[0], -1)
          cated = torch.cat([ims_feats, depth_feats], dim = 1) 
          ## cated: (B, self.feat_size * 2) -> (B, 1024) 
          fused_feats = self.fuser(cated) # (B, 512)

        return self._feats_to_action(fused_feats), {"opts": self.opts}
        ## TODO: depending on how this works, we can add other losses (similarity matrtix between rgb wrist and depth wrist?? somehow incorporate?)
        ## gating? use the fusion conv as a wrighting mevchanism -> `fused_feat = fuser(x) * rgb_feats + (1 - fuser(x)) * depth_feats`, simple attention to what the fuser thinks is important
        ## multiscale fusion?? -> merge at differnet levels, within the conv? conv -> merge -> conv -> merge etc?
        ## contrasive learning? compare the wrist_rgb and d as they go down the network
      case "all_sep":
        ## TODO: There can be smarter ways to fuse these?? maybe dynamically weight what cam to use, is this for 'all_sep'??
        raise NotImplementedError(f"[resnet_grasp_policy - forward]")
      case _: 
        raise ValueError(f"[resnet_grasp_policy - forward] config '{self.config}' is unknown!")
      
  def _get_resnet_out_size(self, name: str) -> int:
    # match self.opts["resnet_name"]:
    match name:
      case "resnet18" | "resnet34":
        return 512
      case "resnet50":
        return 2048
      case _: 
        raise AttributeError(f"[resnet_grasp_policy - _pick_flat_size] unexpected block '{block}'")
      
  def _collate_demos(self, batch):
    ## batch: [(tensor, tensor)] for inputs, labels
    inputs, labels = zip(*batch) #unzip the tuple list

    ## concat on the batch axis, preserve order of input to label
    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0)
  
  def train_policy(self,
    demos: list[Demo],
    epochs: int = 200,
    minibatch_size: int = 1,
    lr: float = 0.01,
    data_label: Literal["joint_velocities", "joint_positions"] = "joint_velocities",
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
      data_label = data_label,
      shuffle_obs_in_demo = f'{shuffle_obs_in_demo} [not being used!]',
      lambda_grasp_loss = lambda_grasp_loss,
      dataset_to_use = f'{dataset_to_use} [not being used!]',
      lock_loader_seed = lock_loader_seed,
      device = device
    )
    
    print(f"Training params: {params_str}")

    if dataset_to_use != "demo":
      raise NameError(f"[resnet_grasp_policy - train_policy] Not allowing the other dataset anymore only allow 'demo'")
    
    shuffle_obs_in_demo = None


    model = self.to(device)

    dataset = DemoDataset(
      demos, 
      cam_type=self.cam_type, 
      get_type = "cat",
      label_get=data_label
    )

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
      
      for inputs, labels, loader_dict in loader:
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
    

    
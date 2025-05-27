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
from modules.cnns.cross_attn_feats import CrossAttentionFeatures
from modules.policy.simple_grasp_policy import SimpleGraspPolicy
from modules.joint_pos_encoder import JointPosEncoder

from modules.dataset.demo_obs_dataset import DemoObsDataset
from modules.dataset.demo_dataset import DemoDataset

## Making a separate class/file here for this differnet than `SimpleGrasp` just so it is more convenient to tweak and experiment with
class DepthGraspPolicy(SimpleGraspPolicy): 
  default_opts = {
    "gated_fuse": True, 
    "attn_num_heads": 8,
    "attn_deep_fuse": True, ## works better
    "use_proprio": False, ## adding joint_positions
    "proprio_opts": {} ## dict of kwargs
  }

  def __init__(self,
    action_shape: int, 
    config: Literal["depth_ch", "depth_feats", "attn"],
    cam_type = CamType.WRIST,
    grasp_thresh = 0.5,
    opts: dict = {} ## set all defaults to none so I don't have to try/catch everytime
    ## "gated_fuse" set to 'True' because it works well
  ):
    self.opts = self.default_opts | opts
    super().__init__(
      action_shape,
      cam_type,
      grasp_thresh,
      use_proprio=self.opts["use_proprio"],
      proprio_opts=self.opts["proprio_opts"]
    )
    # super(DepthGraspPolicy, self).__init__() ##if inherining nn.Module
    # self.grasp_thresh = grasp_thresh
    
    ## keeps all defaults that are not overriden in opts

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
        ## separate conv for the images w + ls + rs (like SimpleGraspPolicy)
        ## but run depth through its own encoder, then undfuse with a separate MLP (or integrate into the later heads?)
        if not (self.cam_type & CamType.WRIST_DEPTH):
          raise ValueError(f"[depth_grasp_policy] Policy cam_type does not include '{CamType.WRIST_DEPTH}' -> current: '{self.cam_type}'")
        
        self.rgb_enc = self.conv
        self.depth_enc = CNNEncoder(in_channels=1) ## depth has one channel
        ## don't want it too deep, will feed into the next MLPs for prediction action and gripper
        ## TODO: multiscale fusion
        if self.opts["gated_fuse"]:
          ## gate: 2 * (B, 128, 2, 2) => (B, 128, 2, 2)
          self.gate = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1), 
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
      case "attn":
        if not (self.cam_type & CamType.WRIST_DEPTH):
          raise ValueError(f"[depth_grasp_policy] 'attn': Policy cam_type does not include '{CamType.WRIST_DEPTH}' -> current: '{self.cam_type}'")
        
        if self.opts["attn_num_heads"] is None or not isinstance(self.opts["attn_num_heads"], int):
          raise ValueError(f"[depth_grasp_policy - (DepthGraspPolicy)] 'attn_num_heads' in 'opts' must be an integer")

        self.attn_feats = CrossAttentionFeatures(
          rgb_channels=self.num_rgb_cams * 3,
          embed_size = self.flat_size // 4,
          feat_size=self.flat_size,
          attn_num_heads=self.opts["attn_num_heads"],
          is_deep_fuse= self.opts["attn_deep_fuse"]
        )

        # this can then be fed into the MLPs
      case _: 
        raise ValueError(f"[depth_grasp_policy - (DepthGraspPolicy)] config '{config}' is unknown!")
    self.config = config
    
    ## add proprioception data



  def __str__(self):
    return f"depth_grasp_policy-config:{self.config}-opts:{self.opts}"
  
  def __repr__(self):
    return f"DepthGraspPolicy(config={self.config}, opts={self.opts})"
  
  def forward(self, image, proprio=None) -> tuple[torch.Tensor, dict]:
    if proprio is None and self.opts["use_proprio"]:
      raise RuntimeError(f"[depth_grasp_policy - forward] Expecting proprio data but none given!")
    
    ret_dict = {}
    ## image: shape = (batch_size, chs, w, h) where chs = 3 * (given cams) + 1 (depth)
    ## so depth is always the final dimension (easier to do it this way for now, might change later)
    if self.config in ["depth_ch"]:
      return super().forward(image)

    if self.config in ["depth_feats", "attn"]:
      if not (self.cam_type & CamType.WRIST_DEPTH):
        raise ValueError(f"[depth_grasp_policy - forward] 'depth_feats': Policy cam_type does not include '{CamType.WRIST_DEPTH}' -> current: '{self.cam_type}'")
      
      images = image[:, :-1, :, :] ## take all rgb cams
      ## take wrist depth //NOTE: only depth cam currently
      depth = image[:, -1, :, :].unsqueeze(dim=1) # for (B, w, h) -> (B, 1, w, h)

      match self.config:
        case "depth_feats":
          rgb_feats: torch.Tensor = self.rgb_enc(images) 
          depth_feats: torch.Tensor = self.depth_enc(depth)
          ## feats have shape (B, 128, 2, 2)
          if self.opts["gated_fuse"]:
            cated = torch.cat([rgb_feats, depth_feats], dim = 1) # (B, 128, 2, 2)
            gate = self.gate(cated) 
            fused_feats = gate * rgb_feats + (1 - gate) * depth_feats
          else:
            ##  flatten them before concat -> (B, 512)
            rgb_feats = rgb_feats.view(rgb_feats.shape[0], -1)
            depth_feats = depth_feats.view(depth_feats.shape[0], -1)
            cated = torch.cat([rgb_feats, depth_feats], dim = 1) 
            ## cated: (B, self.feat_size * 2) -> (B, 1024) 
            fused_feats = self.fuser(cated) # (B, 512)
            
          ## TODO: depending on how this works, we can add other losses (similarity matrtix between rgb wrist and depth wrist?? somehow incorporate?)
          ## gating? use the fusion conv as a wrighting mevchanism -> `fused_feat = fuser(x) * rgb_feats + (1 - fuser(x)) * depth_feats`, simple attention to what the fuser thinks is important
          ## multiscale fusion?? -> merge at differnet levels, within the conv? conv -> merge -> conv -> merge etc?
          ## contrasive learning? compare the wrist_rgb and d as they go down the network
        case "attn":
          fused_feats, ret_dict = self.attn_feats(images, depth)

      action, feats_ret = self._feats_to_action(fused_feats, proprio) # type: ignore - cannot be unbound
      return action, ret_dict | feats_ret | { "opts": self.opts }

        
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

    dataset = DemoDataset(demos,
      cam_type=self.cam_type,
      get_type = "cat",
      use_proprio = self.use_proprio
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

    model.train()
    self.losses = [0 for _ in range(epochs)]
    for epoch in progress(range(epochs)):
      total_pose_loss, total_grasp_loss = 0., 0.
      
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
        pose_loss = mse_loss(pred_actions[:, :-1], labels[:, :-1]) ## only the pose not he gripper action
        grasp_loss = bce_loss(pred_actions[:, -1], labels[:, -1])
        
        loss = pose_loss + lambda_grasp_loss * grasp_loss
        
        loss.backward()
        optimiser.step()

        total_pose_loss += pose_loss.item()
        total_grasp_loss += grasp_loss.item()

        loss = (total_pose_loss + lambda_grasp_loss * total_grasp_loss) / len(loader)

        self.losses[epoch] = loss
      
    print(f"Done Training Policy on {len(demos)} Demos") 
    

    
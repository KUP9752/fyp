from typing import Literal
from modules.simple_policy import SimplePolicy
from modules.cam_attention_policy import CamAttentionPolicy

from rlbench.demo import Demo
from rlbench.backend.observation import Observation

from modules.cam_type import CamType

import torch

class Agent(object):

    def __init__(self,
      action_shape,
      policy: Literal["simple", "cam_attention"],
      cam_type = CamType.WRIST
    ):
      self.cam_type = cam_type
      self.action_shape = action_shape
      match policy:
        case "simple":
          self.policy = SimplePolicy(action_shape, cam_type)
          self.append_type = "cat"
        case "cam_attention":
          self.policy = CamAttentionPolicy(action_shape, cam_type) ## NOTE: other varaible settings here
          self.append_type = "stack"
        case _: 
          raise ValueError(f"[agent] cannot find policy type {policy}")

    
    def save_model(self, model_path: str):
      torch.save(self.policy.state_dict(), f'{model_path}')
      print(f"Saved Model under '{model_path}'")
      
    def load_model(self, model_path: str):
      self.policy.load_state_dict(torch.load(f'{model_path}')) 
    
    ## Ingest calling train_policy on policy
    def ingest(self, demos: list[Demo], **training_params):
      self.policy.train_policy(demos, **training_params)
      
    ## Inference Call
    def act(self, obs:  Observation) -> torch.Tensor:
      # gripper = [1.0]  # Always open
      # return np.concatenate([arm, gripper], axis=-1)
      
      self.policy.eval()
      images = []
      if self.cam_type & CamType.WRIST:
        # print(f"Using Wrist Image")
        wrist_image = torch.tensor(obs.wrist_rgb, dtype = torch.float32)
        wrist_image = torch.permute(wrist_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
        images.append(wrist_image)
      if self.cam_type & CamType.LEFT_SHOULDER:
        # print(f"Using L Shouulder Image")
        ls_image = torch.tensor(obs.left_shoulder_rgb, dtype = torch.float32)
        ls_image = torch.permute(ls_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
        images.append(ls_image)
      if self.cam_type & CamType.RIGHT_SHOULDER:
        # print(f"Using R Shoulder Image")
        rs_image = torch.tensor(obs.right_shoulder_rgb, dtype = torch.float32)
        rs_image = torch.permute(rs_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
        images.append(rs_image)
      
      if not images:
        raise ValueError("[agent] - Agent - act] No images selected !")
      
      match self.append_type:
        case "cat":
          ## cat on the colours channel (3 * num_cams, W, H)
          torch_obs = torch.cat(images, dim = 0)  
        case "stack":
          ## stacked on new channel (num_cams, 3, W, H)
          torch_obs = torch.cat(images, dim = 0)  
        case _:
          raise ValueError(f"[agent - act] Incorrect 'append_type' (f{self.append_type}) for collating tensors")
        
      torch_obs = torch_obs.unsqueeze(0) ## add a batch dimension (1, ...)
      
      with torch.no_grad():
        pred = self.policy(torch_obs)
      return pred
        
     
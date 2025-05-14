from typing import Literal

from modules.policy.simple_policy import SimplePolicy
from modules.policy.simple_grasp_policy import SimpleGraspPolicy
from modules.policy.cam_attention_policy import CamAttentionPolicy

from rlbench.demo import Demo
from rlbench.backend.observation import Observation

from lib.cam_type import CamType
from lib.policy_type import PolicyType

import torch
import numpy as np
from lib.utils import pick_obs_from_cam

class Agent(object):
    def __init__(self,
      action_shape,
      policy_type: PolicyType,
      cam_type = CamType.WRIST,
      **policy_args, 
    ):
      self.cam_type = cam_type
      self.action_shape = action_shape
      self.policy_type = policy_type
      
      match policy_type:
        case PolicyType.SIMPLE:
          self.policy = SimplePolicy(action_shape, cam_type)
        case PolicyType.SIMPLE_GRASP:
          self.policy = SimpleGraspPolicy(action_shape, cam_type, **policy_args)
        case PolicyType.CAM_ATTENTION:
          self.policy = CamAttentionPolicy(action_shape, cam_type, **policy_args) ## NOTE: other varaible settings here
        case _: 
          raise ValueError(f"[agent - Agent] cannot find policy type {policy_type}")

    
    def save_model(self, model_path: str):
      torch.save(self.policy.state_dict(), f'{model_path}')
      print(f"Saved Model under '{model_path}'")
      
    def load_model(self, model_path: str):
      self.policy.load_state_dict(torch.load(f'{model_path}')) 
    
    ## Ingest calling train_policy on policy
    def ingest(self, demos: list[Demo], **training_params):
      self.policy.train_policy(demos, **training_params)
      
    ## this is abstracted out for observing and printing etc
    # def _infer_move(self, observation: torch.Tensor): ## will return whatever the policy returns, wanted to take the match case out of main `act` function
    #   with torch.no_grad():
    #     policy_ret = self.policy(observation)
        
    #   match self.policy_type:
    #     case PolicyType.SIMPLE:
    #       return policy_ret
    #     case PolicyType.CAM_ATTENTION:
    #       pred, att_weights = policy_ret
    #       return pred, att_weights
    #     case _ :
    #       raise ValueError(f"[agent - _infer_move] unknown PolicyType ({self.policy_type})")  
      
      
    ## Inference Call
    def act(self, obs:  Observation) -> tuple[torch.Tensor, dict]: ## possibly returns other things
      # gripper = [1.0]  # Always open
      # return np.concatenate([arm, gripper], axis=-1)
      
      self.policy.eval()
      images = []
      ## wrist -> ls -> rs
      for ct in CamType.uniques():
        if self.cam_type & ct:
          image = torch.tensor(pick_obs_from_cam(ct, obs, normalise_rgb = True), dtype= torch.float32)
          image = torch.permute(image, (2, 0, 1)) ## 64, 64, 3 -> 3, 64, 64
          images.append(image)
      
      if not images:
        raise ValueError("[agent] - act] No images selected !")
      
      match self.policy_type:
        case PolicyType.SIMPLE:
          ## cat on the colours channel (3 * num_cams, W, H)
          torch_obs = torch.cat(images, dim = 0)  
        case PolicyType.SIMPLE_GRASP:
          torch_obs = torch.cat(images, dim = 0)
        case PolicyType.CAM_ATTENTION:
          ## stacked on new channel (num_cams, 3, W, H)
          torch_obs = torch.stack(images, dim = 0)  
        ## NOTE: add more types as implemented
        case _:
          raise ValueError(f"[agent - act] Unknown 'policy_type' (f{self.policy_type}) for collating tensors")
        
      torch_obs = torch_obs.unsqueeze(0) ## add a batch dimension (1, ...)
      
      with torch.no_grad():
        pred, rest = self.policy(torch_obs)
      
      return pred, rest
      
      
        
     
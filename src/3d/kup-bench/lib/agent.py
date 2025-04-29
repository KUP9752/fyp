from typing import Literal
from modules.simple_policy import SimplePolicy
from modules.cam_attention_policy import CamAttentionPolicy

from rlbench.demo import Demo
from rlbench.backend.observation import Observation

from lib.cam_type import CamType
from lib.policy_type import PolicyType

import torch

class Agent(object):

    def __init__(self,
      action_shape,
      policy_type: PolicyType,
      cam_type = CamType.WRIST
    ):
      self.cam_type = cam_type
      self.action_shape = action_shape
      self.policy_type = policy_type
      
      match policy_type:
        case PolicyType.SIMPLE:
          self.policy = SimplePolicy(action_shape, cam_type)
        case PolicyType.CAM_ATTENTION:
          self.policy = CamAttentionPolicy(action_shape, cam_type) ## NOTE: other varaible settings here
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
    def _infer_move(self, observation: torch.Tensor): ## will return whatever the policy returns, wanted to take the match case out of main `act` function
      with torch.no_grad():
        policy_ret = self.policy(observation)
        
      match self.policy_type:
        case PolicyType.SIMPLE:
          return policy_ret
        case PolicyType.CAM_ATTENTION:
          pred, att_weights = policy_ret
          return pred, att_weights
        case _ :
          raise ValueError(f"[agent - _infer_move] unknown PolicyType ({self.policy_type})")  
      
    ## Inference Call
    def act(self, obs:  Observation):# -> torch.Tensor: ## possibly returns other things
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
        raise ValueError("[agent] - act] No images selected !")
      
      match self.policy_type:
        case PolicyType.SIMPLE:
          ## cat on the colours channel (3 * num_cams, W, H)
          torch_obs = torch.cat(images, dim = 0)  
        case PolicyType.CAM_ATTENTION:
          ## stacked on new channel (num_cams, 3, W, H)
          torch_obs = torch.stack(images, dim = 0)  
        case ## NOTE: add more types as implemented
        case _:
          raise ValueError(f"[agent - act] Unknown 'policy_type' (f{self.policy_type}) for collating tensors")
        
      torch_obs = torch_obs.unsqueeze(0) ## add a batch dimension (1, ...)
      
      return self._infer_move(torch_obs)
      
      
        
     
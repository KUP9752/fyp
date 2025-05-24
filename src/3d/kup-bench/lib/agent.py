from typing import Callable, Literal

from modules.policy.simple_policy import SimplePolicy
from modules.policy.simple_grasp_policy import SimpleGraspPolicy
from modules.policy.depth_grasp_policy import DepthGraspPolicy
from modules.policy.cam_attention_policy import CamAttentionPolicy
from modules.policy.resnet_grasp_policy import ResNetGraspPolicy
from modules.policy.rnn_grasp_policy import RNNGraspPolicy

from rlbench.demo import Demo
from rlbench.backend.observation import Observation

from lib.cam_type import CamType
from lib.policy_type import PolicyType

import torch
import numpy as np
from lib.utils import pick_obs_from_cam

class Agent(object):
    def __init__(self,
      action_shape: int,
      policy_type: PolicyType,
      cam_type = CamType.WRIST,
      **policy_args, 
    ):
      self.cam_type = cam_type
      self.action_shape = action_shape
      self.policy_type = policy_type

      ## aggregation method later on when acting
      self.tensor_agg: Callable[[list[torch.Tensor]], torch.Tensor]

      match policy_type:
        case PolicyType.SIMPLE:
          self.policy = SimplePolicy(action_shape, cam_type)
          self.tensor_agg = self._catter
        case PolicyType.SIMPLE_GRASP:
          self.policy = SimpleGraspPolicy(action_shape, cam_type, **policy_args)
          self.tensor_agg = self._catter
        case PolicyType.DEPTH_GRASP:
          self.policy = DepthGraspPolicy(action_shape=action_shape, cam_type = cam_type, **policy_args)
          self.tensor_agg = self._catter
        case PolicyType.RESNET_GRASP:
          self.policy = ResNetGraspPolicy(action_shape=action_shape, cam_type = cam_type, **policy_args)
          self.tensor_agg = self._catter
        case PolicyType.RNN_GRASP:
          self.policy = RNNGraspPolicy(action_shape=action_shape, cam_type = cam_type, **policy_args)
          self.tensor_agg = self._catter
          self.prev_state = None
        case PolicyType.CAM_ATTENTION:
          self.policy = CamAttentionPolicy(action_shape, cam_type, **policy_args) ## NOTE: other varaible settings here
          self.tensor_agg = self._stacker
        case _: 
          raise ValueError(f"[agent - Agent] cannot find policy type {policy_type}")
        
    def __str__(self) -> str:
      return f"agent-policy:{self.policy_type}-cams:{self.cam_type}-policy:{self.policy}"

    def __repr__(self) -> str:
      return f"Agent(policy_type={self.policy_type}, cam_type={self.cam_type}, policy={self.policy})"
    
    def save_model(self, model_path: str):
      torch.save(self.policy.state_dict(), f'{model_path}')
      print(f"Saved Model under '{model_path}'")
      
    def load_model(self, model_path: str):
      self.policy.load_state_dict(torch.load(f'{model_path}')) 
    
    ## Ingest calling train_policy on policy
    def ingest(self, demos: list[Demo], **training_params):
      self.policy.train_policy(demos, **training_params)
      
    ## Following hidden functions are for assigning them to `self.tensor_agg`
    def _catter(self, ts: list[torch.Tensor]) -> torch.Tensor:  
      return torch.cat(ts, dim = 0)
    
    def _stacker(self, ts: list[torch.Tensor]) -> torch.Tensor:
      return torch.stack(ts, dim = 0)
    
    def reset_episode(self):
      ## No other reason to reset state currently
      if self.policy_type == PolicyType.RNN_GRASP:
        self.prev_state = None

    def _act_rnn(self, obs_tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
      assert self.policy_type == PolicyType.RNN_GRASP, f"[agent - (act_rnn) sequential act RNN function is called with a policy that is not RNN]"

      with torch.no_grad():
        action, rets = self.policy(
          obs_tensor, 
          hidden_state=self.prev_state,
          lengths = None ## not passing 'lengths' on purpose to force the inference branch of policy
        )
      self.prev_state = (rets["h"], rets["c"])

      return action, rets

    ## Inference Call
    def act(self, obs:  Observation) -> tuple[torch.Tensor, dict]: ## possibly returns other things
      
      ## === Gather the `Obserevation` as tensor data
      images = []
      ## wrist -> ls -> rs -> wd
      for ct in CamType.uniques():
        if self.cam_type & ct:
          image = torch.tensor(pick_obs_from_cam(ct, obs, normalise_rgb = True), dtype= torch.float32)
          image = torch.permute(image, (2, 0, 1)) ## 64, 64, 3 -> 3, 64, 64
          images.append(image)
      
      if not images:
        raise ValueError("[agent] - act] No images selected !")
      
      if self.tensor_agg is None:
        raise ValueError(f"[agent - act] Tensor aggregation method was not set in the constructor!")
      
      torch_obs = self.tensor_agg(images)

      torch_obs = torch_obs.unsqueeze(0) ## add a batch dimension (1, ...)

      ## === Model Prediciton
      self.policy.eval()

      ### this is a sequence model, so delegate to the other act method
      ## NOTE: add other sequence based models here
      if self.policy_type == PolicyType.RNN_GRASP:
        return self._act_rnn(torch_obs)
      
      with torch.no_grad():
        pred, rest = self.policy(torch_obs)
      
      return pred, rest
      
      
        
     
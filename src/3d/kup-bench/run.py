#%%
# import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig

# Tasks
## No Obstacle
from rlbench.tasks.reach_target_no_obs_side_r import ReachTargetNoObsSideR as ReachNoObs_SideR
from rlbench.tasks.reach_target_no_obs_side_l import ReachTargetNoObsSideL as ReachNoObs_SideL
from rlbench.tasks.reach_target_no_obs_central import ReachTargetNoObsCentral as ReachNoObs_Central
from rlbench.tasks.reach_target_no_obs_random import ReachTargetNoObsRandom as ReachNoObs_PlaceRandom
## Obstacle
from rlbench.tasks.reach_target_obs_static_left import ReachTargetObsStaticLeft as ReachObs_StaticLeft
from rlbench.tasks.reach_target_obs_static import ReachTargetObsStatic as ReachObs_Static 
from rlbench.tasks.reach_target_obs_random_static import ReachTargetObsRandomStatic as ReachObs_RandomStatic
from rlbench.tasks.reach_target_obs_random import ReachTargetObsRandom as ReachObs_Random
from rlbench.tasks.reach_target_obs_ind_random import ReachTargetObsIndRandom as ReachObs_IndepRandom
## Grasp
from rlbench.tasks.simple_grasp import SimpleGrasp as Grasp_Simple
from rlbench.tasks.grasp_and_move import GraspAndMove as Grasp_ThenMove
## Vision Experiments - Grasp
from rlbench.tasks.vision_static import VisionStatic as Vision_Static
from rlbench.tasks.vision_random import VisionRandom as Vision_Random

from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from pyrep.const import RenderMode
from pyrep.objects import Object, VisionSensor, Shape
from pyrep.backend import sim

from modules.policy.cam_attention_old import CamAttentionOld

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torchvision import models, transforms
from PIL import Image, ImageDraw

# from policy import  CamType

from lib.agent import Agent
from lib.cam_type import CamType
from lib.policy_type import PolicyType

from lib.utils import get_task_name, now, params_string, load_demos
from task_utils import save_demos_for, load_demos_for, launch_test_env, run_determined_reach_with_agent, run_determined_grasp_with_agent, run_reachobs_random_task_with_agent
from seed import set_seed
from pprint import pprint

from modules.policy.fusing_policy import FuseConfig

from itertools import product
DATASET  = 'data/20demos'
#%%
task = Vision_Random
  
env = launch_test_env(
  "data/20demos/normal-Vision_Random", 
  enableds = CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
)

normal = {
  "scale": 1.,
  # "wrist_cam_distance": 0.6
}
smaller = {
  "scale": 0.5,
  # "wrist_cam_distance": 0.3
}
training_demos = load_demos_for(
  10,
  env, 
  task, 
  "data/20demos/normal-Vision_Random",
  task_params = normal
)

test_demos = load_demos_for(
    10,
    env, 
    task, 
    "data/test/10demos/normal-Vision_Random",
    task_params = normal
  )

#%%
training_params = {
  "epochs": 50, 
  "minibatch_size": 10,
  "lr": 1e-3,
  "shuffle_obs_in_demo": False,
  "shuffle_data": True,
  "lock_loader_seed": 42,
  "dataset_to_use": "demo",
  "lambda_grasp_loss": 1,
}

all_cams = [comb for comb in CamType.all_combinations() if not comb & CamType.OVERHEAD and not comb & CamType.FRONT]
#%%
configs = [
    FuseConfig.WDLR,
    FuseConfig.WLR_D,
    FuseConfig.DEPTH_FEATS_GATED,
    FuseConfig.DEPTH_FEATS_ATTN,
    FuseConfig.WD_LR,
    FuseConfig.WD_LR_ATTN,
    FuseConfig.Wfilm_D,
    FuseConfig.W_Dfilm,
    FuseConfig.Wfilm_Dfilm,
    FuseConfig.Wfilm_D_LATE,
    FuseConfig.W_Dfilm_LATE,
    FuseConfig.Wfilm_Dfilm_LATE,
    FuseConfig.W_D_L_R_FILM, # TODO not fixed
    FuseConfig.W_D_L_R,
    FuseConfig.W_D_L_R_ATTN,
  ]
# print()
agent = lambda i, p, c, ct: Agent(
  action_shape= 8,
  policy_type=PolicyType.FUSING, 
  cam_type= ct,

  is_grasp = i, 

  fuse_config = c,
  fusing_opts = {}, ## make sure to use defaults 

  use_proprio = p,
  proprio_opts = {}, ## make sure to use defaults 

  ## others are defaulted
)
bb = [True, False]

df = pd.DataFrame(columns=[
  "proprio",
  "grasp",
  "config",
  "cam_type",
  "param_count"
], index = range(
  len(configs)
  * len(bb)
  * len(bb)
  * len(configs)
  * len(all_cams)
))

print(f" proprio \t grasp \t config\t")
idx = 0
for ig, ip, cfg, ct in product(bb, bb, configs, all_cams):
  try:
    count = sum(p.numel() for p in agent(ig, ip, cfg, ct).policy.parameters())
    df.loc[idx] = {
      "proprio": ip,
      "grasp": ig,
      "config":cfg,
      "cam_type":ct,
      "param_count":count
    }
    idx += 1  
  except Exception as e:
    print("not this combo lad")
    
df.to_csv("param_count.csv")
#%%
# for name, param in agent(True, False, cfg).policy.named_parameters():
#    print(f"{name}: {param.shape} → {param.numel()} params")

#%%
# agent.policy.load_state_dict(torch.load("models/PROMISING-rwd-reach-1-demo-wrist+r_shoulder-ReachObs_Random-PolicyType.CAM_ATTENTION.pth"))



#%%


rets = run_determined_grasp_with_agent(
  env, task, 
  agent, 
  test_demos, 
  max_eplen="demo_max", 
  do_extra_outs=False, 
  task_params = normal
)
# rets
# # print(f"Above: {rets['avg_attentions_above_obstacle']}")
# # print(f"Below: {rets['avg_attentions_below_obstacle']}")
# #%%
# plt.plot(range(training_params["epochs"]), agent.policy.action_losses,  color='tab:blue',  marker='', linestyle='-', label='Pose Loss')
# plt.plot(range(training_params["epochs"]), agent.policy.grasp_losses, color='tab:orange', marker='', linestyle='--', label='Grasp Loss')
# plt.title('Loss per Epoch, $k_{mask}$ = %d' % training_params["last_k_grasp_mask"])
# plt.xlabel('Epoch')
# plt.ylabel('Avg Loss')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# # plt.savefig(f"/home/kup/Desktop/code/fyp-report/assets/cam-comb/grasp-simple/k-losses-k7.png", format="png", dpi=1000)
# plt.show()

# #%%
# env.get_task(ReachObs_Random)
# target_rgb = Shape("target").get_color()
#%%
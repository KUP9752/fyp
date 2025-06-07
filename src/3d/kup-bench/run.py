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

normal = {
  "scale": 0.5,
  "wrist_cam_distance": 0.3
}


#%%

env = launch_test_env(
  dataset_root=DATASET,
  enableds = CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER
)
#%%
task = Vision_Static

# agent = Agent(
#   env.action_shape[0],
#   policy_type = PolicyType.FUSING,
#   cam_type = CamType.WRIST,
#   config = FuseConfig.WDLR
#   is_grasp = True
#   use_proprio = False
#   # opts = {}
# )

# model_name = f"rwd-{get_task_name(task)}-{agent}--{now()}"
# model_path = f"./all-models/reach-with-demos/{model_name}.pth"
# print(model_name)
# agent

#%%
task = Vision_Random
normal = {
  "scale": 1.,
  "wrist_cam_distance": 0.6
}

env._dataset_root = f"data/20demos/normal-{get_task_name(task)}"
task_env = env.get_task(task, **normal)

demos = task_env.get_demos(1, live_demos=False)
#%10
training_params = {
  "epochs": 1, 
  "minibatch_size": 5,
  "lr": 1e-3,
  "shuffle_obs_in_demo": False,
  "shuffle_data": True,
  "lock_loader_seed": 42,
  "dataset_to_use": "demo",
  "lambda_grasp_loss": 1,
}

# bs = [False, True]
# for p in bs:
#   for g in bs:
agent = Agent(
  env.action_shape[0],
  policy_type = PolicyType.FUSING,
  cam_type = CamType.WRIST | CamType.WRIST_DEPTH,
  fuse_config = FuseConfig.W_Dfilm,
  is_grasp = True,
  use_proprio = False,
  # opts = {}
)
agent.ingest(demos, **training_params) ## trains here

#%%
# agent.policy.load_state_dict(torch.load("models/PROMISING-rwd-reach-1-demo-wrist+r_shoulder-ReachObs_Random-PolicyType.CAM_ATTENTION.pth"))
#%%
rets = run_determined_grasp_with_agent(
  env, 
  task, 
  agent, 
  demos, 
  max_eplen="demo_max", 
  do_extra_outs=False, 
  task_params = normal
)[0]
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
# from modules.dataset.demo_dataset import DemoDataset
# from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence

# def collate(batch):
#     ## batch contains [(input, labels)] where each input is a complete demo (in terms of the data in sequence rgb for example)
#     ## input: (t, ch, w, h) 
#     inputs, labels, loader = zip(*batch)
#     [print(f"{input.shape}") for input in inputs]

#     ## enforcing types for later
#     print(f"{len(labels) =}")
    
#     real_lengths = torch.LongTensor([inp.shape[0] for inp in inputs])
#     print(f"{real_lengths.shape =}")
    
#     inputs_padded = pad_sequence(inputs, batch_first=True) ## CHECK: if it gives (B, t, ch, w, h)
#     print(f"{inputs_padded.shape = }")

#     labels_padded = pad_sequence(labels, batch_first=True) ## CHECK: if it gives (B, t, ch, w, h)
#     print(f"{labels_padded.shape = }")


#     ## need to return shape (B, t, ch, w, h) for the input and labels
#     ## also returning lenths for LSTM use later
#     return inputs_padded, labels_padded, real_lengths

# dataset = DemoDataset(demos, cam_type= CamType.WRIST | CamType.WRIST_DEPTH | CamType.LEFT_SHOULDER | CamType.RIGHT_SHOULDER, get_type="cat")
# loader = DataLoader(
#   dataset,
#   shuffle = True,
#   batch_size=10,
#   collate_fn=collate,
#   generator=torch.manual_seed(42)
# )

# for ins, out, ls in loader:
#   print(f"{ins.shape = }")
  
#   plt.imshow(ins[4, 20, -1, :, :])
#   break
  
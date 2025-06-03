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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torchvision import models, transforms
from PIL import Image, ImageDraw

# from policy import  CamType

from lib.agent import Agent
from lib.cam_type import CamType
from lib.policy_type import PolicyType

from lib.utils import get_task_name, now, params_string
from task_utils import load_demos_for, launch_test_env, run_determined_reach_with_agent, run_determined_grasp_with_agent
from seed import set_seed
from pprint import pprint

from itertools import product
DATASET  = 'data/20demos'
#%%

env = launch_test_env(
  dataset_root=DATASET,
  enableds = CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER
)
#%%
task = ReachObs_Random

agent = Agent(
  env.action_shape[0],
  policy_type = PolicyType.SIMPLE_GRASP,
  cam_type = CamType.WRIST,
  # grasp_thresh = 0.5,
  # config = "depth_ch",
  # opts = {}
)

try:
  target = Shape("target")
  target_rgb = torch.tensor(target.get_color())
except RuntimeError:
  print(f"'target' doesn't exist meaning this is a different task")
  target_rgb = None

new_agent = lambda: Agent(
  env.action_shape[0],
  policy_type = PolicyType.CAM_ATTENTION,
  cam_type = CamType.WRIST,
  # config="depth_ch",
  target_rgb = target_rgb
)

model_name = f"rwd-{get_task_name(task)}-{agent}--{now()}"
model_path = f"./all-models/reach-with-demos/{model_name}.pth"
print(model_name)
agent

#%%
# demos = load_demos_for(10, env, task, DATASET)
task_env = env.get_task(task)
test_demos = task_env.get_demos(20, live_demos=False)
#%%
demos = test_demos
#%%
task = Vision_Static
normal = {
  "scale": 1.,
  "wrist_cam_distance": 0.6
}
smaller = {
  "scale": 0.5,
  "wrist_cam_distance": 0.3
}

env._dataset_root = f"data/1demo/normal-{get_task_name(task)}"
task_env = env.get_task(task, **normal)
demos = task_env.get_demos(1, live_demos=False)


env._dataset_root = f"data/1demo/smaller-{get_task_name(task)}"
task_env = env.get_task(task, **smaller)
small_demos = task_env.get_demos(1, live_demos=False)

#%%
training_params = {
  "epochs": 100,
  "minibatch_size": 10,
  "lr": 1e-3,
  "shuffle_obs_in_demo": False,
  "shuffle_data": True,
  "lock_loader_seed": 42,
  "dataset_to_use": "demo",
  "lambda_grasp_loss": 1,
}

ingest_num = 10

print(f"-> Using {ingest_num} demos")

agent.ingest(test_demos, **training_params) ## trains here
agent.save_model(model_path)
#%%
#%%
env._dataset_root = f"data/20demos"
task_env = env.get_task(ReachNoObs_PlaceRandom)
demos = task_env.get_demos(5, live_demos=False)
#%%

#%%
import torch.nn as nn
agent = new_agent()

training_params = {
  "epochs": 200,
  "minibatch_size": 1,
  "lr": 1e-3,
  "shuffle_obs_in_demo": False,
  "shuffle_data": True,
  "lock_loader_seed": 42,
  "dataset_to_use": "demo",
  # "lambda_grasp_loss": 1.2,
  # "last_k_grasp_mask": 5
}
agent.ingest(demos, **training_params) ## trains here
#%%
run_determined_reach_with_agent(
  env, 
  task, 
  agent, 
  demos, 
  max_eplen="demo_max", 
  # do_extra_outs=True, 
  # task_params = normal
)[0]
#%%
task = ReachNoObs_Central
env._dataset_root = "data/20demos"
task_env = env.get_task(task)
_, obs = task_env.reset()
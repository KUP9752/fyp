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
import torch
import torch.nn.functional as F

from torchvision import models, transforms
from PIL import Image, ImageDraw

# from policy import  CamType

import plotly.graph_objects as go
import open3d as o3d
from matplotlib import pyplot as plt

from lib.agent import Agent
from lib.cam_type import CamType
from lib.policy_type import PolicyType

from lib.utils import get_task_name, now, params_string
from task_utils import load_demos_for, launch_test_env, run_determined_reach_with_agent
from seed import set_seed
from pprint import pprint
DATASET  = 'data/20demos'
#%%

env = launch_test_env(
  dataset_root=DATASET,
  enableds = CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER
)
#%%
task = ReachObs_Static

agent = Agent(
  env.action_shape[0],
  policy_type = PolicyType.SIMPLE,
  cam_type = CamType.WRIST ,#| CamType.WRIST_DEPTH,
  # grasp_thresh = 0.5,
  # config = "depth_ch",
  # opts = {}
)

model_name = f"rwd-{get_task_name(task)}-{agent}--{now()}"
model_path = f"./all-models/reach-with-demos/{model_name}.pth"
print(model_name)
agent

#%%
# demos = load_demos_for(10, env, task, DATASET)
task_env = env.get_task(task)
test_demos = task_env.get_demos(1, live_demos=True)
#%%
demos = test_demos

#%%
training_params = {
  "epochs": 5000,
  "minibatch_size": 10,
  "lr": 1e-3,
  "shuffle_obs_in_demo": False,
  "shuffle_data": True,
  "lock_loader_seed": 42,
  "dataset_to_use": "obs",
  # "lambda_grasp_loss": 1,
}

ingest_num = 10

print(f"-> Using {ingest_num} demos")

agent.ingest(demos, **training_params) ## trains here
agent.save_model(model_path)
#%%
#%%
rets = run_determined_reach_with_agent(
  env, 
  task, 
  agent, 
  test_demos, 
  "demo_max", 
  within_err_dist=0.1
)

rets
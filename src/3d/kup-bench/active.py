#%%
## 1. Initalise and Import
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, JointPosition
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

import numpy as np
import torch
import torch.nn.functional as F

from lib.agent import Agent
# from modules.active_agent import ActiveAgent_Plan1 as ActiveAgent

import matplotlib.pyplot as plt
from lib.cam_type import CamType
from lib.policy_type import PolicyType

from lib.utils import get_task_name, now
from task_utils import launch_test_env
from seed import set_seed
from pprint import pprint

set_seed()

#%%
## 2. Create Environment and Set Model Name
# To use 'saved' demos, set the path below, and set live_demos=False

env = launch_test_env(
  "data/20demos",
  enabled_config = CameraConfig(
    rgb=True, depth=True, mask=True, point_cloud=True,
    render_mode=RenderMode.OPENGL,image_size=(64, 64)
  ),
  enableds = CamType.WRIST | CamType.LEFT_SHOULDER | CamType.RIGHT_SHOULDER, 
  disabled_config= CameraConfig(
    rgb=False, depth=False, mask=False,
    render_mode=RenderMode.OPENGL
  ), 
  ## TODO: start with Joint Position control then go to velocity if nedeed
  # action_mode = MoveArmThenGripper(
  #     arm_action_mode=JointPosition(absolute_mode=True), 
  #     gripper_action_mode=Discrete()
  # )
  action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), 
    gripper_action_mode=Discrete()
  )
)
#%%
## 3. Attach Task and create Agent
## NOTE: importing cv2 before lauunching the env crashes the system??
from modules.active_agent import ActiveAgent_Plan1 as ActiveAgent
pol_type = PolicyType.DEPTH_GRASP

cam_type = CamType.WRIST | CamType.WRIST_DEPTH

# task = Vision_Random
task = ReachObs_Random
print(env.get_task.__code__.co_filename)

# task_env = env.get_task(task_class = task, **task_params)
task_env = env.get_task(task_class = task)
target_name = "target"

  
wd = CamType.WRIST | CamType.WRIST_DEPTH
# all_cams = CamType.LEFT_SHOULDER | CamType.RIGHT_SHOULDER | CamType.WRIST_DEPTH

## use best il agent
il_agent = Agent(
  env.action_shape[0],
  policy_type=PolicyType.RNN_GRASP,
  cam_type = wd,
  rnn_opts = {
    "config": "depth_feats",
    # "attn_opts": {"is_deep_fuse": True},
    "use_proprio": False
  }
)

agent = ActiveAgent(
  env.action_shape[0],
  il_agent = il_agent, 
  cam_type = wd, # doesnt really matter
  vis_thresh= 0.40,
  sample_radius=0.1
)

model_name = f"rwd-{get_task_name(task)}-{il_agent}--{now()}"
model_path = f"./all-models/reach-with-demos/{model_name}.pth"
print(model_name)
agent
#%%
# demos: list[Demo] = task_env.get_demos(10, live_demos=False)
# demos
#%%
## 5. Train
training_params = {
  "epochs": 600,
  "minibatch_size": 10,
  "lr": 1e-3,
  "shuffle_obs_in_demo": False,
  "shuffle_data": True,
  "lock_loader_seed": 1,
  "dataset_to_use": "demo",
  "lambda_grasp_loss": 1,
  "data_label": "joint_velocities"
}

ingest_num = 10

print(f"-> Using {ingest_num} demos")

# il_agent.ingest(demos[:ingest_num], **training_params) ## trains here
# il_agent.save_model(model_path)
#%%
# demos[0][0].joint_positions
#%%
string = torch.load("all-models/reach-with-demos/rwd-ReachObs_Random-agent-policy:rnn_grasp_policy-cams:wrist+wrist_depth-policy:rnn_grasp_policy-rnn_opts:{'config': 'depth_feats', 'use_proprio': False}--_June08_15-36.pth")
il_agent.policy.load_state_dict(string)
#%%
# _, obs = task_env.reset()
# il_agent.policy.to("cpu")
# # for demo in test_demos:
# #   _, obs = task_env.reset_to_demo(demo)

# count = 0
# done = False
# distances = []
# while not done:
#   obs: Observation
  
#   action, rets = il_agent.act(obs)
#   # rgb_out = rets["rgb_fused"]
#   # rgb_unfsuedout = rets["rgb_unfused"]
#   # depth_out = rets["depth_fused"]

#   # plt.imshow(obs.wrist_depth)
#   # plt.imshow(obs.wrist_rgb)
#   # plot_featmap(rgb_out, "RGB feats (fused w/ attn)")
#   # plot_featmap(rgb_unfsuedout, "RGB feats (not fused)")
#   # plot_featmap(depth_out)

#   action = action.squeeze(0)
#   # print(f"{action.shape =}")
#   # print(f"{action[-1] =}")
#   # print(f"{action.shape = }")
#   obs, reward, done = task_env.step(action)
#   gripper = Object.get_object("Panda_gripper")
#   target = Object.get_object("target")

  
  
#   # vis_score = check_visibility("cam_wrist", "target")
  
#   # print(f"{vis_score = }")
  
  
#   distance = np.linalg.norm(gripper.get_position() - target.get_position())
#   distances.append(distance)
#   # print(f"{done = }")
  
#   count += 1
#   if count == 100:
#     break
  
# print(f"Done Successfull! done in {count} steps" if done else "Failed!")
# print(f"Final distance: {distances[-1]}")

#%%
task_env = env.get_task(ReachObs_Random)
done, ret_dict = agent.act(task_env.reset()[1], task_env, within_obs=0.1)
print(f"Done?: {done}")

for _ in range(100_000_000):
  pass
#%%
env.shutdown()


#%%
## 1. Initalise and Import
import numpy as np

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
from rlbench.tasks.reach_target_no_obs import ReachTargetNoObs as ReachNoObs_PlaceRandom
## Obstacle
from rlbench.tasks.reach_target_obs_static_left import ReachTargetObsStaticLeft as ReachObs_StaticLeft
from rlbench.tasks.reach_target_obs_static import ReachTargetObsStatic as ReachObs_Static
from rlbench.tasks.reach_target_obs_random_static import ReachTargetObsRandomStatic as ReachObs_RandomStatic
from rlbench.tasks.reach_target_obs_random import ReachTargetObsRandom as ReachObs_Random
from rlbench.tasks.reach_target_obs_ind_random import ReachTargetObsIndRandom as ReachObs_IndepRandom
## Grasp
from rlbench.tasks.simple_grasp import SimpleGrasp as Grasp_Simple
from rlbench.tasks.grasp_and_move import GraspAndMove as Grasp_ThenMove

from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from pyrep.const import RenderMode
from pyrep.objects import Object

import numpy as np
import torch.nn.functional as F

# from policy import  CamType

from matplotlib import pyplot as plt
from policy import Agent, CamType
from utils import set_seed

set_seed(42)

num_demos = 10
# cam_type = CamType.WRIST 

#%%
## 2. Create Environment and Set Model Name
# To use 'saved' demos, set the path below, and set live_demos=False


live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(True) ## important to get the data from the joints etc
cam_config = CameraConfig(rgb=True, depth=False, mask=False,
                              render_mode=RenderMode.OPENGL,
                    image_size=(64, 64))
nocam_config = CameraConfig(rgb=False, depth=False, mask=False,
                          render_mode=RenderMode.OPENGL)

obs_config.right_shoulder_camera = cam_config
obs_config.left_shoulder_camera = cam_config
obs_config.overhead_camera = nocam_config
obs_config.front_camera = nocam_config

## active camera: wrist camera
obs_config.wrist_camera = cam_config


action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())

env = Environment(
    action_mode, DATASET, obs_config, False)
env.launch()

#%%
## 3. Attach Task and create Agent
task = ReachObs_IndepRandom
task_env = env.get_task(task)
agent = Agent(env.action_shape[0], cam_type)

model_name = f"rwd-reach-{num_demos}-demo-{cam_type}-{task}"
model_path = f"./models/{model_name}.pth"
print(model_name)
task

# %%
## 4. Request Demos

# demos = []
# for var in [0,1,2]:
#   task_env.set_variation(var)
#   demos += task_env.get_demos(1, live_demos=live_demos, random_selection = True)

demos: list[Demo] = task_env.get_demos(num_demos, live_demos=live_demos, random_selection = True)
# print(f"What is in the demos: {type(demos)} | {type(demos[0])}")
# print(f"{demos = } | {type(demos) = } | {len(demos) = }")
print(f"Observations len: {list(map(len, demos))}")

training_params = {
  "epochs": 1000,
  "minibatch_size": 32,
  "lr": 1e-4,
  "shuffle_data": False
}
agent.ingest(demos, **training_params) ## trains here
agent.save_model(model_path)
# %%
task_env.variation_count()
# lens = list(map(len, demos))
# print(f"Observations len: {lens}")
# print(f"average {sum(lens)/len(lens)}")
# print(f"max {max(lens)}")


# %%
## 5. Load Agent
agent.load_model(model_path)

#%%
## 6. Task Execution
agent.policy.to("cpu")
_, obs = task_env.reset()
# plt.imshow(obs.wrist_rgb)
count = 0
done = False
distances = []
while not done:
  obs: Observation
  action = agent.act(obs).squeeze(0)
  # print(f"{action.shape = }")
  obs, reward, done = task_env.step(action)
  gripper = Object.get_object("Panda_gripper")
  target = Object.get_object("target")
  distance = np.linalg.norm(gripper.get_position() - target.get_position())
  distances.append(distance)
  # print(f"{done = }")
  
  count += 1
  if count == 80:
    break
    
print(f"{f"Done Successfull! done in {count} steps" if done else "Failed!"}")
print(f"Final distance: {distances[-1]}")

#%%
## Getting the initial camera positions per task
# tasks = [SideR, SideL, Central]
# for task in tasks:
#   task_env = env.get_task(task)
#   _, obs = task_env.reset()
  
#   plt.imshow(obs.wrist_rgb)
#   plt.savefig(f"images/wrist-{task.__name__}.png")
#   plt.close()
  
#   plt.imshow(obs.left_shoulder_rgb)
#   plt.savefig(f"images/lshoulder-{task.__name__}.png")
#   plt.close()
  
#   plt.imshow(obs.right_shoulder_rgb)
#   plt.savefig(f"images/rshoulder-{task.__name__}.png")
#   plt.close()


#%%
# 7. Manipulate object positions and calculate distances.


gripper = Object.get_object("Panda_gripper")
target = Object.get_object("target")
# obj.set_position([0.001,0.03,0.96])
print(f"Gripper pos: {gripper.get_position()}")
print(f"target pos: {target.get_position()}")


# %% 
## !! Finish
env.shutdown()

# %%
## Random Testing Cell
from policy import CamType
# CamType.all_combinations()
c = CamType.LEFT_SHOULDER | CamType.WRIST

str(c)



## more testin 
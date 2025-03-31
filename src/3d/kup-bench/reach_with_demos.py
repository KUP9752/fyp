#%%
## 1. Initalise and Import
import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig

from my_tasks.reach_target_no_obs import ReachTargetNoObs
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from pyrep.const import RenderMode

import numpy as np
import torch
import torch.nn.functional as F

from policy import Policy, Agent, CamType

from utils import set_seed

set_seed(42)


#%%
# 2. Create Environment and Set Model Name
# To use 'saved' demos, set the path below, and set live_demos=False
model_name = "reach-1-demo"
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
task_env = env.get_task(ReachTargetNoObs)
agent = Agent(env.action_shape[0], CamType.WRIST)

# %%
## 4. Request Demos
demos: list[Demo] = task_env.get_demos(2, live_demos=live_demos)
print(f"{demos = } | {type(demos) = } | {len(demos) = }")

demos = np.array(demos, dtype=object).flatten()
agent.ingest(demos) ## trains here
agent.save_model(model_name)

training_steps = 120
episode_length = 40
obs = None

## this is for RL version with demos, i want to do IL for now
# for i in range(training_steps):
#     if i % episode_length == 0:
#         print('Reset Episode')
#         descriptions, obs = task.reset()
#         print(descriptions)
#     action = agent.act(obs)
#     print(action)
#     obs, reward, terminate = task.step(action)



# %%
## 5. Load Agent
agent.load_model(model_name)

#%%
## 6. Task Execution
agent.policy.to("cpu")
_, obs = task_env.reset()
done = False
while not done:
  obs: Observation
  action = agent.act(obs).squeeze(0)
  # print(f"{action.shape = }")
  obs, reward, done = task_env.step(action)
  # print(f"{reward = } | {done = }")
    
print("Done Successfully!")



# %%
## 7. Finish
env.shutdown()

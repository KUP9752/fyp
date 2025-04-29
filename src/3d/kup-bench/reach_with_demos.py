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

from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from pyrep.const import RenderMode
from pyrep.objects import Object, VisionSensor, Shape
import numpy as np
import torch
import torch.nn.functional as F

from torchvision import models, transforms
from PIL import Image, ImageDraw

# from policy import  CamType

from matplotlib import pyplot as plt

from lib.agent import Agent
from lib.cam_type import CamType
from lib.policy_type import PolicyType

from utils import get_task_name
from seed import set_seed

set_seed()

num_demos = 10
cam_type = CamType.WRIST| CamType.RIGHT_SHOULDER

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

## active camera: wri camera
obs_config.wrist_camera = cam_config


action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())

env = Environment(
    action_mode, DATASET, obs_config, False)
env.launch()

#%%
## 3. Attach Task and create Agent
pol_type = PolicyType.CAM_ATTENTION

task = ReachObs_Random
task_env = env.get_task(task)
agent = Agent(env.action_shape[0], pol_type, cam_type)

model_name = f"rwd-reach-{num_demos}-demo-{cam_type}-{get_task_name(task)}-{pol_type}"
model_path = f"./all-models/reach-with-demos/{model_name}.pth"
print(model_name)
task
# %%
## 4. Request Demos3
# demos = []
# for var in [0,1,2]:
#   task_env.set_variation(var)
#   demos += task_env.get_demos(1, live_demos=live_demos, random_selection = True)
demos: list[Demo] = task_env.get_demos(num_demos, live_demos=live_demos)
demos


# # print(f"What is in the demos: {type(demos)} | {type(demos[0])}")
# # print(f"{demos = } | {type(demos) = } | {len(demos) = }")
# print(f"Observations len: {list(map(len, demos))}")
# %%
agent.policy
# %%
## 5. Train
training_params = {
  "epochs": 1000,
  "minibatch_size": 64,
  "lr": 1e-2,
  "shuffle_obs_in_demo": False,
  "shuffle_data": False
}

ingest_num = 1

agent.ingest(demos[:ingest_num], **training_params) ## trains here
agent.save_model(model_path)
# %%
## Some detection trials
# task_env.variation_count()

# model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()
# def checkImage(image_arr):
#   # Load and preprocess the image
#   # image = Image.open("../../../assets/demo-trials-no_obs/tasks/static-tasks-camera/rshoulder-side_r.png")
#   image = Image.fromarray(image_arr)
#   image = image.convert("RGB")
  
#   plt.imshow(image)
#   plt.axis('off')  # Hide axes
#   plt.show()

#   transform = transforms.Compose([
#     transforms.ToTensor()
#   ])
#   print(f"{image.size = }")
#   image_tensor = transform(image).unsqueeze(0)
#   print(f"{image_tensor.shape = }")

#   # Perform inference
#   with torch.no_grad():
#       prediction = model(image_tensor)

#   # Check detection confidence
#   threshold = 0.001  # Confidence score threshold
#   from pprint import pprint
#   pprint(prediction, indent = 2)

#   boxes_above_threshold = prediction[0]["boxes"][prediction[0]["scores"] > threshold]
#   print(f"{boxes_above_threshold = }")

#   # Draw bounding boxes on the image
#   draw = ImageDraw.Draw(image)
#   print(len(boxes_above_threshold))
#   for i, box in enumerate(boxes_above_threshold):
#       xmin, ymin, xmax, ymax = box
#       draw.rectangle([xmin, ymin, xmax, ymax], outline="red" if i % 2 == 0 else "blue", width=1)

#   # Display the image with bounding boxes
#   plt.imshow(image)
#   plt.axis('off')  # Hide axes
#   plt.show()

# %%
## Check Visibility


def check_visibility(view_handle: str, target_handle: str, tolerance = 0.1):
  cam = VisionSensor(view_handle)
  target = Shape(target_handle)

  rgb = cam.capture_rgb()
  target_rbg = target.get_color()
  
  mask = np.all(np.abs(rgb - target_rbg) < tolerance, axis = -1)
  visible_pxs = np.count_nonzero(mask)
  print(f"{visible_pxs = }")
  print(f"{mask.size =}")
  
  return visible_pxs / mask.size

#%%
## 6. Task Execution
agent.policy.to("cpu")
# task_env = env.get_task(ReachNoObs_Central)
_, obs = task_env.reset()
count = 0
done = False
distances = []

# %% Auto task Execution
agent.policy.to("cpu")
# task_env = env.get_task(ReachNoObs_Central)
_, obs = task_env.reset()
count = 0
done = False
distances = []
while not done:
  obs: Observation
  
  action, att_weights = agent.act(obs)
  print(f"{att_weights = }")
  
  action = action.squeeze(0)
  # print(f"{action.shape = }")
  obs, reward, done = task_env.step(action)
  gripper = Object.get_object("Panda_gripper")
  target = Object.get_object("target")
  
  # vis_score = check_visibility("cam_wrist", "target")
  
  # print(f"{vis_score = }")
  
  
  distance = np.linalg.norm(gripper.get_position() - target.get_position())
  distances.append(distance)
  # print(f"{done = }")
  
  count += 1
  if count == 100:break
  
print(f"{f"Done Successfull! done in {count} steps" if done else "Failed!"}")
print(f"Final distance: {distances[-1]}")

# %% 
# Single Step
obs: Observation
action = agent.act(obs).squeeze(0)
obs, reward, done = task_env.step(action)
gripper = Object.get_object("Panda_gripper")
target = Object.get_object("target")

vis_score = check_visibility("cam_wrist", "target", 0.5)
plt.imshow(obs.wrist_rgb)
plt.show()
print(f"{vis_score = }")


distance = np.linalg.norm(gripper.get_position() - target.get_position())
distances.append(distance)

if done:
  print(f"{f"Done Successfull! done in {count} steps" if done else "Failed!"}")
  print(f"Final distance: {distances[-1]}")

# checkImage(obs.wrist_rgb)
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
import torch
## more testin 
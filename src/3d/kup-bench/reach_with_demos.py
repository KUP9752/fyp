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
## Vision Experiments - Grasp
from rlbench.tasks.vision_static import VisionStatic as Vision_Static


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

import open3d as o3d
from matplotlib import pyplot as plt

from lib.agent import Agent
from lib.cam_type import CamType
from lib.policy_type import PolicyType

from lib.utils import get_task_name, now
from seed import set_seed
from pprint import pprint

set_seed()

num_demos = 10
cam_type = CamType.WRIST | CamType.RIGHT_SHOULDER


LABELS = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
# === Perception Modules ===
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
class Segmenter:
    def __init__(self, device='cpu'):
      self.model = maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
      self.device = device

    def segment(self, image):
        """
        image: HxWx3 RGB np.uint8
        returns: list of masks (boolean arrays) and bounding boxes
        """
        img_t: torch.Tensor = F.to_tensor(image).to(self.device)
        
        outputs = self.model([img_t.float()])[0]
        masks = (outputs['masks'] > 0.5).squeeze(1).cpu().numpy()
        boxes = outputs['boxes'].cpu().numpy()
        return masks, boxes, outputs



#%%
## 2. Create Environment and Set Model Name
# To use 'saved' demos, set the path below, and set live_demos=False


live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(True) ## important to get the data from the joints etc
cam_config = CameraConfig(rgb=True, depth=True, mask=True, point_cloud=True,
                              render_mode=RenderMode.OPENGL,
                    image_size=(64, 64))
nocam_config = CameraConfig(rgb=False, depth=False, mask=False,
                          render_mode=RenderMode.OPENGL)

obs_config.right_shoulder_camera = cam_config
obs_config.left_shoulder_camera = cam_config
obs_config.wrist_camera = cam_config



obs_config.overhead_camera = nocam_config
obs_config.front_camera = nocam_config

## active camera: wri camera


action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())

env = Environment(
    action_mode, DATASET, obs_config, False)
env.launch()

#%%
## 3. Attach Task and create Agent
pol_type = PolicyType.CAM_ATTENTION

task = Vision_Static
# task = ReachNoObs_Central
print(env.get_task.__code__.co_filename)

task_env = env.get_task(task_class = task, scale = 0.4)
try:
  target = Shape("target")
  target_rgb = torch.tensor(target.get_color())
except RuntimeError:
  print(f"'target' doesn't exist meaning this is a different task")
  target_rgb = None
  
agent = Agent(env.action_shape[0], pol_type, cam_type, target_rgb = target_rgb)

model_name = f"rwd-reach-{num_demos}-demos-{cam_type}-{get_task_name(task)}-{pol_type}--{now()}"
model_path = f"./all-models/reach-with-demos/{model_name}.pth"
print(model_name)
task
# %%
## 4. Request Demos
# demos = []
# for var in [0,1,2]:
#   task_env.set_variation(var)
#   demos += task_env.get_demos(1, live_demos=live_demos, random_selection = True)
demos: list[Demo] = task_env.get_demos(1, live_demos=live_demos)
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
  "lr": 1e-3,
  "shuffle_obs_in_demo": False,
  "shuffle_data": False,
  "lambda_attn": 1e-2,
}

ingest_num = num_demos


agent.ingest(demos[:ingest_num], **training_params) ## trains here
agent.save_model(model_path)

#%%
agent.policy.load_state_dict(torch.load("/home/kup/Desktop/code/fyp/src/3d/kup-bench/models/PROMISING-rwd-reach-1-demo-wrist+r_shoulder-ReachObs_Random-PolicyType.CAM_ATTENTION.pth"))

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
  
print({f"Done Successfull! done in {count} steps" if done else "Failed!"})
print(f"Final distance: {distances[-1]}")

# %% Reset Task Env


# %% Auto task Execution
seg = Segmenter()
def show_pc(arr_pc):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(arr_pc)
  o3d.visualization.draw_geometries([pcd])
  
def run_segmenter():
  with torch.no_grad():
    im = obs.wrist_rgb
    plt.imshow(im)
    image = im / 255
    print(image.dtype)
    
    masks, boxes, outs = seg.segment(image)
    print(f"{masks.shape = } {boxes.shape =}")
    print(f"{len(masks) = } {len(boxes) =}")
  
  # np.transpose(masks, (0, 1, 2))
  # plt.imshow(masks)

  im = Image.fromarray(im)
  draw = ImageDraw.Draw(im)

  pprint(outs, indent = 2)
  colours = ["red", "green", "blue", "orange", "purple", "yellow"]
  for i, box in enumerate(boxes):
    print(box)
    draw.rectangle(box, outline=colours[i % (len(boxes) - 1)], width=1)
    
  for i, label in enumerate(outs["labels"]):
    print(f"label {label}: {LABELS[label]}, col: {colours[i % (len(boxes) - 1)]}")
  # print(f"score {label}")
    
  plt.imshow(im)

agent.policy.to("cpu")
# task_env = env.get_task(ReachNoObs_Central)
_, obs = task_env.reset()
count = 0
done = False
distances = []
# %% 
# Single Step
obs: Observation
action, att_weights = agent.act(obs)
print(f"{att_weights = }")
  
action = action.squeeze(0)
signal = sim.simGetFloatSignal("wrist_target_vis_binary")
print(f"{signal = }")
task_env

# action = torch.tensor([0.0, 0, 0, 0., 0, 0.0, 0.0, 1.0])
print(f"{action.shape = }")

# run_segmenter()
obs, reward, done = task_env.step(action)
gripper = Object.get_object("Panda_gripper")
target = Object.get_object("target")

pc = obs.wrist_point_cloud
print(f"{pc.shape = }")
plt.imshow(obs.wrist_rgb)
plt.imshow(obs.wrist_depth)
show_pc(pc.reshape(-1, 3))
  

distance = np.linalg.norm(gripper.get_position() - target.get_position())
distances.append(distance)

if done:
  print({f"Done Successfull! done in {count} steps" if done else "Failed!"})
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
ts = []
tensor = torch.tensor([1,2,3])
for _ in range(3):
  ts.append(tensor)

torch.stack(ts, dim=0).mean(dim=0, dtype=torch.float32)

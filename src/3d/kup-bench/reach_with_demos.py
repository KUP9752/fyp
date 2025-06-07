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
from seed import set_seed
from pprint import pprint

set_seed()

num_demos = 10
# cam_type = CamType.WRIST #| CamType.WRIST_DEPTH

#%%
## 2. Create Environment and Set Model Name
# To use 'saved' demos, set the path below, and set live_demos=False

live_demos = False
DATASET = '' if live_demos else 'data/20demos'

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


## added cam config to be able to save demos
obs_config.overhead_camera = cam_config
obs_config.front_camera = cam_config 


action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())

env = Environment(
    action_mode, DATASET, obs_config, False)

env.launch()
#%%
## 3. Attach Task and create Agent
pol_type = PolicyType.DEPTH_GRASP

cam_type = CamType.WRIST | CamType.WRIST_DEPTH

# task = Vision_Random
task = Vision_Random
print(env.get_task.__code__.co_filename)

task_params = {
  "scale": 1.,
  "wrist_cam_distance": 0.2
}

smaller_task_params = {
  "scale": 0.5,
  "wrist_cam_distance": 0.3
}

task_env = env.get_task(task_class = task, **smaller_task_params)
# task_env = env.get_task(task_class = task)
target_name = "grasp_cube"


try:
  target = Shape(target_name)
  target_rgb = torch.tensor(target.get_color())
except RuntimeError:
  print(f"'target' doesn't exist meaning this is a different task")
  target_rgb = None
  
## simple policy
# agent = Agent(env.action_shape[0], pol_type, cam_type, target_rgb = target_rgb)
##depth grasp agent
wd = CamType.WRIST | CamType.WRIST_DEPTH
all_cams = CamType.LEFT_SHOULDER | CamType.RIGHT_SHOULDER | CamType.WRIST_DEPTH

# agent = Agent(
#   env.action_shape[0],
#   policy_type=PolicyType.DEPTH_GRASP,
#   cam_type= wd, 
#   grasp_thresh = 0.5,
#   config = "depth_feats",
#   opts = {
#     "attn_deep_fuse": True,
#     "use_proprio": True
#   }
# )

agent = Agent(
  env.action_shape[0],
  policy_type = PolicyType.DEPTH_GRASP,
  cam_type = CamType.WRIST | CamType.WRIST_DEPTH,
  grasp_thresh = 0.3,
  config = "depth_ch",
  opts = {}
)


agent = Agent(
  env.action_shape[0],
  policy_type=PolicyType.RNN_GRASP,
  cam_type = wd,
  rnn_opts = {
    "config": "depth_feats",
    # "attn_opts": {"is_deep_fuse": True},
    "use_proprio": False
  }
)

agent = Agent(
  action_shape= env.action_shape[0],
  policy_type=PolicyType.SIMPLE,
  cam_type=CamType.WRIST
)

model_name = f"rwd-{get_task_name(task)}-{agent}--{now()}"
model_path = f"./all-models/reach-with-demos/{model_name}.pth"
print(model_name)
agent.policy

# %%d
#%%
## 4. Request Demos
# demos = []#
# for var in [0,1,2]:
#   task_env.set_variation(var)
#   demos += task_env.get_demos(1, live_demos=live_demos, random_selection = True)
task = Vision_Random
env._dataset_root = "data/20demos/Vision_Random-size:1-dist:0_6"
task_env = env.get_task(task_class = task, **task_params)
demos: list[Demo] = task_env.get_demos(1, live_demos=False)
# test_demos: list[Demo] = task_env.get_demos(10, live_demos=live_demos)
demos

#%%
normal = {
  "scale": 1.,
  "wrist_cam_distance": 0.6
}

smaller = {
  "scale": 0.5,
  "wrist_cam_distance": 0.3
}

task = Vision_Random
task_env = env.get_task(task, **normal)
task_env.reset()
obs, _ , _ =task_env.step([0] * 8)
#%%
from rlbench.dataset_generator import save_demo

for i, demo  in enumerate(demos):
   save_demo(demo, f"data/demos20-small-{get_task_name(task)}/{i}")


# # print(f"What is in the demos: {type(demos)} | {type(demos[0])}")
# # print(f"{demos = } | {type(demos) = } | {len(demos) = }")
# print(f"Observations len: {list(map(len, demos))}")
# %%
## 5. Train
training_params = {
  "epochs": 800,
  "minibatch_size": 10,
  "lr": 1e-3,
  "shuffle_obs_in_demo": False,
  "shuffle_data": True,
  "lock_loader_seed": 1,
  "dataset_to_use": "demo",
  "lambda_grasp_loss": 1
}

ingest_num = 10

print(f"-> Using {ingest_num} demos")

agent.ingest(demos[:ingest_num], **training_params) ## trains here
agent.save_model(model_path)

#%%
# load_st#r = "models/grasp/rwd-reach-10-demos-wrist-Vision_Static-simple_grasp_policy--worked-with-scale0.4-dist-0.5_May09_14-57.pth"
# load_str = "models/grasp/rwd-reach-10-demos-wrist-Vision_Random-simple_grasp_policy--_May12_14-02.pth"


## Vision_Static | l_shoulder
## interesting l_shoulder only model claps before grabbing
# load_str = "/all-models/task-Vision_Static-demo-1-cam-l_shoulder--_May11_16-24.pth"

# load_str = "models/grasp/very-good-simple-grasp--task-Vision_Random-demo-10-cam-wrist--demo_dataset-10_batch-2000 epochs.pth"
# load_str = "all-models/reach-with-demos/rwd-Vision_Random-agent-policy:depth_grasp_policy-cams:wrist+wrist_depth-policy:depth_grasp_policy-config:attn-opts:{'gated_fuse': True, 'attn_num_heads': 8, 'attn_deep_fuse': True, 'use_proprio': True, 'proprio_opts': {}}--_May27_14-44.pth"
# agent.policy.load_state_dict(torch.load(f"/home/kup/Desktop/code/fyp/src/3d/kup-bench/{load_str}"))
#%%
task_env = env.get_task(task, )
_, obs = task_env.reset()

obs.left_shoulder_mask.shape
# %% 
# Auto task Execution
agent.policy.to("cpu")
# task_env = env.get_task(ReachNoObs_Central)
dones = 0
# _, obs = task_env.reset()
for demo in range(len(demos)):
  _, obs = task_env.reset_to_demo(demos[demo])
# for demo in test_demos:
#   _, obs = task_env.reset_to_demo(demo)
  count = 0
  done = False
  distances = []
  while not done:
    obs: Observation
    
    action, rets = agent.act(obs)
    # rgb_out = rets["rgb_fused"]
    # rgb_unfsuedout = rets["rgb_unfused"]
    # depth_out = rets["depth_fused"]

    # plt.imshow(obs.wrist_depth)
    # plt.imshow(obs.wrist_rgb)
    # plot_featmap(rgb_out, "RGB feats (fused w/ attn)")
    # plot_featmap(rgb_unfsuedout, "RGB feats (not fused)")
    # plot_featmap(depth_out)

    action = action.squeeze(0)
    # print(f"{action.shape =}")
    # print(f"{action[-1] =}")
    # print(f"{action.shape = }")
    obs, reward, done = task_env.step(action)
    gripper = Object.get_object("Panda_gripper")
    target = Object.get_object(target_name)

    
    
    # vis_score = check_visibility("cam_wrist", "target")
    
    # print(f"{vis_score = }")
    
    
    distance = np.linalg.norm(gripper.get_position() - target.get_position())
    distances.append(distance)
    # print(f"{done = }")
    
    count += 1
    if done:
      dones += 1
    if count == 100:
      break
    
  print(f"Done Successfull! done in {count} steps" if done else "Failed!")
  print(f"Final distance: {distances[-1]}")

print(f"Success = {dones}/{len(demos)}")
  

# %% Reset Task Env
# %% Auto task Execution
def show_pc(arr_pc):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(arr_pc)
  o3d.visualization.draw_geometries([pcd])
  

def plot_point_cloud(pcd):

  # Assume `point_cloud` is your (H, W, 3) NumPy array from the simulator
  # Example: Uncomment and set your actual point_cloud array
  # point_cloud = your_simulator.get_point_cloud()  # shape (H, W, 3)

  # For demonstration, creating a random point cloud (remove in real use)
  # point_cloud = np.random.uniform(-1, 1, size=(480, 640, 3))

  # Flatten to Nx3
  points = pcd.reshape(-1, 3)

  # Optional: subsample to ~100k points for performance
  subsample_rate = max(1, len(points) // 100000)
  points_sub = points[::subsample_rate]

  # Create interactive Plotly 3D scatter
  fig = go.Figure(data=[go.Scatter3d(
      x=points_sub[:, 0],
      y=points_sub[:, 1],
      z=points_sub[:, 2],
      mode='markers',
      marker=dict(size=1, opacity=0.6)
  )])

  fig.update_layout(
      scene=dict(
          xaxis_title='X',
          yaxis_title='Y',
          zaxis_title='Z'
      ),
      title='Interactive 3D Point Cloud'
  )
  fig.show()
#%% Single Step Setup
agent.policy.to("cpu")
task_env = env.get_task(ReachNoObs_Central)
_, obs = task_env.reset()
plt.imshow(obs.wrist_rgb)
count = 0
done = False
distances = []
# %% 
#%%
from scipy.spatial.transform import Rotation as R
# Single Step
import cv2
obs: Observation
action, rets = agent.act(obs)
action = action.squeeze(0)
print(f"{obs.wrist_point_cloud.shape = }")
robot_pos = env._robot.gripper.get_position()

print(f"{robot_pos = }")

poses = sample_camera_poses_quat(robot_pos)
poses = np.array(poses)
print(f"{poses = }")
print(f"{env._robot.gripper.get_pose() = }")
print(f"{poses[0][:3]}")

# moves = env._robot.arm.solve_ik_via_sampling(poses[0][:3], quaternion=poses[0][3:])
# task_env(move)
#%%

for i in range(10):

  robot_pos = env._robot.gripper.get_position()
  print(f"{robot_pos = }")
  
  poses = sample_camera_poses_quat(robot_pos)
  poses = np.array(poses)
  moves = env._robot.arm.solve_ik_via_jacobian(poses[0][:3], quaternion=poses[0][3:])
  for move in moves:
     task_env.step(move)

#%%
point_cloud = obs.wrist_point_cloud

hsv = cv2.cvtColor(obs.wrist_rgb, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(hsv, np.array([0, 177, 0]), np.array([179, 255, 255]))
mask = mask.astype(bool)
im = np.zeros_like(hsv)
print(f"{mask.shape = }")
im[mask] = hsv[mask]
plt.imshow(cv2.cvtColor(im, cv2.COLOR_HSV2RGB_FULL))
# plt.imshow(im)

pcd_flat = point_cloud.reshape(-1, 3)
# print(f"{pcd_flat.shape = }")
mask_flat = mask.reshape(-1).astype(bool)
# print(f"{mask_flat.shape = }")

# # plt.imshow(hsv[mask])
# print("before")
plot_point_cloud(pcd_flat)
print(f"{pcd_flat.shape = }")


masked_pcd = pcd_flat[mask_flat]

# print("after")
plot_point_cloud(masked_pcd)
print(f"{masked_pcd.shape = }")
print(f"{len(masked_pcd) = }")
# print(f"{masked_pcd.shape = }")



# Display in notebook




# rgb_attn = rets["rgb_attn_weights"]
# depth_attn = rets["depth_attn_weights"]

# print(f"{rgb_attn.shape = }")
# print(f"{depth_attn.shape = }")

  
# final_attn(rgb_attn)
# plt.imshow(obs.wrist_rgb)

# final_attn(depth_attn)
# plt.imshow(obs.wrist_depth)
# plot_attn_bar(rgb_attn, 0, 0, 0, 64)

# action = torch.Tensor([
#     0.,
#     0.,
#     0.,
#     0.,
#     0.,
#     0.,
#     0.,
#     0.
#   ])

print(f"{action.shape = }")
#%% Tuning the HSV thresholds
import cv2
def nothing(x):
    pass

# 1. Create a window
cv2.namedWindow('HSV Tuner', cv2.WINDOW_NORMAL)

# 2. Create six trackbars for lower/upper H, S, V
for name, val, maxval in [
    ('Low H',   0, 179),
    ('High H', 179, 179),
    ('Low S',   0, 255),
    ('High S', 255, 255),
    ('Low V',   0, 255),
    ('High V', 255, 255),
]:
    cv2.createTrackbar(name, 'HSV Tuner', val, maxval, nothing)

# 3. Capture from camera or load a test image/video
frame = cv2.imread("../../../im.png") # change 0 to path if using a video or image loop

# Create the window and trackbars
cv2.namedWindow('HSV Tuner', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Low H', 'HSV Tuner', 0, 179, nothing)
cv2.createTrackbar('High H', 'HSV Tuner', 179, 179, nothing)
cv2.createTrackbar('Low S', 'HSV Tuner', 0, 255, nothing)
cv2.createTrackbar('High S', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('Low V', 'HSV Tuner', 0, 255, nothing)
cv2.createTrackbar('High V', 'HSV Tuner', 255, 255, nothing)
# Selected HSV range:
# Lower: [  0 255   0]
# Upper: [  0 255 253]

## from wrist camera
# Selected HSV range:
# Lower: [  0 177   0]
# Upper: [179 255 255]

while True:
    # Safely read the trackbar positions
    low_h = cv2.getTrackbarPos('Low H', 'HSV Tuner')
    high_h = cv2.getTrackbarPos('High H', 'HSV Tuner')
    low_s = cv2.getTrackbarPos('Low S', 'HSV Tuner')
    high_s = cv2.getTrackbarPos('High S', 'HSV Tuner')
    low_v = cv2.getTrackbarPos('Low V', 'HSV Tuner')
    high_v = cv2.getTrackbarPos('High V', 'HSV Tuner')

    lower_hsv = np.array([low_h, low_s, low_v])
    upper_hsv = np.array([high_h, high_s, high_v])

    # Convert image to HSV
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    combined = np.hstack((frame, result))
    cv2.imshow('HSV Tuner', combined)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        print("Selected HSV range:")
        print("Lower:", lower_hsv)
        print("Upper:", upper_hsv)
        break

cv2.destroyAllWindows()

#%%
# run_segmenter()
obs, reward, done = task_env.step(action)
gripper = Object.get_object("Panda_gripper")
target = Object.get_object("target")

  

distance = np.linalg.norm(gripper.get_position() - target.get_position())
distances.append(distance)

if done:
  print({f"Done Successfull! done in {count} steps" if done else "Failed!"})
  print(f"Final distance: {distances[-1]}")


# %% 
## !! Finish
env.shutdown()

# %%

#%% Random Testing Cell

#%%


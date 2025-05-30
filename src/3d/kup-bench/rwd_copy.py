#%%
## 1. Initalise and Import
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig

import cv2
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


action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())

env = Environment(
    action_mode, DATASET, obs_config, False)
env.launch()
#%%
## 3. Attach Task and create Agent
pol_type = PolicyType.DEPTH_GRASP

cam_type = CamType.WRIST | CamType.WRIST_DEPTH

task = Vision_Random
# task = ReachNoObs_Central
print(env.get_task.__code__.co_filename)

task_params = {
  "scale": 1.,
  "wrist_cam_distance": 0.6
}

smaller_task_params = {
  "scale": 0.5,
  "wrist_cam_distance": 0.8
}

task_env = env.get_task(task_class = task, **task_params)
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

agent = Agent(
  env.action_shape[0],
  policy_type=PolicyType.DEPTH_GRASP,
  cam_type= wd, 
  grasp_thresh = 0.5,
  config = "depth_feats",
  opts = {
    "attn_deep_fuse": True,
    "use_proprio": True
  }
)

# agent = Agent(
#   env.action_shape[0],
#   pol_type, cam_type,
#   # grasp_thresh = 0.5,
#   config = "depth_feats",
#   opts = {
#     "gated_fuse": True, 
#     "resnet_name": "resnet18",
#     "kernel_size": 3
#   }
# )
agent = Agent(
  env.action_shape[0],
  policy_type=PolicyType.RNN_GRASP,
  cam_type = wd,
  rnn_opts = {
    "config": "attn",
    "attn_opts": {"is_deep_fuse": True},
    "use_proprio": True
  }
)


model_name = f"rwd-{get_task_name(task)}-{agent}--{now()}"
model_path = f"./all-models/reach-with-demos/{model_name}.pth"
print(model_name)
agent.policy

# %%
task_env = env.get_task(task_class = task, **task_params)
task_env.reset()
#%%
## 4. Request Demos
# demos = []#
# for var in [0,1,2]:
#   task_env.set_variation(var)
#   demos += task_env.get_demos(1, live_demos=live_demos, random_selection = True)
demos: list[Demo] = task_env.get_demos(10, live_demos=live_demos)
# test_demos: list[Demo] = task_env.get_demos(10, live_demos=live_demos)
demos

# # print(f"What is in the demos: {type(demos)} | {type(demos[0])}")
# # print(f"{demos = } | {type(demos) = } | {len(demos) = }")
# print(f"Observations len: {list(map(len, demos))}")
# %%
## 5. Train
training_params = {
  "epochs": 400,
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
task_env = env.get_task(task,  **task_params)
#%% 
def plot_featmap(feat, title_add=""):
  # Assume rgb_out is a tensor of shape (1, 128, 8, 8)
  feat_maps = feat.squeeze(0)  # shape: (128, 8, 8)

  # Choose how many feature maps you want to visualize (e.g., first 32)
  num_maps = 32
  maps_to_show = feat_maps[:num_maps]

  # Plot in a grid
  n_cols = 8
  n_rows = num_maps // n_cols

  plt.figure(figsize=(n_cols * 2, n_rows * 2))
  for i in range(num_maps):
      plt.subplot(n_rows, n_cols, i + 1)
      plt.imshow(maps_to_show[i].detach().cpu().numpy(), cmap='viridis')
      plt.title(f'Ch {i}')
      plt.axis('off')

  plt.suptitle(f"First 32 Channels of give feats, {title_add}")
  plt.tight_layout()
  plt.show()
  
# %% 
# Auto task Execution
agent.policy.to("cpu")
# task_env = env.get_task(ReachNoObs_Central)
dones = 0
# _, obs = task_env.reset()
for demo in range(len(demos)):
  _, obs = task_env.reset()
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
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
images = [obs.wrist_rgb, obs.left_shoulder_rgb, obs.right_shoulder_rgb]
for ax, img in zip(axs, images):
  ax.imshow(img)
  ax.set_title("")
  ax.axis("off")
plt.savefig("images.png", bbox_inches="tight")

plt.close()
type(axs)

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

import plotly.graph_objects as go
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
#%%
agent.policy.to("cpu")
task_env = env.get_task(ReachNoObs_Central)
_, obs = task_env.reset()

count = 0
done = False
distances = []
def plot_attn(attn_weights, sample, head):
  B, H, N, _ = attn_weights.shape

  # attention map from token i to j
  attn_map = attn_weights[sample, head].detach().cpu().numpy()  # (N, N)

  plt.figure(figsize=(6, 6))
  plt.imshow(attn_map, cmap='viridis')
  plt.title(f'Attention map (sample {sample}, head {head})')
  plt.xlabel('Key token')
  plt.ylabel('Query token')
  plt.colorbar()
  plt.tight_layout()
  plt.show()

def final_attn(weights):
  # weights: (1, num_heads, 64, 64)
  attn_weights = weights[0]  # Remove batch dim → (8, 64, 64)

  # You likely have a flattened 8x8 feature map → reshape each 64 to (8, 8)
  H = W = 8  # assuming square spatial shape from 8x8 features

  fig, axs = plt.subplots(2, 4, figsize=(16, 8))

  for i in range(8):  # loop over 8 heads
      # Get attention map for head i: shape (64, 64)
      attn_map = attn_weights[i].detach().cpu()  # (64, 64)

      # Let's pick one query location to visualize attention from that point to all keys
      query_index = 32  # center pixel
      attention_from_query = attn_map[query_index]  # (64,)
      attention_2d = attention_from_query.reshape(H, W)  # (8, 8)

      ax = axs[i // 4, i % 4]
      im = ax.imshow(attention_2d, cmap='viridis')
      ax.set_title(f"Head {i}")
      ax.axis('off')
      fig.colorbar(im, ax=ax)

  plt.suptitle("Attention from center pixel (index 32) across 8 heads")
  plt.tight_layout()
  plt.show()

def plot_attn_bar(attn_weights, sample, token_id, head, grid_size):

  attn_map = attn_weights[sample, head, token_id]  # shape: (N,)
  attn_grid = attn_map.view(grid_size, grid_size)

  plt.imshow(attn_grid, cmap='plasma')
  plt.title(f'Where token {token_id} attends (Head {head})')
  plt.colorbar()
# %% 
agent.policy
#%%
# Single Step
obs: Observation
action, rets = agent.act(obs)
action = action.squeeze(0)
print(f"{obs.wrist_point_cloud.shape = }")


point_cloud = obs.wrist_point_cloud

hsv = cv2.cvtColor(obs.wrist_rgb, cv2.COLOR_RGB2HSV)

mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

pcd_flat = point_cloud.reshape(-1, 3)
mask_flat = mask.reshape(-1).astype(bool)



plot_point_cloud(point_cloud)

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
#%%
import cv2
import numpy as np

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
cap = cv2.VideoCapture(obs.wrist_rgb)  # change 0 to path if using a video or image loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Read trackbar positions
    lh = cv2.getTrackbarPos('Low H',   'HSV Tuner')
    hh = cv2.getTrackbarPos('High H',  'HSV Tuner')
    ls = cv2.getTrackbarPos('Low S',   'HSV Tuner')
    hs = cv2.getTrackbarPos('High S',  'HSV Tuner')
    lv = cv2.getTrackbarPos('Low V',   'HSV Tuner')
    hv = cv2.getTrackbarPos('High V',  'HSV Tuner')

    # 5. Convert to HSV and threshold
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([lh, ls, lv])
    upper = np.array([hh, hs, hv])
    mask = cv2.inRange(hsv, lower, upper)

    # 6. Show result
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    combined = np.hstack([frame, masked])
    cv2.imshow('HSV Tuner', combined)

    # 7. Quit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
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
from modules.dataset.demo_dataset import DemoDataset
from modules.dataset.demo_obs_dataset import  DemoObsDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    ## batch contains [(input, labels)] where each input is a complete demo (in terms of the data in sequence rgb for example)
    ## input: (t, ch, w, h) 
    inputs, labels = zip(*batch)
    [print(f"{input.shape}") for input in inputs]

    ## enforcing types for later
    print(f"{len(labels) =}")
    
    real_lengths = torch.LongTensor([inp.shape[0] for inp in inputs])
    print(f"{real_lengths.shape =}")
    
    inputs_padded = pad_sequence(inputs, batch_first=True) ## CHECK: if it gives (B, t, ch, w, h)
    print(f"{inputs_padded.shape = }")

    labels_padded = pad_sequence(labels, batch_first=True) ## CHECK: if it gives (B, t, ch, w, h)
    print(f"{labels_padded.shape = }")


    ## need to return shape (B, t, ch, w, h) for the input and labels
    ## also returning lenths for LSTM use later
    return inputs_padded, labels_padded, real_lengths

cam_type = CamType.WRIST
# dataset = DemoObsDataset(demos, cam_type= cam_type | CamType.WRIST_DEPTH, get_type="cat", shuffle_obs=True)
dataset = DemoDataset(demos, cam_type= cam_type, get_type="cat")
loader = DataLoader(
  dataset,
  shuffle = True,
  batch_size=10,
  collate_fn=collate,
  generator=torch.manual_seed(42)
)
count = 0
print("hello")
for inputs, labels, lengths in loader:
  pred_action = torch.rand((2, 8))
  print(f"{lengths.shape = }")
  print(f"{labels.shape = }")
  print(f"{inputs.size(0) = }")
  sdd = torch.arange(inputs.size(0))
  print(f"{sdd.shape = }")
  print(f"{sdd = }")
  
  idx = lengths - 1
  true_labels = labels[torch.arange(inputs.size(0)), idx]
  # true_labels = labels[:, idx]
  print(f"{true_labels.shape =}")
  

  print()

#%%
import numpy as np
import torch

t = torch.empty((12, 128, 2, 2))
t2 = torch.empty((12, 128, 2, 2))
c = torch.cat([t, t2], dim = 1)
c.shape

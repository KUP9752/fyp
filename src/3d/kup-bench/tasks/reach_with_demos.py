import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks.reach_target_no_obs  import ReachTargetNoObs
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from pyrep.const import RenderMode

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm as progress

class Policy(nn.Module):
  def __init__(self, action_shape):
    super(Policy, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
    )
    flat_size = 2 * 2 * 128
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(flat_size, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 50),
      nn.ReLU(inplace=False),
      nn.Linear(50, action_shape)
    )
    
  def forward(self, image):
    feats = self.conv(image)
    return self.fc(feats)
  
  def train(self, 
            demos: list[Demo],
            epochs: int = 100,
            batch_size: int = 2,
            lr: float = 0.001,
            model_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loader = DataLoader(demos, batch_size = batch_size, shuffle = True);
    model = self.to(device)
    
    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr = lr)
    
    model.train()
    self.losses = [0 for _ in range(epochs)]
    for epoch in progress(range(epochs)):
      running_loss = 0
      for obs in loader:
        obs: Observation
        inputs, labels = obs.wrist_rgb, obs.joint_velocities
        inputs, labels = inputs.to(device), labels.to(device)
        optimiser.zero_grad()
        pred_actions = model(inputs)
        loss = loss_fn(pred_actions, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
      loss = running_loss / len(loader)
      self.losses[epoch] = loss
    print(f"Done Training Policy on Demos")
    
    if model_path:
      torch.save(self.state_dict(), f"{model_path}")

class Agent(object):

    def __init__(self, action_shape):
      self.action_shape = action_shape
      self.policy = Policy(action_shape)

    def ingest(self, demos: list[Demo]):
      self.policy.train(demos)

    def act(self, obs:  Observation):
      # arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
      # gripper = [1.0]  # Always open
      # return np.concatenate([arm, gripper], axis=-1)
      self.policy.eval()
      with torch.no_grad():
        pred = self.policy(obs.wrist_rgb)
        


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(False)
cam_config = CameraConfig(rgb=True, depth=False, mask=False,
                              render_mode=RenderMode.OPENGL,
                              image_size=(64, 64))
nocam_config = CameraConfig(rgb=False, depth=False, mask=False,
                          render_mode=RenderMode.OPENGL)
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.right_shoulder_camera = nocam_config
obs_config.left_shoulder_camera = nocam_config
obs_config.overhead_camera = nocam_config
obs_config.front_camera = nocam_config

## active camera
obs_config.wrist_camera = cam_config


action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(
    action_mode, DATASET, obs_config, False)
env.launch()

task = env.get_task(ReachTargetNoObs)
demos: list[Demo] = task.get_demos(2, live_demos=live_demos)

agent = Agent(env.action_shape)
agent.ingest(demos) ## trains here

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

print('Done')
env.shutdown()

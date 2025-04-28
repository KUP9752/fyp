from typing import Literal
from torch.utils.data import Dataset

import torch
import numpy as np

from rlbench.demo import Demo
from modules.cam_type import CamType
from seed import SEED



class DemoObsDataset(Dataset):
  def __init__(self,
    demos: list[Demo],
    cam_type: CamType,
    shuffle_obs: bool,
    get_type: Literal["cat", "stack"]
  ):
    
    if get_type not in ["cat", "stack"]:
      raise ValueError("[demo_obs_dataset] 'get_type' is assigned an incorrect option")
    self.get_type = get_type
    
    self.cam_type = cam_type
    self.all_data = []
    rng = np.random.default_rng(SEED)
    for demo in demos:
      obss = demo._observations
      # print(f"[loader] Observations len: {len(obss)}")
      if shuffle_obs:
        rng.shuffle(obss)
      
      self.all_data.extend(obss)
      
  def __len__(self):
      return len(self.all_data)

  def __getitem__(self, idx):
    obs = self.all_data[idx]
    ## NOTE: Hard coded only using 3 cameras currently
    images = []
    ## TODO: add some transformations and other augmentations to make generalisation better?
    if self.cam_type & CamType.WRIST:
      # print(f"Using Wrist Image")
      wrist_image = torch.tensor(obs.wrist_rgb, dtype = torch.float32)
      wrist_image = torch.permute(wrist_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
      images.append(wrist_image)
    if self.cam_type & CamType.LEFT_SHOULDER:
      # print(f"Using L Shouulder Image")
      ls_image = torch.tensor(obs.left_shoulder_rgb, dtype = torch.float32)
      ls_image = torch.permute(ls_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
      images.append(ls_image)
    if self.cam_type & CamType.RIGHT_SHOULDER:
      # print(f"Using R Shoulder Image")
      rs_image = torch.tensor(obs.right_shoulder_rgb, dtype = torch.float32)
      rs_image = torch.permute(rs_image, (2, 0, 1))   ## 64, 64, 3  -> 3, 64, 64
      images.append(rs_image)
    
    if not images:
      raise ValueError("[demo_obs_dataset] - DemoObsDataSet - __getitem__] No images selected !")
      
    ## this allows multi rgb cameras   
    if self.get_type == "cat":
      inputs = torch.cat(images, dim = 0)  ## cat on the colours channel
      ## inputs shape should now be (3 * num_cams, 64, 64)
    elif self.get_type == "stack":
      inputs = torch.stack(images, dim = 0)
      ## inputs shape should now be (num_cams, 3, 64, 64)
    labels = np.append(obs.joint_velocities, obs.gripper_open)
    labels = torch.tensor(labels, dtype = torch.float32)
    
    return inputs, labels # type: ignore (unbound 'inputs' will raise in `__init__`)

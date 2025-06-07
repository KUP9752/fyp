from typing import Literal
from torch.utils.data import Dataset

import torch
import numpy as np

from rlbench.demo import Demo
from lib.cam_type import CamType
from seed import SEED

from lib.utils import pick_obs_from_cam

## this is pretty much useless currently not updating its
class DemoObsDataset(Dataset):
  def __init__(self,
    demos: list[Demo],
    cam_type: CamType,
    shuffle_obs: bool,
    get_type: Literal["cat", "stack"],
    rgb_transform = None,
  ):
    
    if get_type not in ["cat", "stack"]:
      raise ValueError("[demo_obs_dataset] 'get_type' is assigned an incorrect option")
    self.get_type = get_type
    self.rgb_transform = rgb_transform
    
    self.cam_type = cam_type
    self.all_data = []
    rng = np.random.default_rng(SEED)
    for demo in demos:
      obss = demo._observations
      # print(f"[loader] Observations len: {len(obss)}")
      if shuffle_obs:
        rng.shuffle(obss)
      
      self.all_data.extend(obss)
      
  ## this is length of all observations so all the data
  def __len__(self):
      return len(self.all_data)

  def __getitem__(self, idx):
    obs = self.all_data[idx]
    # print(f"[data_obs_dataset - (getitem)] index selevted: {idx}")
    
    ## NOTE: Hard coded only using 3 cameras currently
    images = []
    ## TODO: add some transformations and other augmentations to make generalisation better?
    
    ## wrist -> ls -> rs
    # for ct in CamType.uniques():
    for ct in CamType.main4():
      if self.cam_type & ct:
        image = torch.tensor(pick_obs_from_cam(ct, obs, normalise_rgb=True), dtype= torch.float32)
        image = torch.permute(image, (2, 0, 1)) ## 64, 64, 3 -> 3, 64, 64
        ## TODO: depth and depth transform?
        if self.rgb_transform:
          image = self.rgb_transform(image)
          
        images.append(image)
      
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
    
    ## //NOTE: for downstream models/policies I am preserving the order of the selected camtypes
    ## its always WRIST > LEFT_SHOULDER > RIGHT_SHOULDER > ... (if they exist, otherwise miss the early ones)
    
    return inputs, labels, {} # type: ignore (unbound 'inputs' will raise in `__init__`)

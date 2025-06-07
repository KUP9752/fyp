from typing import Literal
from torch.utils.data import Dataset

import numpy as np

import torch
from rlbench.demo import Demo
from rlbench.backend.observation import Observation
from lib.cam_type import CamType

from lib.utils import pick_obs_from_cam, pick_joint_angles

class DemoDataset(Dataset):
  def __init__(self, 
      demos: list[Demo], 
      cam_type: CamType, 
      get_type: Literal["cat", "stack"],
      rgb_transform = None,
      use_proprio: bool = False,
      label_get: Literal["joint_velocities", "joint_positions"] = "joint_velocities"
  ):
    if get_type not in ["cat", "stack"]:
      raise ValueError(f"[demo_dataset - (init)] 'get_type' is assigned an incorrect option: {get_type}")
    
    self.get_type = get_type
    self.rgb_transform = rgb_transform
    self.use_proprio = use_proprio
    self.cam_type = cam_type
    self.label_get = label_get
    ## store in list of lists
    self.all_demos = []

    for demo in demos:
      obss = demo._observations
      self.all_demos.append(obss)

  def __len__(self):
    return len(self.all_demos)

  def __getitem__(self, idx):

    obss = self.all_demos[idx]
    ## this now handles batching: (demo_len, ... )

    # print(f"[demo_dataset (getitem)] index selected is {idx}")
    # print()
    

    # demo_len = len(obss)
    sequence = []
    seq_jangles = []
    seq_labels = []
    for obs in obss:
      obs: Observation
      ## === extract the image from obs
      images = [
        torch.permute(
          torch.tensor(
            pick_obs_from_cam(ct, obs, normalise_rgb=True),
            dtype=torch.float32
          ),
          (2, 0, 1)
        )
        for ct in CamType.main4() if self.cam_type & ct
        # for ct in CamType.uniques() if self.cam_type & ct
        ## NOTE doing main4 just to be safe as I never use the other 2
      ]

      if self.rgb_transform:
        images = map(self.rgb_transform, images)

      sequence.append(images)

      ## === extract the joint angles from the observation
      if self.use_proprio:
        seq_jangles.append(
          torch.tensor(
            pick_joint_angles(obs, normalise = True), 
            dtype=torch.float32
          )
        )

      ## === extract the label from obs
      seq_labels.append(
        torch.tensor(
          np.append(getattr(obs, self.label_get), obs.gripper_open), 
          dtype=torch.float32
        )
      )



    ## this allows multi rgb cameras   
    if self.get_type == "cat":
      batch = [torch.cat(entry, dim = 0) for entry in sequence]  ## cat on the colours channel
      ## batch shape should now be (3 * num_cams, 64, 64)
    elif self.get_type == "stack":
      batch = [torch.stack(entry, dim = 0) for entry in sequence]  ## cat on the colours channel
      ## batch shape should now be (num_cams, 3, 64, 64)
    else:
      raise ValueError(f"[demo_dataset - (getitem)]'get_type' is assigned an incorrect option: {self.get_type}")

    inputs = torch.stack(batch, dim = 0)
    labels = torch.stack(seq_labels, dim = 0)

    # print(f"[demo_dataset (getitem)] {inputs.shape = }")
    # print(f"[demo_dataset (getitem)] {labels.shape = }")

    proprio = None
    if self.use_proprio:
      proprio =torch.stack(seq_jangles, dim = 0)

    return inputs, labels, {"proprio": proprio}
  
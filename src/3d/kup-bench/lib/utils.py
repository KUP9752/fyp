from lib.cam_type import CamType

from rlbench.backend.observation import Observation

import numpy as np
from time import strftime

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
# from rlbench.tasks.vision_random import VisionRandom as Vision_Random

GRIPPER_CLOSE = 0.0
GRIPPER_OPEN = 1.0

def get_task_name(task) -> str:
  ## Reach No Obs
  if task == ReachNoObs_Central:
    return "Reach_Central"
  elif task == ReachNoObs_SideL:
    return "Reach_SideLeft"
  elif task == ReachNoObs_SideR:
    return "Reach_SideRight"
  elif task == ReachNoObs_PlaceRandom:
    return "Reach_PlaceRandom"
  ## Reach with Obstacles
  elif task == ReachObs_StaticLeft:
    return "ReachObs_StaticLeft"
  elif task == ReachObs_Static:
    return "ReachObs_Static"
  elif task == ReachObs_RandomStatic:
    return "ReachObs_RandomStatic"
  elif task == ReachObs_Random:
    return "ReachObs_Random"
  elif task == ReachObs_IndepRandom:
    return "ReachObs_IndRandom"
  ## Grasp
  elif task == Grasp_Simple:
    return "Grasp_Simple"
  elif task == Grasp_ThenMove:
    return "Grasp_ThenMove"
  ## Vision Experiments - Grasp
  elif task == Vision_Static:
    return "Vision_Static"
  # elif task == Vision_Random:
  #   return "Vision_Random"
  else:
    raise ValueError("[utils - get_task_name] Task not found!")

def pick_obs_from_cam(cam_type: CamType, obs: Observation, normalise_rgb: bool = True) -> np.ndarray:
  match cam_type:
    case CamType.WRIST:
      return (obs.wrist_rgb / 255) if normalise_rgb else obs.wrist_rgb
    case CamType.LEFT_SHOULDER:
      return (obs.left_shoulder_rgb / 255) if normalise_rgb else obs.left_shoulder_rgb
    case CamType.RIGHT_SHOULDER:
      return (obs.right_shoulder_rgb / 255) if normalise_rgb else obs.right_shoulder_rgb
    case _:
      raise ValueError(f"[utils - pick_obs_from_cam] Unknown CamType ({cam_type})")

def now(format = "_%B%d_%H-%M") -> str:
  return strftime(format)
  
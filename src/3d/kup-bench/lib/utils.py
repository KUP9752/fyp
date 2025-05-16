from lib.cam_type import CamType

from rlbench.backend.observation import Observation
from rlbench.demo import Demo

import numpy as np
from time import strftime

import pickle

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
  elif task == Vision_Random:
    return "Vision_Random"
  else:
    raise ValueError("[utils - get_task_name] Task not found!")

def ndarray_min_max_norm(arr: np.ndarray, eps = 1e-8) -> np.ndarray:
  min_val = arr.min()
  max_val = arr.max()
  return (arr - min_val) / (max_val - min_val + eps)

def pick_obs_from_cam(cam_type: CamType, obs: Observation, normalise_rgb: bool = True, normalise_depth: bool = True) -> np.ndarray:
  match cam_type:
    case CamType.WRIST:
      return (obs.wrist_rgb / 255) if normalise_rgb else obs.wrist_rgb
    case CamType.LEFT_SHOULDER:
      return (obs.left_shoulder_rgb / 255) if normalise_rgb else obs.left_shoulder_rgb
    case CamType.RIGHT_SHOULDER:
      return (obs.right_shoulder_rgb / 255) if normalise_rgb else obs.right_shoulder_rgb
    ## NOTE: depth normalisation is min-max here, if we want to keep the meanings of metres in the model, maybe use log etc??
    case CamType.WRIST_DEPTH:
      depth_arr = ndarray_min_max_norm(obs.wrist_depth) if normalise_depth else obs.wrist_depth 
      w, h = depth_arr.shape
      return np.reshape(depth_arr, (w, h, 1)) ## expand the channel dinemsion
    case _:
      raise ValueError(f"[utils - pick_obs_from_cam] Unknown CamType ({cam_type})")

def now(format = "_%B%d_%H-%M") -> str:
  return strftime(format)

def params_string(**params):
  s = "\n\t"
  for k, v in params.items():
    s+= f"{k} = {v},\n\t"
  return s

def save_demos(demos: list[Demo], filename: str) -> None:
  states = []
  for demo in demos:
    states.append(
      (demo.random_seed, demo._observations[0].misc["variation_index"])
    )
  with open(f"{filename}-{now()}", "wb") as f:
    pickle.dump(states, f)

## returns a FAKE demo (no observations) and the misc['variation_index'] for observations to restore state of the target object
def load_demos(filename: str) -> list[tuple[Demo, int]]:
  print(f"[task_utils - load_demos] WARNING! Loaded demos are *FAKE*, need to be restored by using `task_env.restore_to_demo()`")
  with open(f"{filename}", "rb") as f:
    states   = pickle.load(f)
  fake_demos = [(Demo([], seed, None), var) for seed, var in states]

  print(f"[task_utils - load_demos] WARNING! USE: `task_env.restore_to_demo(demo, obs_variation_index = var)`")
  return fake_demos
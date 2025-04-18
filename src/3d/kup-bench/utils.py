from typing import Literal, Optional, Type
import torch
import numpy as np

from policy import Agent, CamType

from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.backend.task import Task

from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from pyrep.objects import Object


# Tasks
## No Obstacle
from rlbench.tasks.reach_target_no_obs_side_r import ReachTargetNoObsSideR as ReachNoObs_SideR
from rlbench.tasks.reach_target_no_obs_side_l import ReachTargetNoObsSideL as ReachNoObs_SideL
from rlbench.tasks.reach_target_no_obs_central import ReachTargetNoObsCentral as ReachNoObs_Central
from rlbench.tasks.reach_target_no_obs import ReachTargetNoObs as ReachNoObs_PlaceRandom
## Obstacle
from rlbench.tasks.reach_target_obs_static_left import ReachTargetObsStaticLeft as ReachObs_StaticLeft
from rlbench.tasks.reach_target_obs_static import ReachTargetObsStatic as ReachObs_Static
from rlbench.tasks.reach_target_obs_random_static import ReachTargetObsRandomStatic as ReachObs_RandomStatic
from rlbench.tasks.reach_target_obs_random import ReachTargetObsRandom as ReachObs_Random
from rlbench.tasks.reach_target_obs_ind_random import ReachTargetObsIndRandom as ReachObs_IndepRandom
## Grasp
from rlbench.tasks.simple_grasp import SimpleGrasp as Grasp_Simple
from rlbench.tasks.grasp_and_move import GraspAndMove as Grasp_ThenMove



def set_seed(seed = 42):
  torch.manual_seed(seed)
  np.random.seed(seed)
  
  
## kup-bench helpers

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
  ## TODO:  
  else:
    raise ValueError("[utils - get_task_name] Task not found!")

def request_demos_and_train_for_task(
    env: Environment,
    current_task: Type[Task],
    agent: Agent,
    num_demos: int,
    save_model = False,
    live_demos = True, 
    **training_params
  ) -> tuple[TaskEnvironment, list[Demo]]:
  task_name = get_task_name(current_task) 
  
  ## load this task into env
  task_env = env.get_task(current_task) # removes all other loaded tasks
  ## num_demos defined in the 
  demos: list[Demo] = task_env.get_demos(num_demos, live_demos=live_demos)
  agent.ingest(demos, **training_params)
  
  if save_model:
    model_name = f"task-{task_name}-demo-{num_demos}-cam-{agent.cam_type}"
    model_path = f"./all-models/{model_name}.pth"
    agent.save_model(model_path)
  
  return task_env, demos

## Createes a newe agent and runs the task as given
## The distance metric seems to be only useful for reaching currently
def run_reach_task(
  env: Environment,
  task, ## any of Reach_* or ReachObs_* tasks
  cam_type: CamType,
  demo_count: int,
  max_eplen: int | Literal["demo_max"] = "demo_max",
  **training_params
) -> tuple[list[float], bool]:
  print(f"Training for {demo_count} demos for task: {get_task_name(task)}")
  ## new agent trained each time
  agent = Agent(env.action_shape[0], cam_type)
  
  ## request demos and train
  task_env, demos = request_demos_and_train_for_task(
    env,
    task,
    agent,
    demo_count,
    save_model=True,
    **training_params
  )
  ## if max len is not specified make it the max of the givem demo
  if max_eplen == "demo_max":
    max_eplen = max(list(map(len, demos)))
  
  ## evaluate
  obs: Observation
  _, obs = task_env.reset()
  agent.policy.to("cpu") # move to cpu
  distances = []
  done  = False
  
  for _ in range(max_eplen):
    action = agent.act(obs).squeeze()
    obs, reward, done = task_env.step(action)

    gripper = Object.get_object("Panda_gripper")
    target = Object.get_object("target")
    distance = np.linalg.norm(gripper.get_position() - target.get_position())
    distances.append(distance)
    if done:
      break
    
  return distances, done
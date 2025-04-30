from typing import Literal, Optional, Type

from time import strftime

import torch
import numpy as np

from lib.agent import Agent
from lib.cam_type import CamType
from lib.policy_type import PolicyType
from lib.utils import get_task_name, now

from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.backend.task import Task

from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from pyrep.objects import Object, Shape


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

  
## kup-bench helpers

def demos_and_train_for_task(
    env: Environment,
    current_task: type[Task],
    agent: Agent,
    given_demos: int | list[Demo],
    save_model = False,
    live_demos = True, 
    **training_params
  ) -> tuple[TaskEnvironment, list[Demo]]:
  task_name = get_task_name(current_task) 
  task_env = env.get_task(current_task) # removes all other loaded tasks
  
  if isinstance(given_demos, int):
    demo_count = given_demos
    print(f"1- Requesting {given_demos} demos for task: {task_name}")
    ## load this task into env
    ## num_demos defined in the 
    demos: list[Demo] = task_env.get_demos(demo_count, live_demos=live_demos)
    agent.ingest(demos, **training_params)
  elif isinstance(given_demos, list):
    print(f"2- Using given demos for task: {task_name}")
    demos = given_demos
    demo_count = len(demos)
    agent.ingest(given_demos, **training_params)
  
  
  if save_model:
    model_name = f"task-{task_name}-demo-{demo_count}-cam-{agent.cam_type}"
    model_path = f"./all-models/{model_name}--{now()}.pth"
    agent.save_model(model_path)
  
  return task_env, demos

## Createes a newe agent and runs the task as given
## The distance metric seems to be only useful for reaching currently
def run_reach_task(
  env: Environment,
  task, ## any of Reach_* or ReachObs_* tasks
  policy_type: PolicyType,
  cam_type: CamType,
  demos: int | list[Demo],
  max_eplen: int | Literal["demo_max"] = "demo_max",
  within_err_dist: Optional[float] = None, ## allows the execution to finish early depending on if an error around the target is reached
  **training_params
) -> tuple[list[float], bool]:
  ## new agent trained each time
  agent = Agent(env.action_shape[0], policy_type, cam_type)
  
  ## request demos and train
  
  task_env, demos = demos_and_train_for_task(
    env,
    task,
    agent,
    demos,
    save_model=True,
    **training_params
  )
    
  ## if max len is not specified make it the max of the givem demo
  if max_eplen == "demo_max":
    max_eplen = max(list(map(len, demos)))
  
  ## evaluate
  obs: Observation
  _, obs = task_env.reset()
  agent.policy.to("cpu") # move to cpu if not alr there
  distances = []
  done  = False
  
  for _ in range(max_eplen):
    action, pol_dict = agent.act(obs) # type: ignore
    action = action.squeeze()
    
    obs, reward, done = task_env.step(action)

    gripper = Object.get_object("Panda_gripper")
    target = Object.get_object("target")
    distance = np.linalg.norm(gripper.get_position() - target.get_position())
    distances.append(distance)
    
    ## doing separately to make pylance happy
    if within_err_dist is not None:
      done = done or distance <= within_err_dist
      
    if done: 
      break
    
  return distances, done

def run_reach_task_with_agent(
  env: Environment,
  task: Task, ## any of Reach_* or ReachObs_* tasks
  agent: Agent, 
  demos: list[Demo],
  max_eplen: int | Literal["demo_max"] = "demo_max",
  within_err_dist: Optional[float] = None, ## allows the execution to finish early depending on if an error around the target is reached
  **training_params
) -> tuple[dict, bool]:
  ## new agent trained each time
  
  ## request demos and train
  task_env, demos = demos_and_train_for_task(
    env,
    task, #type: ignore[arg-type]
    agent,
    demos,
    save_model=True,
    **training_params
  )
    
  ## if max len is not specified make it the max of the givem demo
  if max_eplen == "demo_max":
    max_eplen = max(list(map(len, demos)))
  
  ## evaluate
  obs: Observation
  _, obs = task_env.reset()
  
  obstacle = Shape("obstacle")
  
  agent.policy.to("cpu") # move to cpu if not alr there
  distances = []
  done  = False
  
  atts_before_obs = []
  atts_after_obs = []
  
  for _ in range(max_eplen):
    action, pol_dict = agent.act(obs) # type: ignore
    action = action.squeeze()
    
    ## get the attention weights
    atts = pol_dict["attention_weights"]
    
    ## if the z value (height) of arm is negative with respect to obstacle, then we are below
    
    if task_env._robot.arm.get_tip().get_position(relative_to=obstacle)[2] <= 0:
      ## below
      # print("BELOW THE OBS")
      # print(f"{atts = }")
      # print()
      atts_before_obs.append(atts)
    else:
      ## above
      # print("above THE OBS")
      # print(f"{atts = }")
      # print()
      atts_after_obs.append(atts)
    
    obs, reward, done = task_env.step(action)

    gripper = Object.get_object("Panda_gripper")
    target = Object.get_object("target")
    distance = np.linalg.norm(gripper.get_position() - target.get_position())
    distances.append(distance)
    
    ## doing separately to make pylance happy
    if within_err_dist is not None:
      done = done or distance <= within_err_dist
      
    if done: 
      break
    
  return {
    "distances": distances, 
    "avg_attentions_before_obstacle": torch.stack(atts_before_obs, dim=0).mean(dim=0, dtype=torch.float32),
    "avg_attentions_before_obstacle": torch.stack(atts_after_obs, dim=0).mean(dim=0, dtype=torch.float32),
    },  done  
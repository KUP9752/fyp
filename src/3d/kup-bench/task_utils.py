from typing import Literal, Optional, Type

from time import strftime
import os

from rlbench import CameraConfig, ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
import torch
import numpy as np

from lib.agent import Agent
from lib.cam_type import CamType
from lib.policy_type import PolicyType
from lib.utils import get_task_name, now, pick_obs_from_cam, GRIPPER_CLOSE, GRIPPER_OPEN

from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.backend.task import Task

from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from rlbench.dataset_generator import save_demo

from pyrep.objects import Object, Shape
from pyrep.const import RenderMode
import matplotlib.pyplot as plt

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
    training_params: dict = {},
    task_params: dict = {}
  ) -> tuple[TaskEnvironment, list[Demo]]:
  task_name = get_task_name(current_task) 
  task_env = env.get_task(current_task, **task_params) # removes all other loaded tasks
  
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
    model_name = f"policy-{agent.policy_type}-task-{task_name}-demo-{demo_count}-cam-{agent.cam_type}"
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
  training_params: dict = {},
  task_params: dict = {}
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
    training_params = training_params,
    task_params = task_params
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

def run_determined_reach_with_agent(
    env: Environment, 
    task: type[Task], 
    agent: Agent, 
    rec_demos: list[Demo], 
    max_eplen: int | Literal["demo_max"] = "demo_max",
    within_err_dist: Optional[float] = None,
    task_params: dict = {},
) -> list[dict]:
  
  results: list[dict] = []
  if max_eplen == "demo_max":
    max_eplen = max(list(map(len, rec_demos)))

  obs: Observation
  results: list[dict] = []

  agent.policy.to("cpu")
  for rec_demo in rec_demos:
    task_env = env.get_task(task, **task_params)
    task_env.reset()

    _, obs = task_env.reset_to_demo(rec_demo) 
    done = False
    
    distances = []

    for _ in range(max_eplen):
      action, _ = agent.act(obs)
      action = action.squeeze()
      obs, reward, done = task_env.step(action)
      gripper = Object.get_object("Panda_gripper")
      target = Object.get_object("target")
      distance = np.linalg.norm(gripper.get_position() - target.get_position())

      distances.append(distance)

      if within_err_dist is not None:
        done = done or distance <= within_err_dist

      if done: break

    results.append({
      "done": done,
      "distances": distances,
      "max_eplen": max_eplen
    })


  return results


## give UNTRAINED AGENT here
def run_reach_task_with_agent(
  env: Environment,
  task: Task, ## any of Reach_* or ReachObs_* tasks
  agent: Agent, 
  demos: list[Demo],
  max_eplen: int | Literal["demo_max"] = "demo_max",
  within_err_dist: Optional[float] = None, ## allows the execution to finish early depending on if an error around the target is reached
  training_params: dict = {},
  task_params: dict = {}
) -> tuple[dict, bool]:
  ## new agent trained each time
  
  ## request demos and train!
  task_env, demos = demos_and_train_for_task(
    env,
    task, #type: ignore[arg-type]
    agent,
    demos,
    save_model=True,
    training_params = training_params,
    task_params = task_params
  )
    
  ## if max len is not specified make it the max of the givem demo
  if max_eplen == "demo_max":
    max_eplen = max(list(map(len, demos)))
  
  ## evaluate
  obs: Observation
  _, obs = task_env.reset()
  
  # obstacle = Shape("obstacle")
  
  agent.policy.to("cpu") # move to cpu if not alr there
  distances = []
  done  = False
  
  atts_above_obs = []
  atts_below_obs = []
  
  for _ in range(max_eplen):
    action, pol_dict = agent.act(obs) # type: ignore
    action = action.squeeze()
    
    ## get the attention weights
    # atts = pol_dict["attention_weights"]
    # print(f"{atts = }")
    ## if the z value (height) of arm is negative with respect to obstacle, then we are below
    
    # if task_env._robot.arm.get_tip().get_position(relative_to=obstacle)[2] <= 0:
    #   ## below
    #   # print("BELOW THE OBS")
    #   # print(f"{atts = }")
    #   # print()
    #   atts_below_obs.append(atts)
    # else:
    #   ## above
    #   # print("above THE OBS")
    #   # print(f"{atts = }")
    #   # print()
    #   atts_above_obs.append(atts)
    
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
    "max_eplen": max_eplen
    # "avg_attentions_below_obstacle": torch.stack(atts_below_obs, dim=0).mean(dim=0, dtype=torch.float32) if len(atts_below_obs) > 0 else None,
    # "avg_attentions_above_obstacle": torch.stack(atts_above_obs, dim=0).mean(dim=0, dtype=torch.float32) if len(atts_above_obs) > 0 else None,
    },  done  


def run_grasp_with_agent(  
  env: Environment,
  task: Task, ## any of Reach_* or ReachObs_* tasks
  agent: Agent, 
  demos: list[Demo],
  max_eplen: int | Literal["demo_max"] = "demo_max",
  training_params: dict = {},
  task_params: dict = {},
  print_index: Optional[int] = None,
  task_env: Optional[TaskEnvironment] = None ## pass a task environemnt to skip the retraining bit, policy must alreay be trained!
) -> tuple[dict, bool]:
  
  ## Train the agent if it is None
  if task_env is None:
    task_env, demos = demos_and_train_for_task(
      env,
      task, #type: ignore[arg-type]
      agent,
      demos,
      save_model=True,
      training_params = training_params,
      task_params = task_params
    )
  
  ## if max len is not specified make it the max of the givem demo
  if max_eplen == "demo_max":
    max_eplen = max(list(map(len, demos)))

  ## evaluate
  obs: Observation
  _, obs = task_env.reset()
  
  
  agent.policy.to("cpu") # move to cpu if not alr there
  distances = []
  gripper_closing: list[dict] = []
  done  = False
  gripper_image_paths = []
  
  for _ in range(max_eplen):
    action, _ = agent.act(obs) # type: ignore
    action = action.squeeze()
    
    ## if it predicts gripper closed, I want to learn where it thought to do this
      

    obs, reward, done = task_env.step(action)

    gripper = Object.get_object("Panda_gripper")
    target = Object.get_object("grasp_cube")
    distance = np.linalg.norm(gripper.get_position() - target.get_position())
    

    if action[-1] <= 0.5:
      gripper_image_path = f"outputs/run-grasp-with-agent/{now()}/{print_index if print_index is not None else ''}"

      gripper_image_paths.append(gripper_image_path)

      os.makedirs(gripper_image_path, exist_ok=True)



      plot_cameras(agent.cam_type, obs, f"{agent.cam_type}-closed_at_{distance:.4f}-g_pred_({action[-1]:.4f})", gripper_image_path)
      
    distances.append(distance)
    
    ## NOTE: no within_err_dist, so the task is only completed if the object is acutally grasped
    if done: 
      break
    
  return {
    "distances": distances, 
    "task_params": task_params,
    "gripper_image_paths": gripper_image_paths
    },  done  

def plot_cameras(cam_type: CamType, obs: Observation, plot_title: str, save_folder: str):

  to_plot: list[CamType] = [ct for ct in CamType.uniques() ]# if ct & cam_type] ## plot all for now?
  if len(to_plot) <= 0:
    raise ValueError("[task_utils - plot_cameras] No Cameras were given to plot")
  
  plots = {
    str(ct): pick_obs_from_cam(ct, obs, normalise_rgb=False) ## for plotting no reason to normalise 
    for ct in to_plot
   }

  
  fig, axs = plt.subplots(1, len(to_plot), figsize=(3 * len(to_plot), 3))
  fig.suptitle(plot_title, fontsize=14)

  ## needed because subplts returns a single `Axes` or a list depending fig rows/cols
  if len(to_plot) == 1:
    axs = [axs]

  for (s, img), ax in zip(plots.items(), axs):
    ax.imshow(img)
    ax.set_title(s)
    ax.axis("off")

  plt.savefig(f"{save_folder}/{plot_title}-{cam_type}.png", bbox_inches="tight")
  plt.close()

## give a pretrained agent to make it run through the given obstacles
def run_determined_grasp_with_agent(
    # env: Environment, 
    # task: type[Task], 
    task_env: TaskEnvironment,
    agent: Agent, 
    rec_demos: list[Demo],
    max_eplen: int | Literal["demo_max"] = "demo_max",
    # task_params: dict = {}, ## doesnt need task params because the `reset_to_demo` needs the task_env to have already been created with the correct params
):
  if max_eplen == "demo_max":
    max_eplen = max(list(map(len, rec_demos)))

  obs: Observation
  results: list[dict] = []

  agent.policy.to("cpu")
  for rec_demo in rec_demos:
    _, obs = task_env.reset_to_demo(rec_demo) 
    done = False
    gripper_image_paths = []
    distances = []

    for _ in range(max_eplen):
      action, _ = agent.act(obs)
      action = action.squeeze()
      obs, reward, done = task_env.step(action)
      gripper = Object.get_object("Panda_gripper")
      target = Object.get_object("grasp_cube")
      distance = np.linalg.norm(gripper.get_position() - target.get_position())

      if action[-1] <= 0.5:
        gripper_image_path = f"outputs/run-grasp-with-agent/{now()}"
        gripper_image_paths.append(gripper_image_path)
        os.makedirs(gripper_image_path, exist_ok=True)
        plot_cameras(agent.cam_type, obs, f"{agent.cam_type}-closed_at_{distance:.4f}-g_pred_({action[-1]:.4f})", gripper_image_path)
        
      distances.append(distance)
      if done: break
    results.append({
      "done": done,
      "distances": distances,
      "gripper_image_paths": gripper_image_paths,
      "max_eplen": max_eplen
    })

  return results



ALL_CAMS = [
  "wrist_camera",
  "right_shoulder_camera",
  "left_shoulder_camera",
  "overhead_camera",
  "front_camera",
]

CT_DICT = {
  CamType.WRIST: "wrist_camera",
  CamType.RIGHT_SHOULDER: "right_shoulder_camera",
  CamType.LEFT_SHOULDER: "left_shoulder_camera",
  CamType.OVERHEAD: "overhead_camera",
  CamType.FRONT: "front_camera",
}


def launch_test_env(
  dataset_root: Optional[str],
  enabled_config =  CameraConfig(
    rgb=True, depth=True, mask=True, point_cloud=True,
    render_mode=RenderMode.OPENGL, image_size=(64, 64)
  ),
  disabled_config = CameraConfig(
    rgb=False, depth=False, mask=False,
    render_mode=RenderMode.OPENGL,
  ),
  enableds: CamType | list[str] | Literal["all"] = [
    "wrist_camera",
    "right_shoulder_camera",
    "left_shoulder_camera"
  ],
  action_mode = MoveArmThenGripper(
      arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
) -> Environment:
  
  DATASET = dataset_root if dataset_root is not None else ''

  obs_config = ObservationConfig()
  obs_config.set_all(True) ## important to get the data from the joints etc

  if isinstance(enableds, str) and enableds == "all":
    enableds = ALL_CAMS

  if isinstance(enableds, CamType):
    if enableds & CamType.WRIST_DEPTH:
      raise RuntimeError(f"[task_utils - launch_test_env] '{CamType.WRIST_DEPTH}' is not a valid enablable cam, use RGB")
    enableds = [CT_DICT[ct] for ct in CamType.uniques() if enableds & ct]

  for enabled in enableds:
    setattr(obs_config, enabled, enabled_config)

  for disabled in filter(lambda s: s not in enableds, ALL_CAMS):
    setattr(obs_config, disabled, disabled_config)

  env = Environment(
      action_mode, DATASET, obs_config, False)

  env.launch()
  return env
  
## loads all the demos for a requested task can then be sliced later
def load_demos_for(
  amount: int, 
  env: Environment,  
  task: type[Task], 
  new_root: str,
  random_selection: bool = False, 
  task_params: dict = {}
) -> list[Demo]:

  old_root = env._dataset_root
  env._dataset_root = new_root

  task_env = env.get_task(task, **task_params)
  demos = task_env.get_demos(
    amount, 
    live_demos = False, 
    random_selection = random_selection
  ) ## NOTE can add other params as a dict, but probably wont need it

  env._dataset_root = old_root
  return demos

## creates and saves demos for the given task, launches a new env with everything enabled
def save_demos_for(
  amount: int,
  task: type[Task], 
  dir: str,
  task_params: dict = {}, 
  return_env: bool = False
) -> Optional[Environment]:
  
  env = launch_test_env(
    dataset_root='', 
    enableds = "all"
  )

  task_env = env.get_task(task, **task_params)
  demos = task_env.get_demos(amount, live_demos = True)

  for i, demo in enumerate(demos):
    save_demo(demo, f"{dir}/{task_env._task.get_name()}/{i}")

  print(f"[task_utils - save_demos_for] Done creating and saving demos in {dir}")

  if return_env:
    return env
  
  env.shutdown()
  

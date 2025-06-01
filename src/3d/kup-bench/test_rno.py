#%%

# Tasks
## No Obstacle
from rlbench.tasks.reach_target_no_obs_side_r import ReachTargetNoObsSideR as ReachNoObs_SideR
from rlbench.tasks.reach_target_no_obs_side_l import ReachTargetNoObsSideL as ReachNoObs_SideL
from rlbench.tasks.reach_target_no_obs_central import ReachTargetNoObsCentral as ReachNoObs_Central
from rlbench.tasks.reach_target_no_obs_random import ReachTargetNoObsRandom as ReachNoObs_PlaceRandom


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



from lib.utils import get_task_name
from lib.cam_type import CamType
from lib.agent import Agent
from lib.policy_type import PolicyType
from modules.dataset.demo_dataset import DemoDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task_utils import launch_test_env, save_demos_for, load_demos_for, run_determined_reach_with_agent, run_reach_task_with_agent
from lib.utils import save_demos
from itertools import product
#%%
def static_tasks_epoch_search():
  tasks = [
    ReachNoObs_Central,
    ReachNoObs_SideL,
    ReachNoObs_SideR,
  ]

  epochs = [1, 2, 10, 50, 100, 200, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 50000]
  dataset_types = [
    "obs",
    "demo", 
  ]
  env = launch_test_env(
    dataset_root='',
    enableds = CamType.WRIST
  )
  make_agent = lambda: Agent(
    action_shape= env.action_shape[0],
    policy_type=PolicyType.SIMPLE,
    cam_type=CamType.WRIST
  )

  df = pd.DataFrame(columns=[
      "task_name",
      "cam_type",
      "epochs",
      "demo_count",
      "max_eplen",
      "is_success",
      "min_distance",
      "final_distance",
      "dataset_type"
    ], index = range(len(epochs) * len(tasks) * len(dataset_types))
  )

  training_params = {
    "epochs": None,
    "minibatch_size": 32,
    "lr": 1e-3,
    "shuffle_obs_in_demo": False,
    "shuffle_data": False,
    "dataset_to_use": "obs"
  }

  count = 0 

  for dt in dataset_types:
    for task in tasks:
      task_env = env.get_task(task)
      task_env.reset()

      demos = load_demos_for(1, env, task, f"data/demo")

      for ep in epochs:
        training_params["dataset_to_use"] = dt  
        if dt == "demo":
          training_params["minibatch_size"] = 1 

        training_params["epochs"] = ep  
        agent = make_agent()

        rets, is_done = run_reach_task_with_agent(
          env, 
          task, 
          agent,
          demos, 
          "demo_max", 
          within_err_dist= 0.08,
          training_params= training_params,
        )  

        df.loc[count] = {
          "task_name": get_task_name(task), 
          "cam_type": agent.cam_type, 
          "epochs": ep,
          "demo_count": 1, 
          "max_eplen": rets["max_eplen"],
          "is_success": is_done, 
          "min_distance": min(rets["distances"]),
          "final_distance": rets["distances"][-1],
          "dataset_type": training_params["dataset_to_use"]
        }
        count += 1
        df.to_csv("rno_static.csv", index=True)
    
def place_random():
  task = ReachNoObs_PlaceRandom

  epochs = [
    1, 2, 10, 50, 100, 200, 400, 500, 1000]

  demo_counts = [1, 5, 10, 20]

  dataset_types = [
    "obs",
    "demo", 
  ]
  env = launch_test_env(
    dataset_root='',
    enableds = CamType.WRIST
  )
  task_env = env.get_task(task)
  task_env.reset()
  demos = load_demos_for(2, 
    env, 
    task, 
    f"data/20demos", 
  )


  test_demos = task_env.get_demos(10, live_demos = True)
  save_demos(test_demos, f"rno-place_randon--demos")
  make_agent = lambda: Agent(
    action_shape= env.action_shape[0],
    policy_type=PolicyType.SIMPLE,
    cam_type=CamType.WRIST
  )

  df = pd.DataFrame(columns=[
      "task_name",
      "cam_type",
      "epochs",
      "demo_count",
      "max_eplens",
      "done_count",
      "min_distance",
      "final_distance",
      "dataset_type"
    ], index = range(len(epochs) * len(demo_counts) * len(dataset_types))
  )

  training_params = {
    "epochs": None,
    "minibatch_size": 32,
    "lr": 1e-3,
    "shuffle_obs_in_demo": False,
    "shuffle_data": False,
    "dataset_to_use": "obs", 
    "lock_loader_seed": 42

  }
  count = 0 
  for dt in dataset_types:
    
    task_env = env.get_task(task)
    task_env.reset()
    for dc in demo_counts:
      demos = load_demos_for(dc, env, task, f"data/20demos")

      if len(demos) != dc:
        print(f"was not the right size")
        
        demos = demos[:dc]
      

      for ep in epochs:
        training_params["dataset_to_use"] = dt  
        if dt == "demo":
          training_params["minibatch_size"] = 1 

        training_params["epochs"] = ep  
        agent = make_agent()

        agent.ingest(demos, **training_params)

        ret_dicts = run_determined_reach_with_agent(
          env, 
          task, 
          agent,
          test_demos, 
          "demo_max", 
          within_err_dist= 0.08,
        )  
        final_dists = [d["distances"][-1] for d in ret_dicts]
        min_dists = [min(d["distances"]) for d in ret_dicts]

        done_count = len([d["done"] for d in ret_dicts if d["done"]])
        eplens = [d["max_eplen"] for d in ret_dicts]

        df.loc[count] = {
          "task_name": get_task_name(task), 
          "cam_type": agent.cam_type, 
          "epochs": ep,
          "demo_count": dc, 
          "max_eplens": eplens,
          "done_count": done_count, 
          "min_distance": sum(min_dists) / len(min_dists),
          "final_distance": sum(final_dists) / len(final_dists),
          "dataset_type": training_params["dataset_to_use"]
        }
        count += 1
        df.to_csv("rno-place_random.csv", index=True)
  
#%%

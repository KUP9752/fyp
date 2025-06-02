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

from task_utils import launch_test_env, save_demos_for, load_demos_for, run_determined_reach_with_agent, run_reach_task_with_agent, run_determined_grasp_with_agent
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
    
def run_main_test():
  tasks = [ReachObs_Random, ReachObs_IndepRandom]

  epochs = [
    50, 100, 200, 500, 1000, 2000, 5000
  ]

  demo_counts = [10, 20]
  cam_types = [
    CamType.WRIST,
    CamType.LEFT_SHOULDER,
    CamType.RIGHT_SHOULDER,
    CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
    CamType.WRIST | CamType.RIGHT_SHOULDER,
    CamType.WRIST | CamType.LEFT_SHOULDER,
    CamType.LEFT_SHOULDER |  CamType.RIGHT_SHOULDER,
  ]
  dataset_types = [
    "obs",
    "demo", 
  ]
  env = launch_test_env(
    dataset_root='',
    enableds = CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
  )


  
  make_agent = lambda x: Agent(
    action_shape= env.action_shape[0],
    policy_type=PolicyType.SIMPLE,
    cam_type=x
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
    ], index = range(len(tasks) * len(epochs) * len(demo_counts) * len(dataset_types) * len(cam_types))
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
  for task in tasks:
    task_env = env.get_task(task)
    test_demos = task_env.get_demos(10, live_demos = True)
    save_demos(test_demos, f"ro-{get_task_name(task)}--demos")
    for dt in dataset_types:
      ## this already done can skip it
      # if task == ReachObs_Random and dt == "obs": continue ## was already done earlier
      task_env = env.get_task(task)
      task_env.reset()
      for dc in demo_counts:
        demos = load_demos_for(dc, env, task, f"data/20demos")
        for ct in cam_types:

          if len(demos) != dc:
            print(f"was not the right size")
            
            demos = demos[:dc]
          

          for ep in epochs:
            training_params["dataset_to_use"] = dt  
            if dt == "demo":
              training_params["minibatch_size"] = 10

            training_params["epochs"] = ep  
            agent = make_agent(ct)

            agent.ingest(demos, **training_params)

            ret_dicts = run_determined_reach_with_agent(
              env, 
              task, 
              agent,
              test_demos, 
              "demo_max", 
              within_err_dist= 0.11,
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
            df.to_csv("ro-randoms-cam.csv", index=True)
#%%
def test_di():
  env = launch_test_env(
    dataset_root = "data/1demo",
    enableds = CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
  )

  task = Vision_Static
  normal = {
    "scale": 1.,
    "wrist_cam_distance": 0.6
  }
  smaller = {
    "scale": 0.5,
    "wrist_cam_distance": 0.3
  }

  env._dataset_root = f"data/1demo/normal-{get_task_name(task)}"
  task_env = env.get_task(task, **normal)
  demos = task_env.get_demos(1, live_demos=False)

  
  env._dataset_root = f"data/1demo/smaller-{get_task_name(task)}"
  task_env = env.get_task(task, **smaller)
  small_demos = task_env.get_demos(1, live_demos=False)

  epochs = [50, 100, 150, 200, 400, 500]
  repeats = 10
  cam_types =  [
    CamType.WRIST,
    CamType.WRIST | CamType.WRIST_DEPTH,
    CamType.LEFT_SHOULDER |  CamType.RIGHT_SHOULDER,

    CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
    CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER | CamType.WRIST_DEPTH,
    CamType.WRIST | CamType.RIGHT_SHOULDER,
    CamType.WRIST | CamType.LEFT_SHOULDER,
  ]

  make_agent = lambda x: Agent(
    action_shape= env.action_shape[0],
    policy_type=PolicyType.DEPTH_GRASP,
    cam_type=x,
    config = "depth_ch"
    ## others are defaulted
  )

  training_params = {
    "epochs": None,
    "minibatch_size": 1,
    "lr": 1e-3,
    "shuffle_obs_in_demo": False,
    "shuffle_data": True,
    "dataset_to_use": "demo",  ## can use this finally
    "lock_loader_seed": 42,
    "lambda_grasp_loss": 1,
  }

  combs = list(product(epochs, cam_types))
  df_all = pd.DataFrame(columns=[
      "task_name",
      "epochs",
      "cam_type",
      "repeat_idx",
      "demo_count",
      "control_done",
      "test_done",
      "control_min_distance",
      "test_min_distance",
      "control_final_distance",
      "test_final_distance",
      "dataset_type",
      "control_gripper_image_paths",
      "test_gripper_image_paths",
    ], index = range(len(combs) * repeats)
  )

  df_avg = pd.DataFrame(columns=[
      "task_name",
      "epochs",
      "cam_type",
      "demo_count",
      "control_done_count",
      "test_done_count",
      "avg_control_min_distance",
      "avg_test_min_distance",
      "control_final_distance",
      "test_final_distance",
      "dataset_type",
    ], index = range(len(combs))
  )


  
  count = 0
  for i, (ep, ct) in enumerate(combs):
    training_params["epochs"] = ep
    assert len(small_demos) == len(demos) == 1, "more than 1 demo!!"
    
    control_dones = []
    test_dones = []
    control_mins = []
    test_mins = []
    control_finals = []
    test_finals = []

    for rep in range(repeats):
      agent = make_agent(ct)
      agent.ingest(demos, **training_params)

      control_dict = run_determined_grasp_with_agent(
        env, 
        task, 
        agent, 
        demos, 
        max_eplen="demo_max", 
        do_extra_outs=True, 
        task_params = normal
      )[0] ## there is only one demo hence only one output
      test_dict = run_determined_grasp_with_agent(
        env, 
        task, 
        agent, 
        small_demos, 
        max_eplen="demo_max", 
        do_extra_outs=True, 
        task_params = smaller
      )[0]

      control_dones.append(control_dict["done"])
      test_dones.append(test_dict["done"])
      control_mins.append(min(control_dict["distances"]))
      test_mins.append(min(test_dict["distances"]))
      control_finals.append(control_dict["distances"][-1])
      test_finals.append(test_dict["distances"][-1])

      df_all.loc[count] = {
        "task_name": get_task_name(task),
        "epochs": training_params["epochs"],
        "cam_type": ct,
        "repeat_idx": rep,
        "demo_count": 1,
        "control_done": control_dict["done"],
        "test_done": test_dict["done"],
        "control_min_distance": min(control_dict["distances"]),
        "test_min_distance": min(test_dict["distances"]),
        "control_final_distance":control_dict["distances"][-1],
        "test_final_distance":test_dict["distances"][-1],
        "dataset_type": training_params["dataset_to_use"],
        "control_gripper_image_paths":control_dict["gripper_image_paths"],
        "test_gripper_image_paths":test_dict["gripper_image_paths"],
      }
      df_all.to_csv("ALLvs-train_normal-test_small.csv", index=True)
      count += 1
    
    df_avg.loc[i] = {
      "task_name": get_task_name(task),
      "epochs": training_params["epochs"],
      "cam_type": ct,
      "demo_count": 1,
      "control_done_count": len([b for b in control_dones if b]),
      "test_done_count": len([b for b in test_dones if b]),
      "avg_control_min_distance": sum(control_mins) / len(control_mins),
      "avg_test_min_distance": sum(test_mins) / len(test_mins),
      "control_final_distance":sum(control_finals) / len(control_finals),
      "test_final_distance":sum(test_finals) / len(test_finals),
      "dataset_type": training_params["dataset_to_use"],
    }
    df_avg.to_csv("vs-train_normal-test_small.csv", index=True)







#%%
test_di()
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
from pyrep.objects import Object, VisionSensor, Shape
import torch
from rlbench.tasks.vision_static import VisionStatic as Vision_Static
from rlbench.tasks.vision_random import VisionRandom as Vision_Random
from lib.utils import get_task_name
from lib.cam_type import CamType
from lib.agent import Agent
from lib.policy_type import PolicyType
from modules.dataset.demo_dataset import DemoDataset
import pandas as pd
from task_utils import launch_test_env, save_demos_for, load_demos_for, run_determined_reach_with_agent, run_determined_grasp_with_agent, run_reachobs_random_task_with_agent
from lib.utils import save_demos, load_demos
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

        # rets, is_done = run_reach_task_with_agent(
        #   env, 
        #   task, 
        #   agent,
        #   demos, 
        #   "demo_max", 
        #   within_err_dist= 0.08,
        #   training_params= training_params,
        # )  

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
    
  
#%%
def grasp_tuning():
  env = launch_test_env(
    dataset_root = "data/1demo",
    enableds = CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
  )

  task = Vision_Static
  normal = {
    "scale": 1.,
    "wrist_cam_distance": 0.6
  }
  # smaller = {
  #   "scale": 0.5,
  #   "wrist_cam_distance": 0.3
  # }

  env._dataset_root = f"data/1demo/normal-{get_task_name(task)}"
  task_env = env.get_task(task, **normal)
  demos = task_env.get_demos(1, live_demos=False)

  
  # env._dataset_root = f"data/1demo/smaller-{get_task_name(task)}"
  # task_env = env.get_task(task, **smaller)
  # small_demos = task_env.get_demos(1, live_demos=False)

  epochs = [100, 150, 200, 400, 500, 600]
  lambdas = [1.2]
  last_ks = [None, 15, 28]
  repeats = 5

  cam_types =  [
    CamType.WRIST,
    CamType.WRIST | CamType.WRIST_DEPTH,
    CamType.LEFT_SHOULDER |  CamType.RIGHT_SHOULDER,
    CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
    # CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER | CamType.WRIST_DEPTH,
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
  print(f"{make_agent(CamType.WRIST).policy.grasp_head = }")
  

  training_params = {
    "epochs": None,
    "minibatch_size": 1,
    "lr": 1e-3,
    "shuffle_obs_in_demo": False,
    "shuffle_data": True,
    "dataset_to_use": "demo",  ## can use this finally
    "lock_loader_seed": 42,
    "lambda_grasp_loss": None,
    "last_k_grasp_mask": None
  }

  combs = list(product(epochs, cam_types, lambdas, last_ks))
  df_all = pd.DataFrame(columns=[
      "task_name",
      "epochs",
      "cam_type",
      "repeat_idx",
      "demo_count",
      "control_done",
      "min_distance",
      "final_distance",
      "dataset_type",
      "grasp_loss_lambda",
      "last_k_mask",
      "control_gripper_image_paths",
      "max_eplen", 
    ], index = range(len(combs) * repeats)
  )

  df_avg = pd.DataFrame(columns=[
    "task_name",
    "epochs",
    "cam_type",
    "demo_count",
    "done_count",
    "avg_min_distance",
    "avg_final_distance",
    "dataset_type",
    "grasp_loss_lambda",
    "last_k_mask",
    ], index = range(len(combs))
  )

  count = 0
  for i, (ep, ct, l, last_k) in enumerate(combs):
    training_params["epochs"] = ep
    training_params["lambda_grasp_loss"] = l
    training_params["last_k_grasp_mask"] = last_k

    assert len(demos) == 1, "more than 1 demo!!"
    
    dones  = []
    mins  = []
    finals = []
    for rep in range(repeats):
      agent = make_agent(ct)
      agent.ingest(demos, **training_params)

      run_dict = run_determined_grasp_with_agent(
        env, 
        task, 
        agent, 
        demos, 
        max_eplen="demo_max", 
        do_extra_outs=True, 
        task_params = normal
      )[0] ## there is only one demo hence only one output

      dones.append(run_dict["done"])
      mins.append(min(run_dict["distances"]))
      finals.append(run_dict["distances"][-1])

      df_all.loc[count] = {
        "task_name": get_task_name(task),
        "epochs": training_params["epochs"],
        "cam_type": ct,
        "repeat_idx": rep,
        "demo_count": 1,
        "control_done": run_dict["done"],
        "min_distance": min(run_dict["distances"]),
        "final_distance": run_dict["distances"][-1],
        "dataset_type": training_params["dataset_to_use"],
        "grasp_loss_lambda": training_params["lambda_grasp_loss"],
        "control_gripper_image_paths":run_dict["gripper_image_paths"],
        "last_k_mask": training_params["last_k_grasp_mask"],
        "max_eplen":run_dict["max_eplen"],
      }
      df_all.to_csv("ALLvs-tuning-normal--last-k-test2.csv", index=True)
      count += 1
    
    df_avg.loc[i] = {
      "task_name": get_task_name(task),
      "epochs": training_params["epochs"],
      "cam_type": ct,
      "demo_count": 1,
      "done_count": len([b for b in dones if b]),
      "avg_min_distance": sum(mins) / len(mins),
      "avg_final_distance":sum(finals) / len(finals),
      "dataset_type": training_params["dataset_to_use"],
      "grasp_loss_lambda": training_params["lambda_grasp_loss"],
      "last_k_mask": training_params["last_k_grasp_mask"],
    }
    df_avg.to_csv("vs-tuning-normal--last-k-test2.csv", index=True)
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
      agent.ingest(small_demos, **training_params)

      control_dict = run_determined_grasp_with_agent(
        env, 
        task, 
        agent, 
        small_demos, 
        max_eplen="demo_max", 
        do_extra_outs=True, 
        task_params = smaller
      )[0] ## there is only one demo hence only one output
      test_dict = run_determined_grasp_with_agent(
        env, 
        task, 
        agent, 
        demos, 
        max_eplen=100, 
        do_extra_outs=True, 
        task_params = normal
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
        "control_gripper_image_paths": control_dict["gripper_image_paths"],
        "test_gripper_image_paths": test_dict["gripper_image_paths"],
      }
      df_all.to_csv("ALLvs-train_small-test_normal--normal_more_eplen-.csv", index=True)
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
    df_avg.to_csv("vs-train_small-test_normal--normal_more_eplen-.csv", index=True)


#%%
# test_di()
# run_main_test()
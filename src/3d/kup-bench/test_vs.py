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

from rlbench.backend.task import Task
from rlbench.demo import Demo
from rlbench.environment import Environment
from modules.policy.fusing_policy import FuseConfig
import logging

# Set up logger
logger = logging.getLogger('my_test_logger')
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler('test_log.txt')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# Add both handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Example usage
logger.info("Starting long test.")

def main():
  task = Vision_Random
  
  env = launch_test_env(
    "data/20demos/normal-Vision_Random-setdist:0.6", 
    enableds = CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
  )
  
  normal = {
    "scale": 1.,
    "wrist_cam_distance": 0.6
  }
  smaller = {
    "scale": 0.5,
    "wrist_cam_distance": 0.3
  }
  training_demos = load_demos_for(
    10,
    env, 
    task, 
    "data/20demos/normal-Vision_Random-setdist:0.6",
    task_params = normal
  )
  test_normal = load_demos_for(
    10,
    env, 
    task, 
    "data/test/10demos/normal-Vision_Random-setdist:0.6",
    task_params = normal
  )

  test_smaller = load_demos_for(
    10,
    env, 
    task, 
    "data/test/10demos/smaller-Vision_Random-setdist:0.3",
    task_params=smaller
  )
  configs = [
    FuseConfig.W_D_L_R_FILM,
    FuseConfig.W_D_L_R_ATTN,
  ]

  epochs = [100, 200, 600, 1000, 2000]

  repeats = 5
  cam_types =  [ 
    CamType.WRIST,
    CamType.WRIST | CamType.WRIST_DEPTH,
    CamType.LEFT_SHOULDER |  CamType.RIGHT_SHOULDER,
    CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
    CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER | CamType.WRIST_DEPTH,

    # CamType.WRIST | CamType.RIGHT_SHOULDER,
    # CamType.WRIST | CamType.LEFT_SHOULDER,
  ]
  combs = list(product(epochs, cam_types))
  
  training_params = {
    "epochs": None,
    "minibatch_size": 10,
    "lr": 1e-3,
    "shuffle_obs_in_demo": False,
    "shuffle_data": True,
    "dataset_to_use": "demo", 
    "lock_loader_seed": 42,
    "lambda_grasp_loss": 1,
  }

  config_columns = [
    "rep",
    "config",
    "error",
    "task_name",
    "cam_type",
    "epochs",

    "control_done_count",
    "test_done_count",

    "avg_control_final_distance",
    "avg_control_min_distance",
    "avg_test_min_distance",
    "avg_test_final_distance",

  ]
  demo_columns = [
    "rep",
    "config",
    "error",
    "task_name",
    "demo_idx",
    "cam_type",
    "epochs",

    "is_control_done",
    "is_test_done",

    "control_final_distance",
    "control_min_distance",
    
    "test_final_distance",
    "test_min_distance",

    "control_gripper_image_paths",
    "test_gripper_image_paths",
  ]

  make_grasp_agent = lambda ct, cfg,: Agent(
    action_shape= env.action_shape[0],
    policy_type=PolicyType.FUSING,
    cam_type=ct,

    is_grasp = True, 

    fuse_config = cfg,
    fusing_opts = {}, ## make sure to use defaults 

    use_proprio = False,
    proprio_opts = {}, ## make sure to use defaults 

    ## others are defaulted
  )
  logger.info("=== Starting Long Test:")
  for rep in range(repeats):
    for config in configs:

      df_all = pd.DataFrame(
        columns=demo_columns, 
        index = range(len(combs) * 10) 
      )
      df_avg = pd.DataFrame(
        columns=config_columns, 
        index = range(len(combs)) 
      )

      all_count = 0
      avg_count = 0

      df_all.to_csv(f"zz-new-out/ALLvs_random-ns-cfg:{config}-rep{rep}.csv", index=True)
      df_avg.to_csv(f"zz-new-out/vs_random-ns-cfg:{config}-rep{rep}.csv", index=True)

      for (ep, ct) in combs:
        training_params["epochs"] = ep
        control_dones = []
        test_dones = []
        control_mins = []
        test_mins = []
        control_finals = []
        test_finals = []

        try:
          agent = make_grasp_agent(ct, config)
          agent.ingest(training_demos, **training_params)

          control_dicts = run_determined_grasp_with_agent(
              env,
              task,
              agent, 
              test_normal, 
              max_eplen="demo_max",
              do_extra_outs=True,
              task_params=normal
            )
          
          test_dicts = run_determined_grasp_with_agent(
              env,
              task,
              agent, 
              test_smaller, 
              max_eplen="demo_max",
              do_extra_outs=True,
              task_params=smaller
            )
          # assert len(test_dicts) == len(control_dicts) == 10, "10 test demos see if this is respected"
          logger.info(f"{len(test_dicts) = }")
          logger.info(f"{len(control_dicts) = }")

          for i in range(len(control_dicts)):
            cd = control_dicts[i]
            td = test_dicts[i]

            control_dones.append(cd["done"])
            test_dones.append(td["done"])

            control_mins.append(min(cd["distances"]))
            test_mins.append(min(td["distances"]))

            control_finals.append(cd["distances"][-1])
            test_finals.append(td["distances"][-1])

            df_all.loc[all_count] = {
              "rep": rep,
              "config": config,
              "error": False,
              "task_name": get_task_name(task),
              "cam_type": ct,
              "demo_idx": i,
              "epochs": training_params["epochs"],

              "control_final_distance": cd["distances"][-1],
              "control_min_distance": min(cd["distances"]),
              "is_control_done": cd["done"],

              "test_final_distance": td["distances"][-1],
              "test_min_distance": min(td["distances"]),
              "is_test_done": td["done"],
              "control_gripper_image_paths": cd["gripper_image_paths"],
              "test_gripper_image_paths": td["gripper_image_paths"]

            }
            all_count+= 1
            df_all.to_csv(f"zz-new-out/ALLvs_random-ns-cfg:{config}-rep{rep}.csv", index=True)

          df_avg.loc[avg_count] = {
            "rep": rep,
            "config": config,
            "error": False,
            "task_name": get_task_name(task),
            "cam_type": ct,
            "epochs": training_params["epochs"],

            "control_done_count": len([b for b in control_dones if b]),
            "test_done_count":len([b for b in test_dones if b]),

            "avg_control_final_distance": sum(control_finals) / len(control_finals),
            "avg_control_min_distance": sum(control_mins) / len(control_mins),

            "avg_test_final_distance": sum(test_finals) / len(test_finals),
            "avg_test_min_distance": sum(test_mins) / len(test_mins),

          }
          avg_count += 1
          df_avg.to_csv(f"zz-new-out/vs_random-ns-cfg:{config}-rep{rep}.csv", index=True)

        except Exception as e:
          
          logger.exception(f"Something Went Wrong! Marking row in DF")
          logger.error(e)
          for i in range(10):
            df_all.loc[all_count] = {
              "rep": rep,
              "config": config,
              "error": True,
              "task_name": get_task_name(task),
              "cam_type": ct,
              "epochs": training_params["epochs"],
            }
            all_count+= 1
            df_all.to_csv(f"zz-new-out/ALLvs_random-ns-cfg:{config}-rep{rep}.csv", index=True)

          df_avg.loc[avg_count] = {
            "rep": rep,
            "config": config,
            "error": True,
            "task_name": get_task_name(task),
            "cam_type": ct,
            "epochs": training_params["epochs"],
          }
          avg_count += 1
          df_avg.to_csv(f"zz-new-out/vs_random-ns-cfg:{config}-rep{rep}.csv", index=True)

          logger.info("Added error rows to dfs")

  logger.info("DONE")
if __name__ == "__main__":
  main()


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
from modules.policy.fusing_policy import FuseConfig

import logging
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

if __name__ == "__main__": 
  task = ReachObs_Random #, ReachObs_Random]

  epochs = [
    100, 200, 500, 1000, #2000, 5000 # maybe do again if needed later
  ]

  seeds = [
    3790,
    # 1901,
    # 4248,
    # 6689,
    # 7653
  ]

  demo_count = 10
  cam_types = [
    CamType.WRIST,
    CamType.WRIST | CamType.WRIST_DEPTH,
    CamType.LEFT_SHOULDER |  CamType.RIGHT_SHOULDER,
    CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
    CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER | CamType.WRIST_DEPTH,

    # CamType.WRIST | CamType.RIGHT_SHOULDER,
    # CamType.WRIST | CamType.LEFT_SHOULDER,
  ]

  env = launch_test_env(
    dataset_root='',
    enableds = CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
  )

  configs = [
    FuseConfig.WDLR,
    FuseConfig.WLR_D,
    FuseConfig.DEPTH_FEATS_GATED,
    FuseConfig.DEPTH_FEATS_ATTN,
    FuseConfig.WD_LR,
    FuseConfig.WD_LR_ATTN,
    FuseConfig.Wfilm_D,
    FuseConfig.W_Dfilm,
    FuseConfig.Wfilm_Dfilm,
    FuseConfig.Wfilm_D_LATE,
    FuseConfig.W_Dfilm_LATE,
    FuseConfig.Wfilm_Dfilm_LATE,
    # FuseConfig.W_D_L_R,
    FuseConfig.W_D_L_R_FILM,
    # FuseConfig.W_D_L_R_ATTN, ##bad
  ]
  ##fixed mb size this time
  training_params = {
    "epochs": None,
    "minibatch_size": 10,
    "lr": 1e-3,
    "shuffle_obs_in_demo": False,
    "shuffle_data": True, ##let "demo" use shuffling and other things, that benefit its learning for more than 1 demo
    "dataset_to_use": "demo", 
    "lock_loader_seed": 42,
  }
  # fix the test dataset as this one
  main_folder = "ZZ-film-ro"
  df_all_str = lambda config, seed: f"{main_folder}/ALLro_random-cfg:{config}-seed{seed}.csv"
  df_avg_str = lambda config, seed: f"{main_folder}/ro_random-cfg:{config}-seed{seed}.csv"

  all_columns=[
    "task_name",
    "cam_type",
    "epochs",
    "demo_idx",
    "max_eplen",
    "done",
    "min_distance",
    "final_distance",
    "dataset_type",
    "seed",
    "config",
    "agent",
    "error",
  ]

  avg_columns = [
    "task_name",
    "cam_type",
    "epochs",
    "done_count",
    "avg_min_distance",
    "avg_final_distance",
    "dataset_type",
    "seed",
    "config",
    "agent",
    "error",
  ]
  make_agent = lambda ct, cfg,: Agent(
    action_shape= env.action_shape[0],
    policy_type=PolicyType.FUSING, ## NO RNN!!
    cam_type=ct,

    is_grasp = False,  ## shouldnt matter realistically

    fuse_config = cfg,
    fusing_opts = {}, ## make sure to use defaults 

    use_proprio = False,
    proprio_opts = {}, ## make sure to use defaults 

    ## others are defaulted
  )

  ## this already done can skip it
  # if task == ReachObs_Random and dt == "obs": continue ## was already done earlier
  task_env = env.get_task(task)
  task_env.reset()
  test_demos = load_demos_for(10, env,  task, f"data/test/10demos")
  demos = load_demos_for(demo_count, env, task, f"data/20demos")

  all_count = 0 
  df_all = pd.DataFrame(
    columns=all_columns, 
    index= range(
      len(epochs) 
      * len(configs) 
      * len(cam_types) 
      * len(seeds)
      * len(test_demos)
    )
  )
  
  avg_count = 0 
  df_avg = pd.DataFrame(
    columns=avg_columns, 
    index= range(
      len(epochs) 
      * len(configs) 
      * len(cam_types) 
      * len(seeds)
    )
  )

  if len(demos) != demo_count:
    raise RuntimeError("demo lengths are wrong")
  
  for seed, config, ct, ep in product(seeds, configs, cam_types, epochs):
    training_params["epochs"] = ep  
    training_params["lock_loader_seed"] = seed  
    try:
      agent = make_agent(ct, config)

      agent.ingest(demos, **training_params)

      ret_dicts = run_determined_reach_with_agent(
        env, 
        task, 
        agent,
        test_demos, 
        "demo_max", 
        within_err_dist= 0.11,
      )  

      finals = []
      mins = []
      dones = []
      for demo_idx, ret_dict in enumerate(ret_dicts):
        
        dones.append(ret_dict["done"])
        finals.append(ret_dict["distances"][-1])
        mins.append(min(ret_dict["distances"]))

        df_all.loc[all_count] = {
          "task_name": get_task_name(task), 
          "cam_type": ct,
          "epochs": ep,
          "demo_idx": demo_idx,
          "max_eplen": ret_dict["max_eplen"],
          "done": ret_dict["done"],
          "min_distance": min(ret_dict["distances"]),
          "final_distance": ret_dict["distances"][-1],
          "dataset_type": training_params["dataset_to_use"],
          "seed": seed,
          "config": config,
          "agent": agent,
          "error": False,
        }
        all_count += 1
        df_all.to_csv(df_all_str(config, seed), index=True)

      df_avg.loc[avg_count] = {
        "task_name": get_task_name(task), 
        "cam_type": ct,
        "epochs": ep,
        "done_count": len([b for b in dones if b]),
        "avg_min_distance": sum(mins) / len(mins),
        "avg_final_distance": sum(finals) / len(finals),
        "dataset_type": training_params["dataset_to_use"],
        "seed": seed,
        "config": config,
        "agent": agent,
        "error": False,
        }
      avg_count += 1
      df_avg.to_csv(df_avg_str(config, seed), index=True)

    except Exception as e:
      logger.exception(f"Something Went Wrong! Marking row in DF")
      logger.error(e)
      for i in range(10):
        df_all.loc[all_count] = {
          "seed": seed,
          "config": config,
          "error": True,
          "task_name": get_task_name(task),
          "cam_type": ct,
          "epochs": training_params["epochs"],
        }
        all_count+= 1
        df_all.to_csv(df_all_str(config, seed), index=True)

      df_avg.loc[avg_count] = {
        "seed": seed,
        "config": config,
        "error": True,
        "task_name": get_task_name(task),
        "cam_type": ct,
        "epochs": training_params["epochs"],
      }
      avg_count += 1
      df_avg.to_csv(df_avg_str(config, seed), index=True)

      logger.info("Added error rows to dfs")

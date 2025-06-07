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

if __name__ == "__main__": 
  tasks = [ReachObs_Random] #, ReachObs_IndepRandom]

  epochs = [
    100, 200, 500, 1000, 2000, #5000 # seems unnecessary
  ]

  demo_counts = [
    10, 
    # 20,
  ]

  cam_types = [
    # CamType.WRIST,
    # CamType.LEFT_SHOULDER,
    # CamType.RIGHT_SHOULDER,
    CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
    CamType.WRIST | CamType.RIGHT_SHOULDER,
    CamType.WRIST | CamType.LEFT_SHOULDER,
    CamType.LEFT_SHOULDER |  CamType.RIGHT_SHOULDER,
  ]

  attn_losses = [
    1, 5, 10
  ]

  pool_types = [
    "max", 
    "mean", 
  ]
  attns = [
    False, 
    True
  ]

  env = launch_test_env(
    dataset_root='',
    enableds = CamType.WRIST | CamType.RIGHT_SHOULDER | CamType.LEFT_SHOULDER,
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
      "dataset_type",
      "is_multi_cnn",
      "lambda_attn",
      "colour_pool",
      "belows",
      "aboves",
    ], index = range(
      len(tasks) 
      * len(epochs) 
      * len(demo_counts) 
      * len(pool_types) 
      * len(cam_types)
      * len(attns)
      * len(attn_losses)
    )
  )

  ##fixed mb size this time
  dataset_params = {
    "demo": {
      "epochs": None,
      "minibatch_size": 10,
      "lr": 1e-3,
      "shuffle_obs_in_demo": False,
      "shuffle_data": True, ##let "demo" use shuffling and other things, that benefit its learning for more than 1 demo
      "dataset_to_use": "demo", 
      "lock_loader_seed": 42,
      "lambda_attn": None
    },
    "obs": {
      "epochs": None,
      "minibatch_size": 32,
      "lr": 1e-3,
      "shuffle_obs_in_demo": False,
      "shuffle_data": False, ##let "demo" use shuffling and other things, that benefit its learning for more than 1 demo
      "dataset_to_use": None, 
      "dataset_to_use": "obs", 
      "lock_loader_seed": 42,
      "lambda_attn": None,
      
    }
  }

  # fix the test dataset as this one

  task_env = env.get_task(ReachObs_Random)
  test_demos = load_demos_for(10, env,  ReachObs_Random, f"data/test/10demos")
  

  target = Shape("target")
  target_rgb = torch.tensor(target.get_color())

  make_agent = lambda ct, mc, pt: Agent(
    action_shape= env.action_shape[0],
    policy_type=PolicyType.CAM_ATTENTION,
    cam_type=ct,
    # config = "depth_ch",
    target_rgb = target_rgb, 
    is_multi_cnn = mc,
    colour_score_pooling = pt
  )

  count = 0 
  for task in tasks:
    for pt in pool_types:
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
            for mc in attns:
              for la in attn_losses:
                training_params = dataset_params["demo"]
                # if dt == "demo":
                #   training_params["minibatch_size"] = 10

                # training_params["dataset_to_use"] = dt  
                training_params["epochs"] = ep  
                training_params["lambda_attn"] = la

                agent = make_agent(ct, mc, pt)

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

                belows =[d["avg_attentions_below_obstacle"] for d in ret_dicts]
                aboves =[d["avg_attentions_above_obstacle"] for d in ret_dicts]

                df.loc[count] = {
                  "task_name": get_task_name(task), 
                  "cam_type": agent.cam_type, 
                  "epochs": ep,
                  "demo_count": dc, 
                  "max_eplens": eplens,
                  "done_count": done_count, 
                  "min_distance": sum(min_dists) / len(min_dists),
                  "final_distance": sum(final_dists) / len(final_dists),
                  "dataset_type": training_params["dataset_to_use"],
                  "is_multi_cnn": mc,
                  "lambda_attn": la,
                  "colour_pool": pt,
                  "belows": belows, 
                  "aboves": aboves,
                  
                }
                count += 1
                df.to_csv("cam_attn-maxpooling--ro-randoms-cam.csv", index=True)
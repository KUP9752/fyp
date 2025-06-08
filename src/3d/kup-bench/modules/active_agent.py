from typing import Literal, Optional
import numpy as np
import torch
import cv2
from rlbench.backend.robot import Robot
from rlbench.task_environment import TaskEnvironment
from rlbench.backend.observation import Observation

from scipy.spatial.transform import Rotation as rot

from pyrep.errors import ConfigurationError

from lib.agent import Agent
from lib.cam_type import CamType
from lib.utils import pick_obs_from_cam

class ActiveAgent_Plan1:
  def __str__(self) -> str:
    return f"active-agent-plan1:il-agent:{self.il_agent}"
  def __repr__(self) -> str:
    return f"ActiveAgent_Plan1(il_agent={repr(self.il_agent)})"
  
  def __init__(
    self, 
    action_dim: int, 
    il_agent: Agent,
    cam_type: CamType = CamType.WRIST | CamType.WRIST_DEPTH, ## rgb and point cloud esentially
    vis_thresh: float = 0.5,
  ):
    self.action_dim = action_dim
    self.cam_type = cam_type ## not really used
    self.il_agent = il_agent
    self.vis_thresh = vis_thresh
    
    self.best = {"vis_score": -1, "joint_pos": None }
    

  def _set_new_best(self, score: float, pose: np.ndarray): 
    self.best["vis_score"] = score
    self.best["joint_pos"] = pose
  
  def _get_best_joint_pos(self) -> Optional[np.ndarray]:
    return self.best["joint_pos"]

  ## TODO: this can combine multuple different things down the line
    # s_coverage = |target_pc| / max(|pc_seen_historcial|) to normalise for object size
    # proj_area = mask.sum() / |mask|
    ## spread = np.mean(np.linalg.norm(target_pc - centroid, axis = 1))
    # np.linalg.ei
  def visibility_score(self, 
    obs: Observation, ## NOTE: VisionSensor with no 'rel_to' param gives in terms of world
    hsv_low: np.ndarray = np.array([0, 177, 0]),
    hsv_high: np.ndarray = np.array([179, 255, 255]),
  ):
    ## change if trying other cams? dont think so tho
    rgb = obs.wrist_rgb
    world_pc = obs.wrist_point_cloud

    ## possible issues, object size per view, the object may be far
    # centroid = target_pc.mean(axis=0)
    target_pc, mask = self._get_masked_pc_and_mask(rgb, world_pc, hsv_low, hsv_high)

    ## currently a simple colour checking in the area
    
    return mask.sum() / mask.size ## h * w
    
  ## simple proportional control - can be made more advanced if needed
  def _velocity_action_from_pose(self, 
    robot: Robot,
    target_pose: np.ndarray, ## also 7-dims
    prop_gain: float = 1.,
  ) -> np.ndarray:
    curr_pose = robot.gripper.get_pose()
    ##make sure they are np arrays, they might be tensors at this point
    curr_pose = np.asarray(curr_pose, np.float32).reshape(-1) ## (7, ) -> (7)
    target_pose = np.asarray(target_pose, np.float32).reshape(-1) ## (7, ) -> (7)

    if curr_pose.shape[0] != 7 or target_pose.shape[0] != 7:
      raise RuntimeError(f"[active_agent - (joint_velocity_from_pose)] Error the joint poses are not 7-dims (only panda arm supported currently)")

    print(f"[joint_velocity_from_pose] {curr_pose = }")
    print(f"[joint_velocity_from_pose] {target_pose = }")
    

    error = target_pose - curr_pose
    vs = prop_gain * error 
    print(f"[joint_velocity_from_pose] velocities = {vs}")
    action = np.append(vs, [0.]) ## add the gripper move, dont care about it

    return action

  def _get_masked_pc_and_mask(self, 
    rgb: np.ndarray, 
    world_pc: np.ndarray, ## NOTE: VisionSensor with no 'rel_to' param gives in terms of world
    hsv_low: np.ndarray,
    hsv_high: np.ndarray,
  ) -> tuple:
    ## the vision_sensor gives the point cloud in world frame
    hsv =  cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high) ## mention in report how this cam to be
    mask = mask / 255 ## above gives in range [0, 255]

    pc_flat = world_pc.reshape(-1, 3)
    mask_flat = mask.reshape(-1).astype(bool)

    masked_pc = pc_flat[mask_flat]
    return masked_pc, mask

  def _sample_camera_poses(self, 
    ee_pos,
    n_samples: int, 
    radius: float = 0.005,
  ):
    poses = []
    for _ in range(n_samples):
      ## get position
      # uniform sampling on sphere surface around gripper pose
      theta = np.arccos(2 * np.random.rand() - 1)
      phi   = 2 * np.pi * np.random.rand()
      x_off = radius * np.sin(theta) * np.cos(phi)
      y_off = radius * np.sin(theta) * np.sin(phi)
      z_off = radius * np.cos(theta)
      off = np.array([x_off, y_off, z_off])
      pos = ee_pos + off


      ## get rotation
      ## TODO add math for this in the report
      # forward = ee_pos - pos ## same as - off
      forward = -off
      forward /= np.linalg.norm(forward)
      world_up = np.array([0.0, 0.0, 1.])
      right = np.cross(world_up, forward)

      ## handle case when up ~= forward
      if np.linalg.norm(right) < 1e-6:
          world_up = np.array([1.0, 0.0, 0.0])
          right = np.cross(world_up, forward)
      right /= np.linalg.norm(right)
      true_up = np.cross(forward, right)

      R_mat = np.stack([right, true_up, forward], axis=1)  # 3 x 3 rot
      quat = rot.from_matrix(R_mat).as_quat()  

      pose = np.zeros(7, dtype=float) ## xyz + quat
      pose[0:3] = pos
      pose[3:7] = quat
      poses.append(pose)

    return poses
  
  ## returns the joint positions to achieve pose: (7, )
  def _solve_ik(self, robot: Robot, pose: np.ndarray) -> Optional[np.ndarray]:
    try: 
      return robot.arm.solve_ik_via_sampling(
        position=pose[:3], # (x, y, z)
        quaternion=pose[3:] # (qx, qy, qz, qw)
      )
    except ConfigurationError as err:
      print(f"[active_agent (solve_ik)] Error: '{err}', continuing to sample")
      return None
      

  def act_il(self, 
    obs: Observation,
    task_env: TaskEnvironment,
    max_eplen: Optional[int] = None
  ) -> tuple[Observation, int, bool]: ##obs, reward, done
    done = False
    count = 0

    while not done:
      action, rets = self.il_agent.act(obs)
      action = action.squeeze(0)

      obs, _, done = task_env.step(action)
      if done:
        self.state = "done"
        return obs, _, done

      vis_score = self.visibility_score(obs)
      print(f"il - {vis_score = }")
      if vis_score < self.vis_thresh:
        self.state = "active"
        return obs, _, done
      
      if max_eplen is not None:
        if count >= max_eplen:
          self.state = "done"
          return obs, _, False ## not done fail this iteration get to active
        count += 1
      
  def act_active(self, 
    obs: Observation,
    task_env: TaskEnvironment, 
    pose_samples: int
  ) -> tuple[Observation, int, bool]: ##obs, reward, done
    robot = task_env._robot

    vis_score = self.visibility_score(obs)

    if vis_score >= self.vis_thresh:
      return obs, 0, False ## not sure placeholder values, i gueuss this sohuld do act_il?
      

    for pose in self._sample_camera_poses(
      robot.gripper.get_position(), 
      pose_samples
    ):
      ## finds the joint positions to get this pose
      jpos_target = self._solve_ik(robot, pose)

      if jpos_target is None:
        continue ## check next sample

      print(f"act_active {jpos_target.shape = } (from solve_ik)")
      jpos_target = jpos_target.squeeze(0)
      action =  self._velocity_action_from_pose(robot, jpos_target)

      print(f"act_active {action.shape = }")
      
      obs, _, done = task_env.step(action) ## step to new pose
      if done: 
        self.state = "done"
        return obs, _ , done

      s = self.visibility_score(obs)
      print(f"act_active - {vis_score = }")

      if s > self.best["vis_score"]:
        self._set_new_best(s, jpos_target)
    
    best_jpos = self._get_best_joint_pos()
    if  best_jpos is None:
      raise NotImplementedError("[active_agent - act] No better pose found, but I cant jsut give up here, needs to do something")
    ## otherwise move to best:
    ## when velocities get involved this may need more calculations
    action = self._velocity_action_from_pose(robot, best_jpos)

    return  task_env.step(action) ## get to the previous best, not sure, this made sense
    

  ## full action loop unlike the other agents
  ## call as .act(task_env.reset()[1], task_env)
  def act(self, 
    init_obs: Observation,
    task_env: TaskEnvironment,
    pose_samples: int = 20,
    max_loops: int = 200
  ) -> bool:
    
    done = False    
    obs = init_obs
    vis_score = self.visibility_score(obs)
    print(f"(start) act - {vis_score = }")

    if vis_score >= self.vis_thresh:
      self.state = "il"
    else:
      self.state = "active"

    ## look for a better pose until it finds a good enogh one to start il
    for _ in range(max_loops):
      if self.state == "active":
        obs, _ , done = self.act_active(obs, task_env, pose_samples)

        if done: 
          self.state = "done"
          break

        vis_score = self.visibility_score(obs)
        print(f"active - {vis_score = }")
        
        if vis_score >= self.vis_thresh:
          self.state = "il"
      
      if self.state == "il":
        ## act il should only return if the vis score drops significantly
        obs, _ , done = self.act_il(obs, task_env)
        if done: 
          self.state = "done"
          break

    return done


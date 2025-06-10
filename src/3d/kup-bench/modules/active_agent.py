from typing import Literal, Optional
import numpy as np
import torch
import cv2
from rlbench.backend.robot import Robot
from rlbench.task_environment import TaskEnvironment
from rlbench.backend.observation import Observation

from scipy.spatial.transform import Rotation as rot

from pyrep.errors import ConfigurationError
from pyrep.objects import Shape

from lib.agent import Agent
from lib.cam_type import CamType
from lib.utils import pick_obs_from_cam, GRIPPER_OPEN

import pdb

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
    sample_radius: float = 0.05
  ):
    self.action_dim = action_dim
    self.cam_type = cam_type ## not really used
    self.il_agent = il_agent
    self.vis_thresh = vis_thresh
    self.state = "init"
    self.sample_radius = sample_radius
    self.obstacle = Shape("obstacle")
    
    self.best = {"vis_score": -1, "joint_pos": None }
    

  def _set_new_best_pose(self, score: float, pose: np.ndarray): 
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

    # print(f"[joint_velocity_from_pose] {curr_pose = }")
    # print(f"[joint_velocity_from_pose] {target_pose = }")
    

    error = target_pose - curr_pose
    vs = prop_gain * error 
    # print(f"[joint_velocity_from_pose] velocities = {vs}")
    action = np.append(vs, [GRIPPER_OPEN]) ## add the gripper move, dont care about it

    return action
  
  def _apply_collision_safety(self, 
    robot: Robot,
    action: np.ndarray,
    obs: Observation,
    first_n_centre: int = 20,
    clearance: float = 0.2, # 20 cm  (min_pc_dist) and 10cm ignore so 10 cm after that
    blend: float = 0.02,
    gain: float = 0.1,
    min_pc_dist: float = 0.1 # want to ignore the grippers becuase they wil always be clsoe, gripper is about 0.0885 from the camera
  ) -> np.ndarray:
    
    gripper_pos = robot.gripper.get_position()

    pc = obs.wrist_point_cloud.reshape(-1, 3) ## get into (w*h, 3)
    target_pc, _ = self._get_masked_pc_and_mask(obs.wrist_rgb, obs.wrist_point_cloud)
    pc -= target_pc ## this is to exclude the target from this backoff
    ## calculate the distances from the gripper
    dists = np.linalg.norm(pc - gripper_pos, axis = 1)
    dists = dists[dists > min_pc_dist] # ignore the grippers that we can see
    idxs = np.argsort(dists)[:first_n_centre]
    centroid = pc[idxs].mean(axis = 0)
    min_dist = dists.min()


    ##repulse dir
    repulse_dir = gripper_pos - centroid
    repulse_dir /= np.linalg.norm(repulse_dir) ## make into a unit vector

    ## joint space to cartesion space
    J = robot.arm.get_jacobian() ## row-major (7, 6)
    
    cartesian_twist = np.zeros(6, dtype=np.float32)
    cartesian_twist[:3] = repulse_dir * gain ## linear translation away
    
    repulse_jvel = J @ cartesian_twist

    if min_dist <= clearance:
      alpha_blend = 1. ## full repulsion
    elif min_dist >= clearance + blend:
      alpha_blend = 0. ## no repulsion
    else:
      alpha_blend = (clearance + blend - min_dist) / blend

    ## calculate repulsive action from obstacle
    
    action[:7] = (1 - alpha_blend) * action[:7] + alpha_blend * repulse_jvel
    return action
  
  def _get_masked_pc_and_mask(self, 
    rgb: np.ndarray, 
    world_pc: np.ndarray, ## NOTE: VisionSensor with no 'rel_to' param gives in terms of world
    hsv_low: np.ndarray = np.array([0, 177, 0]),
    hsv_high: np.ndarray = np.array([179, 255, 255]),
  ) -> tuple:
    ## the vision_sensor gives the point cloud in world frame
    hsv =  cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high) ## mention in report how this cam to be
    mask = mask / 255 ## above gives in range [0, 255]

    pc_flat = world_pc.reshape(-1, 3)
    mask_flat = mask.reshape(-1).astype(bool)

    masked_pc = pc_flat[mask_flat]
    return masked_pc, mask
  
  # \(k\) from \({pc}^{world}\) where \( m_k \in \{\text{True, False}\} \) as \( \{ {pc}^{target}_k | m_k = True \} \) where $m$ comes from the aforementioned colour range extraction

  def _sample_camera_poses(self, 
    robot: Robot,
    n_samples: int, 
    radius: float,
    max_angle_rad = np.deg2rad(60),
    # k_sample: int = 10
  ) -> list[np.ndarray]:
    curr_pose = robot.gripper.get_pose()
    curr_pos = curr_pose[:3]
    curr_quat = curr_pose[3:]
    curr_rot = rot.from_quat(curr_quat)

    samples = []
    joint_sets = []
    for _ in range(n_samples):
      axis = np.random.rand(3) ## pick an axis
      axis /= np.linalg.norm(axis)

      angle = (np.random.rand() * 2 - 1) * max_angle_rad
      delta_rot = rot.from_rotvec(axis * angle)
      rot_new = delta_rot * curr_rot

      new_quat = rot_new.as_quat()
      new_pose = np.zeros(7, dtype=float) ## xyz + quat
      new_pose[0:3] = curr_pos
      new_pose[3:7] = new_quat
      samples.append((abs(angle), new_pose))
    

    poses = [pose for (_, pose) in sorted(samples, key = lambda s: s[0])]
    # return poses[:k_sample] ## only take top k
    return poses ## don't really care about top k here, rotation is cheap
  

  def _sample_camera_poses(self, 
    robot: Robot,
    n_samples: int, 
    radius: float,
    k_sample: int = 10
  ):
    
    ee_pos = robot.gripper.get_position()
    ## arbitrary pick maybe make this look towards the target later
    centre = ee_pos + np.array([0., 0., - radius])
    samples = []
    for _ in range(n_samples):
      ## get position
      # uniform sampling on sphere surface around gripper pose
      theta = np.arccos(2 * np.random.rand() - 1)
      phi   = 2 * np.pi * np.random.rand()
      x_off = radius * np.sin(theta) * np.cos(phi)
      y_off = radius * np.sin(theta) * np.sin(phi)
      z_off = radius * np.cos(theta)
      off = np.array([x_off, y_off, z_off])
      pos = centre + off


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
      samples.append((np.linalg.norm(pos - ee_pos), pose))

    poses = [pose for (_, pose) in sorted(samples, key = lambda s: s[0])]
    return poses[:k_sample] ## take closest k
  
  ## returns the joint positions to achieve pose: (7, )
  def _solve_ik(self, robot: Robot, pose: np.ndarray) -> Optional[np.ndarray]:
    try: 
      return robot.arm.solve_ik_via_sampling(
        position=pose[:3], # (x, y, z)
        quaternion=pose[3:],  # (qx, qy, qz, qw),
        ignore_collisions=True
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
    robot = task_env._robot
      
    while not done:
      action, rets = self.il_agent.act(obs)
      action = action.squeeze(0)

      obs, _, done = task_env.step(action)
      if done:
        self.state = "done"
        return obs, _, done

      vis_score = self.visibility_score(obs)
      print(f"il - {vis_score = }")

      ## mark if a good view is found
      if vis_score > self.best["vis_score"]:
        print(f"il - adding to best pose, {vis_score = }")
        self._set_new_best_pose(vis_score, robot.gripper.get_pose())

      if vis_score < self.vis_thresh:
        self.state = "active"
        return obs, _, done
      
      if max_eplen is not None:
        if count >= max_eplen:
          self.state = "done-il"
          return obs, _, False ## not done fail this iteration get to active
        count += 1
      
  def act_active(self, 
    obs: Observation,
    task_env: TaskEnvironment, 
    pose_samples: int
  ) -> tuple[Observation, int, bool]: ##obs, reward, done
    robot = task_env._robot
    done = False

    ## we already know the vis score is low if we are here
    for pose in self._sample_camera_poses(
      robot, 
      pose_samples, 
      radius = self.sample_radius
    ):
      ## finds the joint positions to get this pose
      jpos_target = self._solve_ik(robot, pose)

      if jpos_target is None:
        continue ## check next sample, already printing the error in _solve_ik

      # print(f"act_active {jpos_target.shape = } (from solve_ik)")
      jpos_target = jpos_target.squeeze(0)
      action =  self._velocity_action_from_pose(robot, jpos_target)

      # print(f"act_active {action.shape = }")
      
      obs, _, done = task_env.step(action) ## step to new pose

      if done: 
        self.state = "done"
        return obs, _ , done

      vis_score = self.visibility_score(obs)
      print(f"act_active1 - {vis_score = }")

      if vis_score > self.best["vis_score"]:
        print(f"active - adding to best pose, {vis_score = }")
        self._set_new_best_pose(vis_score, robot.gripper.get_pose()) ## if the visuals here are good, mark this to fall back to later
    
    best_jpos = self._get_best_joint_pos()
    ## this only hits in the first 'active' loop when no sample can be reached (certainly possible as the dam solve always fails)
    if  best_jpos is None:
      ## not sure if we have no views, maybe force one when close to t he obstacle so we don't got too far back?
      return obs, -1,  done ## -1 for error, no one cares about the int anyway
      
    ## otherwise move to best:
    ## when velocities get involved this may need more calculations
    action = self._velocity_action_from_pose(robot, best_jpos)

    return task_env.step(action) ## get to the previous best, not sure, this made sense
    

  ## full action loop unlike the other agents
  ## call as .act(task_env.reset()[1], task_env)
  def act(self, 
    init_obs: Observation,
    task_env: TaskEnvironment,
    pose_samples: int = 100,
    max_loops: int = 5,
    within_obs: float = 0.1 # this is gonna be the vertical distance, I wanna see when we are about level with the obstacle
  ) -> tuple[bool, dict]:
    
    self.obstacle = Shape("obstacle") ## this was not initialising properly for some reason

    done = False    
    obs = init_obs
    robot = task_env._robot

    self.state = "init"
    get_z_dist = lambda: np.linalg.norm(self.obstacle.get_position()[2] - robot.gripper.get_position()[2])
    
    ## We want to move close to the obstacle first
    if self.state == "init": ## just started and are at the very top
      # pdb.set_trace()
      while get_z_dist() >= within_obs:
        
        action, rets = self.il_agent.act(obs)
        action = action.squeeze(0)

        ## in case we are getting too close (clearance is gretar than the limit in while)
        action = self._apply_collision_safety(robot, np.asarray(action), obs)

        obs, _, done = task_env.step(action) ## cannot be done here but will ad shortcut
        if done: 
          self.state = "done"
          break
      print(f"Reached within {within_obs} of obstacle")
      self.state = "start"

    # return False, {}
  
    if done:
      print(f"Done somehow before any active vision")
      return done, {"when": "done trying to reach the obstacle"}
      
    for _ in range(max_loops):
      if done:
        return done, {"when": "during the max loops", "last_state": self.state}

      vis_score = self.visibility_score(obs)
      print(f"(start) act - {vis_score = }")

      if vis_score >= self.vis_thresh:
        self.state = "il"
        obs, _, done = self.act_il(obs, task_env)
      else:
        self.state = "active"
        obs, _ , done = self.act_active(obs, task_env, pose_samples)

    return done, {"when": "At the end did not complete"}
          

      
        




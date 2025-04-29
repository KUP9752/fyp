from typing import List, Optional
from rlbench.backend.task import Task
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors as colours
from pyrep.objects import Object

from math import radians
from pyquaternion import Quaternion


class ReachTargetObsRandomTracking(Task):
    ## Random: The obstacle and the target are placed randomly, 
    ## first place the obstacle (which is bound to the bounding box `target_boundary`)
    ## so the target can then be placed within this carried bounding box, ensures the target is mostly covered by the obstacle if not always
    def init_task(self) -> None:
      self.target = Shape("target")
      self.obs = Shape("obstacle")
      self.target_w = Dummy("waypoint0")
      
      success_sensor =  ProximitySensor("success")
      self.register_success_conditions([
        DetectedCondition(self.robot.arm.get_tip(), success_sensor)
      ])
      
      self.target_boundary = Shape("target_boundary")
      self.obs_boundary = Shape("obs_boundary")
      
      
    ## The obstacle and the target are independently sampled from 2 different boundaries, 
    ## which are both within the view of the wrist camera
    def init_episode(self, index: int) -> List[str]:
      
      ## pick from candidates to request a demo from specific direction of object
      SpawnBoundary([self.obs_boundary]).sample(
        self.obs,
        ignore_collisions=False, 
        min_distance = 0, #min distance doesn't matter no other objects yet
        min_rotation = (0, 0, 0),
        max_rotation = (0, 0, 0)
        ## TODO: add rotation to the obstacle for more dynamics?
      )
      
      SpawnBoundary([self.target_boundary]).sample(
        self.target,
        ignore_collisions=False, 
        min_distance = 0, #min distance doesn't matter no other objects yet
        min_rotation = (0, 0, 0),
        max_rotation = (0, 0, 0)
      )
      
      
      return [f"reach the sphere target behind an obstacle"]
    
    ## I want to move the wrist (and hence the wrist camera) to look at the obstacle on every step 
    ## once the tip is under the obstacle
    def step(self):
      ## x y z coords of tip relative to obstacle
      print("in step")
      tip_to_obs = self.robot.arm.get_tip().get_position(relative_to=self.obs)
      
      ## z-axis is up-down, check if below
      if tip_to_obs[2] <= 0:
        print("BELOW THE TING")
        
        orientation = self.robot.arm.get_tip().get_orientation(relative_to=self.target_w) 
        tip = self.robot.arm.get_tip()
        
        q_tip = Quaternion(tip.get_quaternion())
        q_target = Quaternion(self.target_w.get_quaternion())
        q_rot = Quaternion(axis=[0, 0, 1], angle=radians(90))
        
        q_step = Quaternion.slerp(q_tip, q_target, amount=0.1)
        q_new = q_tip * q_rot
        ## (x, y, z) + (Qx, Qy, Qz, Qw)
        pose = list(tip.get_position()) + list(q_new.elements)
        
        print(pose)
        ## needs (x, y, z, Qx, Qy, Qz, Qw)
        self.robot.arm._ik_target.set_pose(pose)
        for i in range(10000000):
          pass
      
      
      
    def variation_count(self) -> int:
      # TODO: The number of variations for this task.
      return 1 ## add more colours distractors etc?
      
    def base_rotation_bounds(self):
      return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    
    def is_static_workspace(self):
      return True
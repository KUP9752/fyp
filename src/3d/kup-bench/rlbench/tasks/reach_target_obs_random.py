from typing import List, Optional
from rlbench.backend.task import Task
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import DetectedCondition
from rlbench.const import colors as colours
from pyrep.objects import Object
from rlbench.backend.spawn_boundary import SpawnBoundary

class ReachTargetObsRandom(Task):

    def init_task(self) -> None:
      self.target = Shape("target")
      self.obs = Shape("obstacle")
      
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
    

    def variation_count(self) -> int:
      # TODO: The number of variations for this task.
      return 1 ## add more colours distractors etc?
      
    def base_rotation_bounds(self):
      return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    
    def is_static_workspace(self):
      return True
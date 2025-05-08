from typing import List, Optional
from rlbench.backend.task import Task
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.backend.conditions import DetectedCondition
from rlbench.const import colors as colours
from pyrep.objects import Object

from rlbench.backend.spawn_boundary import SpawnBoundary

from pyrep import PyRep
from rlbench.backend.robot import Robot


import numpy as np
MAX_SCALE = 1.3 ## anything larger cannot be grasped
MIN_SCALE = 0.3 ## arbitrary didn't want it too small


## from wrist_cam
MAX_Z_DIST = 0.8 ## not sure fully try these
MIN_Z_DIST = 0.3 ## need to get closer for experiments??

class VisionStatic(Task):
  
    def __init__(self,
      pyrep: PyRep,
      robot: Robot,
      name: Optional[str] = None,
      scale: Optional[float] = None
    ): 
      super().__init__(pyrep, robot, name)
      self.scale = scale

    def init_task(self) -> None:
      self.grasp_target = Shape("grasp_target")
      self.grasp_target_visual = Shape("grasp_cube")
      self.wrist_cam = VisionSensor("cam_wrist")
      
      # success_sensor =  ProximitySensor("success")
      self.register_graspable_objects([self.grasp_target])
      self.boundary = Shape("boundary")
      # self.register_success_conditions([
      #   DetectedCondition(self.robot.arm.get_tip(), success_sensor)
      # ])
      
    def init_episode(self, index: int) -> List[str]:
      ## currently randomises colour, but we might not want that initially
      if self.scale:
        self.picked_scale = self.scale
      else:
        self.picked_scale = np.random.uniform(MIN_SCALE, MAX_SCALE, (1))[0]
        
      print(f"scale: {self.picked_scale} {'picked' if self.scale else ''}")
      self.grasp_target_visual.scale_object(
        self.picked_scale,
        self.picked_scale,
        self.picked_scale
      )
      self.grasp_target.set_position(
        [0, 0, 0.25],
        relative_to=self.wrist_cam
      )
      
      #  # create a spawn boundary
      # sb = SpawnBoundary([self.boundary])
      # sb.sample(
      #   self.grasp_target, 
      #   ignore_collisions = False,
      #   min_distance = 0.09,
      #   min_rotation = (0, 0, 0),
      #   max_rotation = (0, 0, 0)
      # ) 
      
      return [f"Vision_Static desc"]
    
    ## revert back to original size
    def cleanup(self) -> None:
      inv = 1. / self.picked_scale
      self.grasp_target_visual.scale_object(
        inv, inv, inv
      )
    
    def variation_count(self) -> int:
      # TODO: The number of variations for this task.
      return 3 ## bigger or smaller than starting
      
    def base_rotation_bounds(self):
      return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    
    def is_static_workspace(self):
      return True
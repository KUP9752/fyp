from typing import List, Optional
from rlbench.backend.task import Task
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import DetectedCondition
from rlbench.const import colors as colours
from pyrep.objects import Object
from rlbench.backend.spawn_boundary import SpawnBoundary

class ReachTargetObsRandomStatic(Task):
    ## Random: the waypoint generation is left free from the start, 
    ##  depending on rlbench pathfinding any demo can be given
    ## Static: the target nor the obstacle moves 
    def init_task(self) -> None:
      self.target = Shape("target")
      success_sensor =  ProximitySensor("success")
      self.boundary = Shape("boundary")
      self.register_success_conditions([
        DetectedCondition(self.robot.arm.get_tip(), success_sensor)
      ])
      
      
    def init_episode(self, index: int) -> List[str]:
      
       
      return [f"reach the sphere target behind an obstacle"]
    

    def variation_count(self) -> int:
      return 1
      
    def base_rotation_bounds(self):
      return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    
    def is_static_workspace(self):
      return True
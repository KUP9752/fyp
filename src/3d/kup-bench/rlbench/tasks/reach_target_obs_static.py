from typing import List, Optional
from rlbench.backend.task import Task
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import DetectedCondition
from rlbench.const import colors as colours
from pyrep.objects import Object
from rlbench.backend.spawn_boundary import SpawnBoundary

class ReachTargetObsStatic(Task):

    def init_task(self) -> None:
      self.target = Shape("target")
      success_sensor =  ProximitySensor("success")
      self.boundary = Shape("boundary")
      self.register_success_conditions([
        DetectedCondition(self.robot.arm.get_tip(), success_sensor)
      ])
      
      self.candidate_ws = [
        Dummy("waypointLeft"),
        Dummy("waypointDown"),
        Dummy("waypointRight"),
      ]
      
      self.directions = [
        "Left",
        "Down",
        "Right"
      ]
      self.ep_pick: Optional[int] = None
      
    def init_episode(self, index: int) -> List[str]:
      print(f"Variation: {index}")
      
      ## pick from candidates to request a demo from specific direction of object
      self.ep_pick = index 
      direction = self.directions[index]
      w0 = self.candidate_ws[index]
      w0.set_name("waypoint0")
      
       
      return [f"reach the sphere target behind an obstacle", f"reach the target by moving {direction} of obstacle"]
    
    ## we changed the name of the waypoint reset it back, apparently this changes the ttm file
    def cleanup(self) -> None:
      if self.ep_pick is None:
        raise ValueError("'ep_pick' should be set, as 'cleanup' is always called at the end of an episode")
      
      ##restore the name of the picked waypoint
      picked_dir = self.directions[self.ep_pick]
      self.candidate_ws[self.ep_pick].set_name(f"waypoint{picked_dir}")
      
      self.ep_pick = None # set to `None` for debugging purposes

    def variation_count(self) -> int:
      return 3
      
    def base_rotation_bounds(self):
      return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    
    def is_static_workspace(self):
      return True
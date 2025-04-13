from typing import List
from rlbench.backend.task import Task
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.const import colors as colours
from pyrep.objects import Object
from rlbench.backend.spawn_boundary import SpawnBoundary

class SimpleGrasp(Task):

    def init_task(self) -> None:
      self.grasp_target = Shape("grasp_target")
      self.grasp_target_visual = Shape("grasp_cube")
      # success_sensor =  ProximitySensor("success")
      self.register_graspable_objects([self.grasp_target])
      self.boundary = Shape("boundary")
      # self.register_success_conditions([
      #   DetectedCondition(self.robot.arm.get_tip(), success_sensor)
      # ])
    def init_episode(self, index: int) -> List[str]:
      ## currently randomises colour, but we might not want that initially
      color_name, color_rgb = colours[index]
      self.grasp_target_visual.set_color(color_rgb)      
      
      
      ## create a spawn boundary
      sb = SpawnBoundary([self.boundary])
      sb.sample(
        self.grasp_target, 
        ignore_collisions = False,
        min_distance = 0.09,
        min_rotation = (0, 0, 0),
        max_rotation = (0, 0, 0)
      ) 
      
      return [f"simple grasping task test, to try out grasping"]

    def variation_count(self) -> int:
      # TODO: The number of variations for this task.
      return len(colours)
      
    def base_rotation_bounds(self):
      return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    
    def is_static_workspace(self):
      return True
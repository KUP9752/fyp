from typing import List
from rlbench.backend.task import Task
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.const import colors as colours
from pyrep.objects import Object
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import NothingGrasped

class GraspAndMove(Task):

    def init_task(self) -> None:
      self.grasp_cube = Shape("grasp_target")
      self.grasp_target_visual = Shape("grasp_cube")
      self.move_target = Shape("move_target")
      # success_sensor =  ProximitySensor("success")
      self.register_graspable_objects([self.grasp_cube])
      self.grab_boundary = Shape("grab_boundary")
      self.move_boundary = Shape("move_boundary")
      success_sensor = ProximitySensor("success")
      self.register_success_conditions([
        DetectedCondition(self.grasp_target_visual, success_sensor),
        NothingGrasped(self.robot.gripper) ## makes sure the gripper drops the item (this is done in the `Extension string of waypoint3`)
      ])
    def init_episode(self, index: int) -> List[str]:
      ## create a spawn boundary
      ## TODO: change if we want different colours
      # color_name, color_rgb = colours[index]
      # self.grasp_target_visual.set_color(color_rgb)      
      color_name = self.grasp_target_visual.get_color()
      
      SpawnBoundary([self.grab_boundary]).sample(
        self.grasp_cube, 
        ignore_collisions = False,
        min_distance = 0.09,
        min_rotation = (0, 0, 0),
        max_rotation = (0, 0, 0)
      )
      
      SpawnBoundary([self.move_boundary]).sample(
        self.move_target, 
        ignore_collisions = False,
        min_distance = 0.09,
        min_rotation = (0, 0, 0),
        max_rotation = (0, 0, 0)
      )
      return [f"grasp the {color_name} target and move it to the {self.move_target.get_color()} marked location"]

    def variation_count(self) -> int:
      # TODO: The number of variations for this task.
      # return len(colours)
      return 1 ## TODO: change if we want different colours
      
    def base_rotation_bounds(self):
      return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    
    def is_static_workspace(self):
      return True
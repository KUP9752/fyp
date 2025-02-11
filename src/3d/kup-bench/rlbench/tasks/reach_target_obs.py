from typing import List
from rlbench.backend.task import Task
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.const import colors as colours

class ReachTargetObs(Task):

    def init_task(self) -> None:
      self.target = Shape("target")
      success_sensor =  ProximitySensor("success")
      self.register_success_conditions([
        DetectedCondition(self.robot.arm.get_tip(), success_sensor)
      ])
    def init_episode(self, index: int) -> List[str]:
      color_name, color_rgb = colours[index]
      self.target.set_color(color_rgb)      
      return [f"reach the {color_name} target", f"reach the {color_name} thing", f"reach the {color_name} sphere"]

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return len(colours)
      
    def base_rotation_bounds(self):
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
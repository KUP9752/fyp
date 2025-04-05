from typing import List
import numpy as np

from rlbench.backend.task import Task

from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape

from rlbench.backend.conditions import DetectedCondition
from rlbench.const import colors as colours
from rlbench.backend.spawn_boundary import SpawnBoundary

class RandomReach(Task):

    def init_task(self) -> None:
      self.target = Shape("target")
      self.d0 = Shape("d0")
      self.d1 = Shape("d1")
      sensor = ProximitySensor("success")
      self.boundary = Shape("boundary")
      
      self.register_success_conditions([
        DetectedCondition(self.robot.arm.get_tip(), sensor)
      ])

    def init_episode(self, index: int) -> List[str]:
      
      ## colour the distractors
      colour_choices = np.random.choice(
        list(range(index)) + list(range(index + 1, len(colours))),
        size = 2, 
        replace = False
      )
      for d, i in zip([self.d0, self.d1], colour_choices):
        col_name, col_rgb = colours[i]
        d.set_color(col_rgb)
      
      col_name, col_rgb = colours[index]
      self.target.set_color(col_rgb)
      
      b = SpawnBoundary([self.boundary])
      for ob in [self.target, self.d0, self.d1]:
        b.sample(
          ob,
          min_distance = 0.2,
          min_rotation = (0, 0, 0),
          max_rotation = (0, 0, 0)
        )
      
      return [f'reach the target {col_name} target']



    def variation_count(self) -> int:
      ## variations are the different colours
      return len(colours)

    def base_rotation_bounds(self):
      return [0., 0., 0.], [0., 0., 0.]

    def is_static_workspace(self):
      return True
from cam_type import CamType
from pyrep.objects import VisionSensor, Shape

import numpy as np


## This is a simple colour rbg value check, not the best maybe use depth later
def check_visibility(view_handle: str, target_handle: str, tolerance = 0.1):
  cam = VisionSensor(view_handle)
  target = Shape(target_handle)

  rgb = cam.capture_rgb()
  target_rbg = target.get_color()
  
  mask = np.all(np.abs(rgb - target_rbg) < tolerance, axis = -1)
  visible_pxs = np.count_nonzero(mask)
  print(f"{visible_pxs = }")
  print(f"{mask.size =}")
  
  return visible_pxs / mask.size



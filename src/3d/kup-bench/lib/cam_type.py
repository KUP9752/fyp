from enum import Flag, auto
from typing import Self

class CamType(Flag):
  WRIST = auto()
  LEFT_SHOULDER = auto()
  RIGHT_SHOULDER = auto()
  WRIST_DEPTH = auto() ## NOTE: closer means lower float value, in meters i think
  OVERHEAD = auto() ## NOTE: closer means lower float value, in meters i think
  FRONT = auto() ## NOTE: closer means lower float value, in meters i think
  
  def __str__(self):
    parts = []
    if self & CamType.WRIST:
      parts.append("wrist")
    if self & CamType.LEFT_SHOULDER:
      parts.append("l_shoulder")
    if self & CamType.RIGHT_SHOULDER:
      parts.append("r_shoulder")
    if self & CamType.WRIST_DEPTH:
      parts.append("wrist_depth")
    if self & CamType.OVERHEAD:
      parts.append("overhead")
    if self & CamType.FRONT:
      parts.append("front")
    
    return "+".join(parts)
  
  def is_single_type(self) -> bool:
    return self.value != 0 and self.value & (self.value - 1) == 0
  
  @classmethod
  ## recreates everytime, but couldn't find a good way to cache
  def all_combinations(cls) -> list[Self]:
    all_combs = []
    for i in range(1, 2**len(CamType)):
      comb = CamType(0)
      for j in range(len(CamType)):
        if i & (1 << j):
          comb |= CamType(1 << j)
      all_combs.append(comb)
    return all_combs

  @classmethod
  ## in order returns the existing CamTypes
  def uniques(cls):
    return [CamType(1 << i) for i in range(len(CamType))]
    
    
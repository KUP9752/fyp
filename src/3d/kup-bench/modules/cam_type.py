from enum import Flag, auto

class CamType(Flag):
  WRIST = auto()
  LEFT_SHOULDER = auto()
  RIGHT_SHOULDER = auto()
  
  def __str__(self):
    parts = []
    if self & CamType.WRIST:
      parts.append("wrist")
    if self & CamType.LEFT_SHOULDER:
      parts.append("l_shoulder")
    if self & CamType.RIGHT_SHOULDER:
      parts.append("r_shoulder")
    
    return "+".join(parts)
  
  @classmethod
  ## recreates everytime, but couldn't find a good way to cache
  def all_combinations(cls):
    all_combs = []
    for i in range(1, 2**len(CamType)):
      comb = CamType(0)
      for j in range(len(CamType)):
        if i & (1 << j):
          comb |= CamType(1 << j)
      all_combs.append(comb)
    return all_combs

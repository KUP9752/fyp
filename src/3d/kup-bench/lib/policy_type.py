from enum import Enum, auto

class PolicyType(Enum):
  SIMPLE = auto()
  CAM_ATTENTION = auto()
  
  def __str__(self) -> str:
    match self:
      case PolicyType.SIMPLE:
        return "simple_policy"
      case PolicyType.SIMPLE:
        return "cam_attn_policy"
      case _:
        raise NotImplementedError("[policy_type - str] Add the string representations of other policies")

        


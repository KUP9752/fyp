from enum import Enum, auto

class PolicyType(Enum):
  SIMPLE = auto()
  SIMPLE_GRASP = auto()
  CAM_ATTENTION = auto()
  
  def __str__(self) -> str:
    match self:
      case PolicyType.SIMPLE:
        return "simple_policy"
      case PolicyType.SIMPLE_GRASP:
        return "simple_grasp_policy"
      case PolicyType.CAM_ATTENTION:
        return "cam_attn_policy"
      case _:
        raise NotImplementedError("[policy_type - str] Add the string representations of other policies")

        


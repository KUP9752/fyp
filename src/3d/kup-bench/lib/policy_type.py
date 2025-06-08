from enum import Enum, auto

class PolicyType(Enum):
  SIMPLE = auto()
  SIMPLE_GRASP = auto()
  DEPTH_GRASP = auto()
  RESNET_GRASP = auto()
  RNN_GRASP = auto()
  FUSING = auto()
  FUSING_RNN = auto()
  CAM_ATTENTION = auto()
  
  def __str__(self) -> str:
    match self:
      case PolicyType.SIMPLE:
        return "simple_policy"
      case PolicyType.SIMPLE_GRASP:
        return "simple_grasp_policy"
      case PolicyType.DEPTH_GRASP:
        return "depth_grasp_policy"
      case PolicyType.RESNET_GRASP:
        return "resnet_grasp_policy"
      case PolicyType.RNN_GRASP:
        return "rnn_grasp_policy"
      case PolicyType.FUSING:
        return "fusing_policy"
      case PolicyType.FUSING_RNN:
        return "fusing_rnn_policy"
      case PolicyType.CAM_ATTENTION:
        return "cam_attn_policy"
      case _:
        raise NotImplementedError("[policy_type - (str)] Add the string representations of other policies")

        


from typing import Optional
import numpy as np

import torch.nn as nn

from lib.cam_type import CamType

from modules.cnns.cnn_encoder import CNNEncoder

    
class MultiCamCnn(nn.Module):
  def __init__(self, cam_type: CamType):
    ## here we want a combination of cam_types, single is also possible
    super(MultiCamCnn, self).__init__()
    
    cnns = {}
    if cam_type & CamType.WRIST:
      cnns[f"{CamType.WRIST}"] = CNNEncoder(in_channels = 3)
    if cam_type & CamType.LEFT_SHOULDER:
      cnns[f"{CamType.LEFT_SHOULDER}"] = CNNEncoder(in_channels = 3)
    if cam_type & CamType.RIGHT_SHOULDER:
      cnns[f"{CamType.RIGHT_SHOULDER}"] = CNNEncoder(in_channels = 3)
    if cam_type & CamType.WRIST_DEPTH:
      cnns[f"{CamType.WRIST_DEPTH}"] = CNNEncoder(in_channels = 1)
    
    self.out_shape = (128, 2, 2) ## this is per CNNEncoder
    
    ## ModuleDict indexing must be done with strings
    self.conv_encodes = nn.ModuleDict(cnns)
    
  def forward(self, image, cam_type: CamType):
    
    assert cam_type.is_single_type(), f"[multi_cam_cnn] The camera type given '{cam_type}' has multiple cameras in it"
    
    return self.conv_encodes[f"{cam_type}"](image) #type: ignore lets see if this works
    


        
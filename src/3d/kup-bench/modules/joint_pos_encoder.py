import torch 
import torch.nn as nn
import torch.nn.functional as F

class JointPosEncoder(nn.Module):
  def __init__(self, 
    input_dim: int = 7, ## rlbench default
    hidden_layers: list[int] = [64],
    output_size = 128
   ):
    super(JointPosEncoder, self).__init__()
    self.output_size = output_size
    dims = [input_dim] + hidden_layers + [output_size]


    layers = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
      layers.append(nn.Linear(in_dim, out_dim))
      layers.append(nn.ReLU())
      ## NOTE: BatchNorm is not used as this is not a very deep network can use if training is unstable or not smooth

    layers = layers[:-1] ## remove the final relu
                
    self.fc = nn.Sequential(*layers)

  def forward(self, joint_angles):
    # joint_angles: (B, 7)

    return self.fc(joint_angles), {}
  

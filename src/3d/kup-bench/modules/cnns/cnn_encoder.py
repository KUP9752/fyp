import torch.nn as nn

class CNNEncoder(nn.Module):
  def __init__(self,
    in_channels = 3  
  ):
    super(CNNEncoder, self).__init__()
    self.conv_encode = nn.Sequential(
      # 3 64 64
      nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      # 32 31 31 
      nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      # 48 14 14
      nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      # 64 6 6
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      # 128 2 2 : (B, 128, 2, 2)
    )
    
    
  ## given an image batch with multi cams (batch_size, num_cams, c, w, h)
  def forward(self, image):
    return self.conv_encode(image)
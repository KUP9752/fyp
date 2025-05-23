import torch.nn as nn

class CNNEncoder(nn.Module):
  def __init__(self,
    in_channels = 3,
    layers = [32, 48, 64, 128], ## generified, but this is default still
    use_batchnorm = True, 

  ):
    super(CNNEncoder, self).__init__()
    self.conv_encode = nn.Sequential(
      # 3 64 64
      nn.Conv2d(in_channels=in_channels, out_channels=layers[0], kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.BatchNorm2d(layers[0]) if use_batchnorm else nn.Identity(),
      nn.ReLU(inplace=False),
      # layers[0] 31 31 
      nn.Conv2d(in_channels=layers[0], out_channels=layers[1], kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.BatchNorm2d(layers[1]) if use_batchnorm else nn.Identity(),
      nn.ReLU(inplace=False),
      # layers[1] 14 14
      nn.Conv2d(in_channels=layers[1], out_channels=layers[3], kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.BatchNorm2d(layers[3]) if use_batchnorm else nn.Identity(),
      nn.ReLU(inplace=False),
      # layers[3] 6 6
      nn.Conv2d(in_channels=layers[3], out_channels=layers[3], kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.BatchNorm2d(layers[3]) if use_batchnorm else nn.Identity(),
      nn.ReLU(inplace=False),
      # layers[3] 2 2 : (B, layers[3], 2, 2)
    )
    
    
  ## given an image batch with multi cams (batch_size, num_cams, c, w, h)
  def forward(self, image):
    return self.conv_encode(image)
  

## to be used with attention, the above encodere is too coarse
class ConvEncoder(nn.Module):
    def __init__(self, 
        in_channels,
        base_channels = 32,
        out_channels = 128,
      ):
      super().__init__()
      # Calculate number of downsampling steps needed
      # Input assumed 64x64, we use conv+pool twice to go to 16x16
      self.encoder = nn.Sequential(
        # Block 1: 64x64 -> 32x32
        nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(base_channels),
        nn.MaxPool2d(2), ## NOTE: this is the same as `nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)` less explicit though
        nn.ReLU(inplace=True),
        # Block 2: 32x32 -> 16x16
        nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(base_channels*2),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        # Block 3: keep at 16x16, expand channels
        nn.Conv2d(base_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2), 
        ## output is 8x8 now, 16x16 was too memory inefficient (needed > 24gigs, so I am cutting now)
        nn.ReLU(inplace=True),
      )

    def forward(self, x):
      return self.encoder(x)  # (B, out_channels, 16, 16)
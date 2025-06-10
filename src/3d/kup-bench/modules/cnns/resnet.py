import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet

FALLBACK = "resnet18"

class ResNetEncoder(nn.Module):
  def __init__(self,
    in_channels = 5,
    resnet_name: str = FALLBACK,
    kernel_size = None,
  ):
    super(ResNetEncoder, self).__init__()
    
    try: 
      self.feats: ResNet = getattr(models.resnet, resnet_name)(weights = None)
      self.resnet_name = resnet_name
    except AttributeError:
      print(f"[resnet - (ResNetFeats)] WARNING: '{resnet_name}' does not exist, falling back to 'resnet18")
      self.feats: ResNet = getattr(models.resnet, FALLBACK)(weights = None)
      self.resnet_name = FALLBACK


    ## modifying to add whatever in channels I want

    og: nn.Conv2d = self.feats.conv1 
    self.feats.conv1 = self.conv1 = nn.Conv2d(
      in_channels=in_channels, 
      out_channels=og.out_channels, 
      kernel_size=kernel_size if kernel_size is not None else og.kernel_size, 
      stride=og.stride, 
      padding=og.padding, 
      bias=og.bias is not None
    )

    ## disabling the class prediction
    self.feats.avgpool = nn.Identity()
    self.feats.fc = nn.Identity()


  def forward(self, image):
    x = self.feats.conv1(image)
    x = self.feats.bn1(x)
    x = self.feats.relu(x)
    x = self.feats.maxpool(x)

    x = self.feats.layer1(x)
    x = self.feats.layer2(x)
    x = self.feats.layer3(x)
    x = self.feats.layer4(x)
    ## NOTE: resnet has a classification head after this
    return x
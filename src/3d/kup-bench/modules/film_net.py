import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    def __init__(
      self, 
      in_channels: int, 
      cond_dim: int,
      identity_start: bool = True
    ):
      super().__init__()
      # MLP that outputs 2*in_channels (for gamma and beta)
      self.film_gen = nn.Linear(cond_dim, 2 * in_channels)

      if identity_start:
        nn.init.zeros_(self.film_gen.bias)
        with torch.no_grad():
          # set gamma bias init to 1, beta bias init to 0, so start close to identity modulation
          self.film_gen.bias[:in_channels] = 1.0

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
      ## FiLM = gamma * X + beta
      ## cond: (B, cond_dim)
      
      B, ch, h, w = x.shape

      gamma_beta = self.film_gen(cond) #(B, 2ch)
      gamma, beta = gamma_beta.chunk(2, dim=1)## each (B, ch)

      gamma = gamma.view(B, ch, 1, 1)
      beta  = beta.view(B, ch, 1, 1)

      out = gamma * x + beta ##(B, ch, h, w)
      return out


class FilmModulator(nn.Module):
  def __init__(self,
    in1: int, ## modulated
    in2: int, ##modulatee
    feat_dim: int = 64,
    do_both: bool = False
  ):
    super(FilmModulator, self).__init__()
    self.do_both = do_both
  
    ## will be modulating 1 with 2, does notdo any down sampling modulate on entire resolution
    self.enc1 = ShallowEncoder(in1, out_channels=feat_dim)
    self.enc2 = ShallowEncoder(in2, out_channels=feat_dim)

    self.one_on_two = FiLMLayer(in_channels=feat_dim, cond_dim=feat_dim)

    if self.do_both:
      self.two_on_one = FiLMLayer(in_channels=feat_dim, cond_dim=feat_dim)

  def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

    x1feats: torch.Tensor = self.enc1(x1)
    x2feats: torch.Tensor = self.enc2(x2)

    x2pool = F.adaptive_avg_pool2d(x2feats, (1, 1)).view(x2feats.size(0), -1)
    if self.do_both:
      x1pool = F.adaptive_avg_pool2d(x1feats, (1, 1)).view(x1feats.size(0), -1)
      return self.one_on_two(x1feats, x2pool), self.two_on_one(x2feats, x1pool)
    
    ## modulate x1 features on x2


    return self.one_on_two(x1feats, x2pool)


## TODO this might be too large, maybe do only 1 level or even smaller steps
class ShallowEncoder(nn.Module):
  def __init__(self, 
      in_channels,
      base_channels = 32,
      out_channels = 64,
    ):
    super(ShallowEncoder, self).__init__()
    # Calculate number of downsampling steps needed
    # Input assumed 64x64, we use conv+pool twice to go to 16x16
    self.encoder = nn.Sequential(
      # Block 1: 64x64 -> 32x32
      nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(base_channels),
      nn.ReLU(inplace=True),
      # Block 2: 32x32 -> 16x16
      nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(base_channels*2),
      nn.ReLU(inplace=True), ## (B, out, 8, 8)
    )

  def forward(self, x):
    return self.encoder(x) 
import torch
from torch import nn

from modules.cnns.cnn_encoder import ConvEncoder
from modules.cnns.cross_modal_attn import CrossModalAttention

## as we are fusing, this must always include a depth channel
class CrossAttentionFeatures(nn.Module):
  # default_opts = {}
  def __init__(self,
    rgb_channels: int, ## this is cams * 3 already
    embed_size: int,
    feat_size: int,
    attn_num_heads: int = 8,
    is_deep_fuse: bool = True ,
    # opts: dict = {}          
  ):
    super(CrossAttentionFeatures, self).__init__()

    # self.opts = self.default_opts | opts
    self.embed_size = embed_size
    self.feat_size = feat_size
    self.is_deep_fuse = is_deep_fuse

    self.rgb_enc = ConvEncoder(in_channels=rgb_channels, out_channels = embed_size)
    self.depth_enc = ConvEncoder(in_channels = 1, out_channels = embed_size)

    self.attn_dtor = CrossModalAttention(self.embed_size, num_heads = attn_num_heads)
    self.attn_rtod = CrossModalAttention(self.embed_size, num_heads = attn_num_heads)

    ## TODO: maybe allow more layers here later??
    if self.is_deep_fuse:
      middle_dim = (self.embed_size + self.feat_size) // 2
      self.fuser = nn.Sequential(
        ## initial fusing layer from non deep_fuse
        # nn.Conv2d(self.embed_size * 2, self.embed_size * 2, kernel_size=1, bias=False),
        # nn.BatchNorm2d(self.embed_size * 2),
        # nn.ReLU(),
        ## (B, 2 * embed, 8, 8)
        nn.Conv2d(self.embed_size * 2, self.embed_size, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(self.embed_size),
        nn.ReLU(),
        ## (B, embed, 4, 4)
        nn.Conv2d(self.embed_size, middle_dim, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(middle_dim),
        nn.ReLU(),
        ## (B, feat, 2, 2)
        nn.Conv2d(middle_dim, self.feat_size, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(self.feat_size),
        nn.ReLU(),
        ## (B, feat, 1, 1)
      )
    else:
      self.fuser = nn.Sequential(
        nn.Conv2d(self.embed_size * 2, self.embed_size * 2, kernel_size=1, bias=False),
        nn.BatchNorm2d(self.embed_size * 2),
        nn.ReLU(),
        nn.Conv2d(self.embed_size * 2, self.feat_size, kernel_size=3, bias=False),
        nn.BatchNorm2d(self.feat_size),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)) ## (B, C, 1, 1)
      )

  ## NOTE: ensure the image nad depth split beforehand
  def forward(self, rgb, depth) -> tuple[torch.Tensor, dict]:
    ## rgb: (B, ch, w, h)
    ## depth: (B, ch, w, h)
    assert len(rgb.shape) == len(depth.shape) == 4, f"[cross_attn_feats - forward] wrong format of inputs 'rgb.shape: {len(rgb.shape)}' and 'depth.shape: {len(depth.shape)} should be 4"

    rgb_feats: torch.Tensor = self.rgb_enc(rgb) 
    depth_feats: torch.Tensor = self.depth_enc(depth) 
    
    rgb_attn, rgb_ret = self.attn_dtor(depth_feats, rgb_feats)
    depth_attn, depth_ret = self.attn_rtod(rgb_feats, depth_feats)

    ##residal fuse ## TODO add a learnable parameter here?
    #ie depth_fused = α * depth_feats + (1−α) * depth_attn
    rgb_fused = rgb_feats + rgb_attn
    depth_fused = depth_feats + depth_attn
    
    combined = torch.cat([rgb_fused, depth_fused], dim = 1)

    fused_feats = self.fuser(combined) ## also does pooling
    return fused_feats, {
      "rgb_attn_weights": rgb_ret["attn_weights"],
      "depth_attn_weights": depth_ret["attn_weights"],
      "rgb_attn_out": rgb_attn,
      "depth_attn_out": depth_attn,
      "rgb_fused": rgb_fused,
      "depth_fused": depth_fused,
      "rgb_unfused": rgb_feats,
      "depth_unfused": depth_feats,
      "attn_embed_size": self.embed_size,
      "attn_feat_size": self.feat_size,
      "attn_is_deep_fuse": self.is_deep_fuse
    }
import torch
import torch.nn as nn

from lib.cam_type import CamType

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, img_size):
      super().__init__()
      self.patch_size = patch_size
      self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
      self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
      # x: (B, C, H, W)
      x = self.proj(x)           # (B, E, H/P, W/P)
      x = x.flatten(2)           # (B, E, N)
      x = x.transpose(1, 2)      # (B, N, E)
      return x

class MultiViewEncoder(nn.Module):
    def __init__(self, 
      cam_type: CamType,
      embed_dim: int = 128, 
      num_heads: int = 8, 
      num_layers: int = 4, 
      patch_size: int = 8,
      img_size: int = 64,
      max_eplen: int = 100,
    ):
      super(MultiViewEncoder, self).__init__()
      self.cam_type = cam_type
      self.trans_num_layers = num_layers
      self.max_eplen = max_eplen
      self.embed_dim = embed_dim

      self.num_views = len([ct for ct in CamType.main4() if ct & self.cam_type])

      if self.num_views < 2:
        raise RuntimeError(f"[vit_encoder (MultiViewEncoder)] Need at least 2 cams here given: {self.cam_type}")
      
      patch_embeds_dict = {
         f"{ct}": PatchEmbed(3, embed_dim, patch_size, img_size)
        for ct in CamType.main3() if ct & self.cam_type
      }

      if self.cam_type & CamType.WRIST_DEPTH:
        patch_embeds_dict[f"{CamType.WRIST_DEPTH}"] = PatchEmbed(
           1, embed_dim, patch_size, img_size
        )
      self.patch_embeds = nn.ModuleDict(patch_embeds_dict)

      self.num_patches = next(iter(self.patch_embeds.values())).num_patches
      self.total_tokens = self.num_views * self.num_patches

      self.pos_embed = nn.Parameter(torch.zeros((1, self.total_tokens, embed_dim)))

      encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads)
      self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.trans_num_layers)

      self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, 
      image: torch.Tensor, 
    ) -> tuple[torch.Tensor, dict]:
      ## image: (B, ch, H, W)
      B, _, h, w = image.shape

      patches = []
      ch_count = 0

      for ct in CamType.main3():
        if self.cam_type & ct:
          rgb = image[:, ch_count:ch_count+3, :, :]
          patch = self.patch_embeds[f"{ct}"](rgb) ## (B, n, E)
          patches.append(patch)
      
      if self.cam_type & CamType.WRIST_DEPTH:
        d = image[:, ch_count:ch_count+1, :, :] ## gives (B, 1, h, w)
        patch = self.patch_embeds[f"{CamType.WRIST_DEPTH}"](d)
        patches.append(patch)
      
      x = torch.cat(patches, dim = 1) ## ## (B, tokens, E)
      ## flatted spatial
      x = x + self.pos_embed

      x = x.transpose(0, 1) ## transformer wants (n_tokens, B, E), 
      vit_out = self.encoder(x)
      vit_out = vit_out.transpose(0, 1) ## back to (B, n_tokens, E)

      pool_x = vit_out.transpose(1, 2) ## (B, E, n_tokens)
      pooled_out = self.pool(pool_x) # (B, E)

      return pooled_out, {"full_vit_repr": vit_out}
    
class MultiViewTemporalEncoder(nn.Module):
    def __init__(self, 
      cam_type: CamType,
      embed_dim: int = 128, 
      num_heads: int = 8, 
      num_layers: int = 4, 
      patch_size: int = 8,
      img_size: int = 64,
      max_eplen: int = 100,
    ):
      super(MultiViewTemporalEncoder, self).__init__()
      self.cam_type = cam_type
      self.trans_num_layers = num_layers
      self.max_eplen = max_eplen

      self.num_views = len([ct for ct in CamType.main4() if ct & self.cam_type])

      if self.num_views < 2:
        raise RuntimeError(f"[vit_encoder (MultiViewEncoder)] Need at least 2 cams here given: {self.cam_type}")
      
      patch_embeds_dict = {
         f"{ct}": PatchEmbed(3, embed_dim, patch_size, img_size)
        for ct in CamType.main3() if ct & self.cam_type
      }

      if self.cam_type & CamType.WRIST_DEPTH:
        patch_embeds_dict[f"{CamType.WRIST_DEPTH}"] = PatchEmbed(
           1, embed_dim, patch_size, img_size
        )
      self.patch_embeds = nn.ModuleDict(patch_embeds_dict)

      self.num_patches = next(iter(self.patch_embeds.values())).num_patches
      self.view_patches = self.num_views * self.num_patches

      self.pos_embed_space = nn.Parameter(torch.zeros(1, self.view_patches, embed_dim))
      self.pos_embed_time = nn.Parameter(torch.zeros(1, self.max_eplen, embed_dim))

      encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads)
      self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.trans_num_layers )
    
    def forward(self, 
      image: torch.Tensor, 
      lengths: None | torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
      ## image: (B, t, ch, H, W)
      B, t, _, h, w = image.shape

      patches = []
      ch_count = 0

      for ct in CamType.main3():
        if self.cam_type & ct:
          rgb = image[:, :, ch_count:ch_count+3, :, :]
          rgb = rgb.view(B*t, 3, h, w)
          patch = self.patch_embeds[f"{ct}"](rgb) ## (B*t, n, E)
          reshaped_patch = patch.view(B, t, self.num_patches, -1)
          patches.append(reshaped_patch)
      
      if self.cam_type & CamType.WRIST_DEPTH:
        d = image[:, :, ch_count:ch_count+1, :, :].unsqueeze(dim=2) ## gives (B, t, 1, h, w)
        d = d.view(B*t, 1, h, w)
        patch = self.patch_embeds[f"{CamType.WRIST_DEPTH}"](d)
        reshaped_patch = patch.view(B, t, self.num_patches, -1)
        patches.append(reshaped_patch)
      
      if lengths is None:
         raise NotImplementedError("No inference yet, only training")

      x = torch.stack(patches, dim = 2) ## stack after hte sequence dim
      ## flatted spatial
      x = x.view(B, t, self.view_patches, -1) ## (B, t, vp, E)

      ## spatial and the temporal pos embeddings
      p_sp = self.pos_embed_space.unsqueeze(1) ## (1, 1, vp, E) -> (1, 1, 1, vp, E)
      p_t = self.pos_embed_space.unsqueeze(2) ## (1, t, 1, E) -> (1, t, 1, 1, E)

      x = x + p_sp + p_t
      x = x.view(B, t*self.view_patches, -1) ## (B, t*vp, E)

      S = t * self.view_patches

      mask = torch.arange(S, device=lengths.device).unsqueeze(0)
      valid_mask = (mask < (lengths.unsqueeze(1) * self.view_patches))
      key_padding_mask = ~valid_mask ## not valit bitmask


      x = x.transpose(0, 1) ## transformer wants (S, B, E)
      vit_out = self.encoder(x, src_key_padding_mask=key_padding_mask)
      vit_out = vit_out.transpose(0, 1) ## back to (B, S, E)

      valid_token_mask = valid_mask.unsqueeze(-1).float() ## (B, S, 1)
      sum_feats = (vit_out * valid_token_mask).sum(dim = 1) ## (B, E)
      denom = valid_mask.sum(dim = 1).clamp(min = 1) # (B, 1)
      pooled_out = sum_feats / denom ## (B, E)


      return pooled_out, {"full vit repr": vit_out}
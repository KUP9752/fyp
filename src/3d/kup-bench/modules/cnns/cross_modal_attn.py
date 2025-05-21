import torch 
import torch.nn as nn


class CrossModalAttention(nn.Module):
    def __init__(self,
      embed_dim,
      num_heads, 
      batch_first: bool = False    ## dont use
    ):
      super(CrossModalAttention, self).__init__()
      assert embed_dim % num_heads == 0, f"[cross_modal_attn - (CrossModalAttention)]'embed_dim' must be divisible by 'num_heads'"
      print(f"{num_heads = }")
      self.attn = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=batch_first
      )

    def forward(self, query_feats: torch.Tensor, key_feats: torch.Tensor):
      # query_feats, key_feats: (B, C, H, W)
      B, ch, h, w = query_feats.shape
      # Flatten spatial dims: (H*W, B, C)
      q = query_feats.flatten(2).permute(2, 0, 1)
      k = key_feats.flatten(2).permute(2, 0, 1)
      v = k
      print(f"[CMA] {q.shape = }")
      print(f"[CMA] {k.shape = }")
      print(f"[CMA] {v.shape = }")

      # Attention
      out, weights = self.attn(
        query=q,
        key=k,
        value=v,
        need_weights = True,
        average_attn_weights = False
      )


      print(f"[CMA] {weights.shape = }")
      

      # Reshape back: (B, C, H, W)
      out = out.permute(1, 2, 0).view(B, ch, h, w)
      print(f"[CMA] {out.shape = }")
      return out, {"attn_weights": weights}
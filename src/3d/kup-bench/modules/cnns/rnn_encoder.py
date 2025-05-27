from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from lib.cam_type import CamType
from modules.cnns.cnn_encoder import CNNEncoder
from modules.cnns.cross_attn_feats import CrossAttentionFeatures
from modules.joint_pos_encoder import JointPosEncoder
class RNNEncoder(nn.Module):
  def __init__(self,
    cam_type: CamType,
    ## lstm options, pass as a dict when creating
    encoding_size = 512, ## default ouutput of CNNEncoder
    hidden_size = 256,
    num_layers = 2,
    batch_first = True, ## keep this true the dataloader handles it as batch first so, (B, t, ...)
    is_bidir = False, ## dont need it to be bidirectional ever i dont think
    config: Literal["depth_feats", "attn"] = "depth_feats",
    attn_opts: dict = {},
    use_proprio: bool = False,
    proprio_opts: dict = {}
    # cnn_encoder_opts: dict = {} ## introduce if need to pass more settings
  ):
    super(RNNEncoder, self).__init__()
    self.config = config
    self.use_proprio = use_proprio
    
    self.cam_type = cam_type
    num_ch = 0
    if cam_type & CamType.WRIST:
      num_ch += 3
    if cam_type & CamType.LEFT_SHOULDER:
      num_ch += 3
    if cam_type & CamType.RIGHT_SHOULDER:
      num_ch += 3

    self.rgb_conv = CNNEncoder(in_channels=num_ch)
    self.flat_size = 128 * 2 * 2

    if self.cam_type & CamType.WRIST_DEPTH:
      self.depth_conv= CNNEncoder(in_channels=1)
      self.flat_size *= 2 ## doubule double the flat size

    match self.config:
      case "depth_feats":
        encoding_size = self.flat_size
        assert encoding_size == self.flat_size, f"[rnn_encoder - (RNNEncoder)] encoding_size ({encoding_size}) != flat_size ({self.flat_size})"
      case "attn":
        if not (self.cam_type & CamType.WRIST_DEPTH):
          raise ValueError(f"[rnn_encoder (RNNEncoder)] Policy cam_type does not include '{CamType.WRIST_DEPTH}' -> current: '{self.cam_type}'")
        
        ## making this smaller than before
        encoding_size = encoding_size // 2  ## 256

        self.attn_feats = CrossAttentionFeatures(
          rgb_channels=num_ch,
          embed_size = 512,
          feat_size = encoding_size,
          **attn_opts
          # attn_num_heads = 8 ## default
          # is_deep_fuse= True ## also default
        )
        print(f"[rnn-encoder] attn opts: {num_ch = }, embed_size = {self.flat_size = }, {encoding_size = }, {attn_opts = }")
        
      case _:
        raise ValueError(f"[rnn_encoder (RNNEncoder)] wrong value for 'config' ({self.config})")
    
    if self.use_proprio:
      self.jpos_feats = JointPosEncoder(**proprio_opts) if proprio_opts else JointPosEncoder()
      encoding_size += self.jpos_feats.output_size

    self.rnn = nn.LSTM(
      input_size = encoding_size, 
      hidden_size = hidden_size,
      num_layers=num_layers, 
      batch_first=batch_first,
      bidirectional = is_bidir
    )
    self.encoding_size = hidden_size


  ## processes the convs and gives out a flattened version of the vector
  def _get_image_feats(self, image: torch.Tensor) -> tuple[torch.Tensor, dict]: ## (B, t, out_size)
    b, t, ch, w, h = image.shape

    if self.cam_type & CamType.WRIST_DEPTH:
      rgb = image[:, :, :-1, :, :] ## take all rgb cams
      depth = image[:, :, -1, :, :].unsqueeze(dim=1) # for (B, w, h) -> (B, 1, w, h)

      rgb = rgb.view(b * t, ch - 1, w, h)
      depth = depth.view(b * t, 1, w, h)

      match self.config:
        case "depth_feats":
          im_feats = self.rgb_conv(rgb)  ##(B * T, 128, 2, 2)  
          depth_feats = self.depth_conv(depth)## (B * T, 128, 2, 2)

          ## -1 at the end flattens the vectors
          im_feats = im_feats.view(b, t, -1) ## (B, T, 512)
          depth_feats = depth_feats.view(b, t, -1) ## (B, T, 512)
          ## concat on the feature dimension -> (B, T, im + d = 1024)
          feats = torch.cat([im_feats, depth_feats], dim = -1) 

          return feats, {}
        
        case "attn":
          feats, ret_dict = self.attn_feats(rgb, depth)
          feats = feats.view(b, t, -1) ## flatten
          
          return feats, ret_dict
          
        case _:
          raise ValueError(f"[rnn_encoder (get_feats)] wrong value for 'config' ({self.config})")
    else: 
      ## there is no 'attn' branch when there is no depth involved
      image = image.view(b * t, ch, w, h)
      im_feats = self.rgb_conv(image)  ##(B * T, 512) ## flatteened by -1

      im_feats = im_feats.view(b, t, -1) ## (B, T, 512)
      return im_feats, {}
    
  ## Not needed to separate with `forward()` but feels nicer to know which branch we are taking 3 classes removed from the task
  def inference_forward(self, 
    image: torch.Tensor, 
    hidden_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    proprio: Optional[torch.Tensor]= None,
  ):
    if proprio is None and self.use_proprio:
      raise RuntimeError(f"[rnn_encoder - forward] Proprioceptive training selected but data not given!")
    
    image =  image.unsqueeze(0) ## (1, 1, ..) add seq_len = 1
    feats, ret_dict = self._get_image_feats(image) ## (1, ..) size because now its inference

    feats =  feats.view(1, 1, -1) ## (1, 1, ..) add seq_len = 1

    if self.use_proprio:
      assert proprio is not None, f"[rnn_encoder - inference_forward] must have proprio at this point"
      proprio = proprio.unsqueeze(0) ## as abovee add seq_len (1, 1, ..)
      jpos_feats, jpos_dict = self.jpos_feats(proprio)
      
      feats = torch.cat([feats, jpos_feats], dim = -1)
      ret_dict |= jpos_dict

    rnn_out, (h, c) = self.rnn(feats, hidden_state)

    enc = h[-1]

    return enc, ret_dict | {
      "rnn_ret": rnn_out,
      "h": h, 
      "c": c
    }

  ## output size is the hidden size, this can later be used to do whatever
  ## THIS IS FOR TRAINING WITH ENTIRE KNOWN LENGTHS
  def forward(self,
    image,
    lengths,
    hidden_state: Optional[tuple] = None, ## of tensors (h, c)
    proprio: Optional[torch.Tensor]= None,
  ): ## image here can contain channels from differnt camears including depth
    if proprio is None and self.use_proprio:
      raise RuntimeError(f"[rnn_encoder - forward] Proprioceptive training selected but data not given!")
    ## we want these of shape: (B, t, c, w, h)
    ## t is the time series that is going to be fed into the lstm, so we want some ordering now from dataset
    
    feats, ret_dict = self._get_image_feats(image)    

    if self.use_proprio:
      assert proprio is not None, f"[rnn_encoder - forward] must have proprio at this point"
      jpos_feats, _ = self.jpos_feats(proprio)
      
      feats = torch.cat([feats, jpos_feats], dim = -1)

    packed_in = pack_padded_sequence(
      feats, lengths.cpu(), batch_first=True, enforce_sorted=False
    )

    packed_out, (h_n, c_n) = self.rnn(packed_in, hidden_state)

    

    rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)

    return rnn_out, ret_dict | {
      "rnn_ret": rnn_out, ## might be useful to have down the line 
      "h_n": h_n,
      "h_c": c_n
    } ## returns the final last time step of rnn





    



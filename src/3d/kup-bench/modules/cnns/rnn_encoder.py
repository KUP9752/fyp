from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from lib.cam_type import CamType
from modules.cnns.cnn_encoder import CNNEncoder


class RNNEncoder(nn.Module):
  def __init__(self,
    cam_type: CamType,
    merge_feats: bool = True, ## merge means explicit merging before lstm, otherwise shove all into lstm
    ## lstm options, pass as a dict when creating
    encoding_size = 512, ## default ouutput of CNNEncoder
    hidden_size = 256,
    num_layers = 2,
    batch_first = True,
    is_bidir = False
    # cnn_encoder_opts: dict = {} ## introduce if need to pass more settings
  ):
    super(RNNEncoder, self).__init__()

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

      ## explicit merging of features before the LSTM
    ## TODO: encoding size can be expanded to be something other than the flat size for the merge_feats branch?
    if merge_feats:
      raise NotImplementedError(f"[lstm_encoder (LSTMEncoder)] will do at some point")
    else: 
      encoding_size = self.flat_size
    ## maybe this is unnecessary
    # self.flatten = nn.Sequential(
    #   nn.Flatten(),
    #   nn.Linear(self.flat_size, encoding_size),
    #   nn.ReLU(inplace=False)
    # )

    assert encoding_size == self.flat_size, f"[rnn_encoder - (RNNEncoder)] encoding_size ({encoding_size}) != flat_size ({self.flat_size})"

    self.rnn = nn.LSTM(
      input_size = encoding_size, 
      hidden_size = hidden_size,
      num_layers=num_layers, 
      batch_first=batch_first,
      bidirectional = is_bidir
    )
    self.encoding_size = hidden_size
    ## output size is the hidden size, this can later be used to do whatever
  ## GPT suggesteed hidden_state => tuple? of (h0, c0) : (num_layers, B, hidden)
  ## NOTE: this might help with next state, or next view prediction??
  def forward(self, image, lengths, hidden_state=None): ## image here can contain channels from differnt camears
    ## we want these of shape: (B, t, c, w, h)
    ## t is the time series that is going to be fed into the lstm, so we want some ordering now from dataset
    b, t, ch, w, h = image.shape
    print(f"{image.shape = } (on forward() call")
    

    if self.cam_type & CamType.WRIST_DEPTH:
      rgb = image[:, :, :-1, :, :] ## take all rgb cams
      depth = image[:, :, -1, :, :].unsqueeze(dim=1) # for (B, w, h) -> (B, 1, w, h)

      ## TODO: view manipulation here?? conv wants (B, C, W, H)
      rgb = rgb.view(b * t, ch - 1, w, h)
      depth = depth.view(b * t, 1, w, h)
      print(f"{rgb.shape =} (reshaped)")
      print(f"{depth.shape =}")
      
      im_feats = self.rgb_conv(rgb)  ##(B * T, 128, 2, 2)  
      print(f"{im_feats.shape =}")
      depth_feats = self.depth_conv(depth)## (B * T, 128, 2, 2)
      print(f"{depth_feats.shape =}")

      ## -1 at the end flattens the vectors
      im_feats = im_feats.view(b, t, -1) ## (B, T, 512)
      depth_feats = depth_feats.view(b, t, -1) ## (B, T, 512)
      print(f"{im_feats.shape = } (reshaped)")
      print(f"{depth_feats.shape = } (reshaped)")
      ## concat on the feature dimension -> (B, T, im + d = 1024)
      feats = torch.cat([im_feats, depth_feats], dim = -1) 
    else: 
      image = image.view(b * t, ch, w, h)
      print(f"{image.shape = } (reshaped)")
      
      im_feats = self.rgb_conv(image)  ##(B * T, 512) ## flatteened by -1
      print(f"{im_feats.shape = }")

      im_feats = im_feats.view(b, t, -1) ## (B, T, 512)
      print(f"{im_feats.shape = }")

      feats = im_feats
      print(f"{feats.shape = }")


    packed_in = pack_padded_sequence(
      feats, lengths, batch_first=True, enforce_sorted=False
    )
    
    rnn_ret, (h_n, c_n) = self.rnn(packed_in, hidden_state)
    final_enc = h_n[-1]




    return final_enc, {
      "rnn_ret": rnn_ret, ## NOTE: this will need pad_packed_sequence to unpack, if per frame information is needed
      "h_n": h_n,
      "h_c": c_n
    } ## returns the final last time step of rnn





    



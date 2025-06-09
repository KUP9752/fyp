from enum import Enum, auto

class FuseConfig(Enum):
  ## wdlr represents: wrist_rgb, wrist_depth, left_rgb, right_rgb
  WDLR = auto()

  ## wlr + d
  WLR_D = auto() 
  DEPTH_FEATS_GATED = auto() #wlr + d
  DEPTH_FEATS_ATTN = auto() #wlr + d

  ##wd + lr
  WD_LR = auto() 
  WD_LR_ATTN = auto() 

  #w + d
  ## dont want other cams here, the lr cams will just be ignored in this case

  Wfilm_D = auto() ## depth modulated colour
  W_Dfilm = auto() ##  colour modulated depth
  Wfilm_Dfilm = auto() ## depth modulated colour

  #w + d + l + r
  W_D_L_R = auto() ## all separated but cated before linear layer
  W_D_L_R_FILM = auto() ## 
  W_D_L_R_ATTN = auto() #' 4 way cross attention?
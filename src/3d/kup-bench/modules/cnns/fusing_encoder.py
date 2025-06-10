from typing import Literal, Optional

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
transforms.Normalize
from rlbench.demo import Demo

from tqdm import tqdm as progress
from lib.cam_type import CamType

from lib.utils import params_string

from modules.cnns.cnn_encoder import CNNEncoder
from modules.cnns.multi_cam_cnn import MultiCamCnn
from modules.cnns.vit_encoder import MultiViewEncoder

from modules.film_net import FilmModulator, DeepEncoder
from modules.cnns.cross_attn_feats import CrossAttentionFeatures

from modules.joint_pos_encoder import JointPosEncoder

from modules.dataset.demo_obs_dataset import DemoObsDataset
from modules.dataset.demo_dataset import DemoDataset
from lib.fuse_config import FuseConfig
  
## Making a separate class/file here for this differnet than `SimpleGrasp` just so it is more convenient to tweak and experiment with
class FusingEncoder(nn.Module): 
  default_opts = {
    "feat_size_per": 128,
    "cnn":{
      ##NOTE: if not 4 layers the final size will be different
      "cnn_rgb_layers": [32, 48, 64, 128], 
      "cnn_depth_layers": [32, 48, 64, 128],
    },
    "attn": {
      "embed_size": 128,
      "attn_num_heads": 8,
      "attn_deep_fuse": True, ## works better
    },
    "film":{
      "downer_cnn_layers": [32, 48, 64, 128],
      "double_downer_cnn_layers": [36, 72, 128, 128],
    },
    "mvt":{
      "embed_dim":  128, 
      "num_heads":  8, 
      "num_layers": 4, 
      "patch_size": 8,
      "img_size":  64,
      "max_eplen":  100,
    },
  }

  def __init__(self,
    cam_type: CamType,
    config: FuseConfig,
    opts: dict = {},
  ):
    ## keeps all defaults that are not overriden in opts
    self.opts = self.default_opts | opts
    super(FusingEncoder, self).__init__()
    self.cam_type = cam_type
    self.config = config

    self.feat_size = self.opts["feat_size_per"] * sum([1 for ct in CamType.main4() if ct & self.cam_type])

    all_rgb_chs = 0
    for ct in CamType.main3():
      if self.cam_type & ct:
        all_rgb_chs += 3 
    
    match self.config:
      case FuseConfig.WDLR:
        num_ch = all_rgb_chs
        if self.cam_type & CamType.WRIST_DEPTH:
          num_ch += 1

        layers = self.opts["cnn"]["cnn_rgb_layers"]
        self.conv = CNNEncoder(in_channels=num_ch, layers = layers)
        self.feat_size = self.conv.flat_out_size
        

      case (
        FuseConfig.WLR_D |
        FuseConfig.DEPTH_FEATS_GATED 
      ):
        self._check_cam(needed=[CamType.WRIST_DEPTH])

        layers = self.opts["cnn"]["cnn_rgb_layers"]
        d_layers = self.opts["cnn"]["cnn_depth_layers"]

        self.rgb_enc = CNNEncoder(in_channels=all_rgb_chs, layers = layers)
        self.depth_enc = CNNEncoder(in_channels=1, layers = d_layers)

        ## this is the flat size
        self.feat_size = self.rgb_enc.flat_out_size + self.depth_enc.flat_out_size
        conv_size = self.rgb_enc.flat_out_shape[0] + self.depth_enc.flat_out_shape[0]

        ## simply concat an
        if self.config == FuseConfig.WLR_D:
          self.depth_fuse = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(self.feat_size, self.feat_size),
            nn.BatchNorm1d(self.feat_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(self.feat_size, self.feat_size // 2)
          )
          self.feat_size = self.feat_size // 2

        if self.config == FuseConfig.DEPTH_FEATS_GATED:
          self.gate = nn.Sequential(
            nn.Conv2d(conv_size, conv_size // 2, kernel_size=1),
            nn.Conv2d(conv_size // 2, conv_size // 2, kernel_size=1),
            nn.Sigmoid()
          )
          self.feat_size = self.feat_size // 2 ## works because the conv_size // 2 respects the flat size (2, 2) dimensions

      case FuseConfig.DEPTH_FEATS_ATTN:
        self._check_cam(needed=[CamType.WRIST_DEPTH], any_one=CamType.main3())

        self.feat_size = self.opts["feat_size_per"] * 2 ## 128 for at least 1 rbg, 128 for depth

        self.attn_feats = CrossAttentionFeatures(
          rgb_channels=all_rgb_chs,
          depth_channels= 1,
          embed_size = self.opts["attn"]["embed_size"], 
          feat_size = self.feat_size,
          attn_num_heads = self.opts["attn"]["attn_num_heads"],
          is_deep_fuse = self.opts["attn"]["attn_deep_fuse"],
        )
      case (
        FuseConfig.WD_LR |
        FuseConfig.WD_LR_ATTN
      ):
        self._check_cam(needed=CamType.wrists(), any_one=CamType.shoulders())

        num_sh_ch = 0
        for ct in CamType.shoulders():
          if self.cam_type & ct:
            num_sh_ch += 3

        if self.config == FuseConfig.WD_LR_ATTN:
          self.feat_size = self.opts["feat_size_per"] * 2

          self.attn_feats = CrossAttentionFeatures(
            rgb_channels= 3 + 1, ##wrist rgb + depth
            depth_channels= num_sh_ch,  
            embed_size = self.opts["attn"]["embed_size"], 
            feat_size= self.feat_size,
            attn_num_heads = self.opts["attn"]["attn_num_heads"],
            is_deep_fuse = self.opts["attn"]["attn_deep_fuse"],
          )

        else: ## self.config == WD_LR
          layers = self.opts["cnn"]["cnn_rgb_layers"]
          self.wd_enc = CNNEncoder(4, layers)
          self.lr_enc = CNNEncoder(num_sh_ch, layers)
          self.feat_size = self.wd_enc.flat_out_size + self.lr_enc.flat_out_size

      case FuseConfig.Wfilm_D:
        ## depth modulated colour
        self._check_cam(needed=CamType.wrists())
        self._fail_if_cam(CamType.shoulders())

        ## NOTE: will ignore the other cameras given to it!!

        self.film = FilmModulator(
          in1 = 3, ## wrist rgb, 
          in2 = 1, ## wrist d, 
          feat_dim = 6,
          do_both = False,
        )
        self.mod_downer = CNNEncoder(6, layers = self.opts["film"]["downer_cnn_layers"]) ## use default layers
        ##NOTE: modulation happens at the resolution level need to downsample

        self.feat_size = self.mod_downer.flat_out_size

      case FuseConfig.W_Dfilm :
        ## colour modulated depth
        self._check_cam(needed=CamType.wrists())
        self._fail_if_cam(CamType.shoulders())

        self.film = FilmModulator(
          in1 = 1, ## wrist d, 
          in2 = 3, ## wrist rgb, 
          feat_dim = 6,
          do_both = False,
        )
        self.mod_downer = CNNEncoder(6, layers = self.opts["film"]["downer_cnn_layers"]) ## use default layers
        self.feat_size = self.mod_downer.flat_out_size

      case FuseConfig.Wfilm_Dfilm:
        ## both ways then concatenate
        self._check_cam(needed=CamType.wrists())
        self._fail_if_cam(CamType.shoulders())

        self.film = FilmModulator(
          in1 = 3, ## wrist rgb, 
          in2 = 1, ## wrist d, 
          feat_dim = 6,
          do_both = True,
        )
        self.mod_downer = CNNEncoder(12, layers = self.opts["film"]["double_downer_cnn_layers"]) ## double the size of last time, 2 of them
        self.feat_size = self.mod_downer.flat_out_size
      case (
        FuseConfig.Wfilm_D_LATE |
        FuseConfig.W_Dfilm_LATE |
        FuseConfig.Wfilm_Dfilm_LATE 
      ):
        self._check_cam(needed=CamType.wrists())
        self._fail_if_cam(CamType.shoulders())

        self.enc_w = DeepEncoder(3)## 4 -> 8 -> 16, 8, 8
        self.enc_d = DeepEncoder(1)## 4 -> 8 -> 16, 8, 8

        if self.config == FuseConfig.Wfilm_Dfilm_LATE:
          self.film = FilmModulator(
            in1 = 16, 
            in2 = 16, 
            feat_dim = 16,
            do_both = True
          )
          self.feat_size = self.enc_w.flat_out_size + self.enc_d.flat_out_size ## only one is needed
        else:
          self.film = FilmModulator(
            in1 = 16, 
            in2 = 16, 
            feat_dim = 16,
            do_both = False
          )
          self.feat_size = self.enc_w.flat_out_size ## only one is needed
        # if self.config == FuseConfig.Wfilm_D_LATE:
        # if self.config == FuseConfig.W_Dfilm_LATE:

          

      case FuseConfig.W_D_L_R:
        self.multi_enc = MultiCamCnn(self.cam_type)
        no_cams =  len([ct for ct in CamType.main4() if ct & self.cam_type])
        ## provide at least 3 cams here, otherwise no point
        if no_cams < 3: 
          raise RuntimeError(f"[fusing_encoder - (FusingEncoder)] We want at least 3 cams here")
        self.feat_size = no_cams * 128 * 2 * 2

      case FuseConfig.W_D_L_R_FILM:
        # force all 4, cant be asked to figure it out for 3 no real point for 3
        self._check_cam(needed=CamType.main4())
        ## the order of fusing is w+d then l+r then wd+lr

        self.wd_film = FilmModulator(
          in1 = 3, 
          in2 = 1, 
          feat_dim = 6,
          do_both=False
        )
        self.wd_downer = CNNEncoder(6, layers = self.opts["film"]["downer_cnn_layers"])

        self.lr_film = FilmModulator(
          in1 = 3, 
          in2 = 3, 
          feat_dim = 3,
          do_both=True
        )

        self.lr_downer = CNNEncoder(6, layers = self.opts["film"]["double_downer_cnn_layers"])

        self.feat_size = self.lr_downer.flat_out_size + self.wd_downer.flat_out_size

      case FuseConfig.W_D_L_R_ATTN:
        self.mvt = MultiViewEncoder(
          cam_type=self.cam_type,
          **self.opts["mvt"]
        )
        self.feat_size = self.mvt.embed_dim
        ## NOTE: no cam checks here the attention thing already takes care of that

      case _:
        raise ValueError(f"[fusing_encoder - (FusingEncoder)] unknown value for FuseConfig: '{self.config}")

    self.final_feat_size = self.feat_size
  
  def _fail_if_cam(self, fail_cases: list[CamType]):
    if any([self.cam_type & ct for ct in fail_cases]):
      raise ValueError(f"[fusing_encoder - (fail_if_cam)] Given a can that is not compatible with this configuration! no '{fail_cases}")

  def _check_cam(self, needed: list[CamType] | None = None, any_one: list[CamType] | None = None):

    if needed is not None and not all([self.cam_type & ct for ct in needed]):
      raise ValueError(f"[fusing_encoder - (check_cam)] Policy cam_type does not include all of '{needed}' -> current: '{self.cam_type}'")

    if any_one is not None and not any([self.cam_type & ct for ct in any_one]):
      raise ValueError(f"[fusing_encoder - (check_cam)] Policy cam_type does not include any of '{any_one}' -> current: '{self.cam_type}'")
    
  def forward(self, image) -> tuple[torch.Tensor, dict]:
    ret_dict = {"config": self.config}
    match self.config:
      case  FuseConfig.WDLR:
        return self.conv(image), ret_dict
      case (
        FuseConfig.WLR_D |
        FuseConfig.DEPTH_FEATS_GATED
      ):  
        self._check_cam(needed=[CamType.WRIST_DEPTH])
        rgbs = image[:, :-1, :, :] ## take all rgb cams
        depth = image[:, -1, :, :].unsqueeze(dim=1) # for (B, w, h) -> (B, 1, w, h)
    
        rgb_feats: torch.Tensor = self.rgb_enc(rgbs) 
        depth_feats: torch.Tensor = self.depth_enc(depth)
        
        cated = torch.cat([rgb_feats, depth_feats], dim = 1)

        if self.config == FuseConfig.WLR_D:
          fused_feats =  self.depth_fuse(cated)
          
        elif self.config == FuseConfig.DEPTH_FEATS_GATED:
          gate = self.gate(cated) 
          fused_feats = gate * rgb_feats + (1 - gate) * depth_feats
        else:
          raise RuntimeError(f"[fusing_encoder - forward] Cannot be here!")   
        
        return fused_feats, ret_dict

      case FuseConfig.DEPTH_FEATS_ATTN:
        self._check_cam(needed = [CamType.WRIST], any_one = CamType.main3())
        rgbs = image[:, :-1, :, :] ## take all rgb cams
        depth = image[:, -1, :, :].unsqueeze(dim=1) # for (B, w, h) -> (B, 1, w, h)

        fused_feats, attn_dict = self.attn_feats(rgbs, depth)
        return fused_feats, ret_dict | attn_dict
      
      case (
        FuseConfig.WD_LR | 
        FuseConfig.WD_LR_ATTN
      ):
        self._check_cam(needed=CamType.wrists(), any_one=CamType.shoulders())
        w = image[:, :3, :, :]
        d = image[:, -1, :, :]

        wdX = torch.cat([image[:, :3, :, :], image[:, -1, :, :].unsqueeze(dim=1)], dim = 1)
        lrX = image[:, 3:-1, :, :]

        self.feat_size = self.opts["feat_size_per"] * 2


        if self.config == FuseConfig.WD_LR_ATTN:
          fused_feats, attn_dict = self.attn_feats(wdX, lrX)
          return fused_feats, ret_dict | attn_dict
        else: ## self.config == WD_LR
          wd_feats = self.wd_enc(wdX)
          lr_feats = self.lr_enc(lrX)
          cated = torch.cat([wd_feats, lr_feats], dim = 1)
          return cated, ret_dict
        
      case (
        FuseConfig.Wfilm_D | 
        FuseConfig.W_Dfilm |
        FuseConfig.Wfilm_Dfilm
      ):
        self._check_cam(needed=CamType.wrists())
        wX = image[:, :3, :, :]
        dX = image[:, -1, :, :].unsqueeze(dim=1) 

        if self.config == FuseConfig.W_Dfilm:
          mod = self.film(dX, wX)
          
        elif self.config == FuseConfig.Wfilm_D:
          mod = self.film(wX, dX)
        else:# self.config == FuseConfig.Wfilm_Dfilm:
          mod1, mod2 = self.film(wX, dX)
          mod = torch.cat([mod1, mod2], dim = 1)

        feats = self.mod_downer(mod)
        return feats, ret_dict

      case FuseConfig.W_D_L_R:
        feats = []
        curr_index = 0

        for ct in CamType.main3():
          if ct & self.cam_type:
            feat = self.multi_enc(image[:, curr_index:curr_index+3, :, :], ct)
            feats.append(feat)
            curr_index +=3

        if self.cam_type & CamType.WRIST_DEPTH:
          feat = self.multi_enc(image[:, -1, :, :].unsqueeze(dim=1), CamType.WRIST_DEPTH)
          feats.append(feat)

        cated = torch.cat(feats, dim = 1)
        return cated, ret_dict
      
      case (
        FuseConfig.Wfilm_D_LATE |
        FuseConfig.W_Dfilm_LATE |
        FuseConfig.Wfilm_Dfilm_LATE 
      ):
        self._check_cam(needed=CamType.wrists())
        wX = image[:, :3, :, :]
        dX = image[:, -1, :, :].unsqueeze(dim=1) 

        f_w = self.enc_w(wX)
        f_d = self.enc_d(dX)

        if self.config == FuseConfig.W_Dfilm_LATE:
          feats = self.film(f_d, f_w)
        elif self.config == FuseConfig.Wfilm_D_LATE:
          feats = self.film(f_w, f_d)
        else:# self.config == FuseConfig.Wfilm_Dfilm:
          mod1, mod2 = self.film(f_w, f_d)
          feats = torch.cat([mod1, mod2], dim = 1)

        return feats, ret_dict

      case FuseConfig.W_D_L_R:
        feats = []
        curr_index = 0

        for ct in CamType.main3():
          if ct & self.cam_type:
            feat = self.multi_enc(image[:, curr_index:curr_index+3, :, :], ct)
            feats.append(feat)
            curr_index +=3

        if self.cam_type & CamType.WRIST_DEPTH:
          feat = self.multi_enc(image[:, -1, :, :].unsqueeze(dim=1), CamType.WRIST_DEPTH)
          feats.append(feat)

        cated = torch.cat(feats, dim = 1)
        return cated, ret_dict



      case FuseConfig.W_D_L_R_FILM:
        self._check_cam(needed=CamType.main4())
        ## all must be here
        wX = image[:, :3, :, :]
        lsX = image[:, 3:6, :, :]
        rsX = image[:, 6:9, :, :]
        dX = image[:, -1, :, :].unsqueeze(dim=1) 

        wd_mod = self.wd_film(wX, dX)
        wd_feats = self.wd_downer(wd_mod)

        lr_mod, rl_mod = self.lr_film(lsX, rsX)
        s_mod_cat = torch.cat([lr_mod, rl_mod], dim = 1)
        lr_feats = self.lr_downer(s_mod_cat)

        cated = torch.cat([wd_feats, lr_feats], dim = 1)
        return cated, ret_dict
      case FuseConfig.W_D_L_R_ATTN:
        
        feats, mvt_dict = self.mvt(image)
        return feats, ret_dict | mvt_dict

      case _:
        raise ValueError(f"[fusing_encoder - (FusingEncoder)] unknown value for FuseConfig: '{self.config}")

    
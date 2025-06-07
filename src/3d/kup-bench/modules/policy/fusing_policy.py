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

from modules.film_net import FilmModulator
from modules.cnns.cross_attn_feats import CrossAttentionFeatures

from modules.joint_pos_encoder import JointPosEncoder

from modules.dataset.demo_obs_dataset import DemoObsDataset
from modules.dataset.demo_dataset import DemoDataset
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
  ##TODO: the next 2
  W_D_L_R_FILM = auto() ## all separated but cated before linear layer
  W_D_L_R_ATTN = auto() #' 4 way cross attention?

  
## Making a separate class/file here for this differnet than `SimpleGrasp` just so it is more convenient to tweak and experiment with
class FusingPolicy(nn.Module): 
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
      "downer_cnn_layers": [64, 64, 128, 128],
      "double_downer_cnn_layers": [128, 128, 128, 128],
    },
    "mvt":{
      "embed_dim":  128, 
      "num_heads":  8, 
      "num_layers": 4, 
      "patch_size": 8,
      "img_size":  64,
      "max_eplen":  100,
    },
    "proprio_opts": {} ## dict of kwargs
  }

  def __str__(self):
    return f"fusing_policy-is_grasp:{self.is_grasp}-use_proprio@{self.use_proprio}config:{self.config}-opts:{self.opts}"
  
  def __repr__(self):
    return f"FusingPolicy(is_grasp={self.is_grasp}, use_proprio={self.use_proprio}, config={self.config}, opts={self.opts})"
  
  def __init__(self,
    action_shape: int, 
    cam_type: CamType,
    config: FuseConfig,
    is_grasp: bool,
    use_proprio: bool = False, 
    opts: dict = {},

    
  ):
    ## keeps all defaults that are not overriden in opts
    self.opts = self.default_opts | opts
    super(FusingPolicy, self).__init__()
    self.cam_type = cam_type
    self.config = config
    self.is_grasp = is_grasp
    self.use_proprio = use_proprio

    self.feat_size = self.opts["feat_size_per"] * sum([1 for ct in CamType.main4() if ct & self.cam_type])

    print(f"full feat size {self.feat_size = }")

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
          do_both = False,
        )
        self.mod_downer = CNNEncoder(64, layers = self.opts["film"]["downer_cnn_layers"]) ## use default layers
        ##NOTE: modulation happens at the resolution level need to downsample

        self.feat_size = self.mod_downer.flat_out_size

      case FuseConfig.W_Dfilm :
        ## colour modulated depth
        self._check_cam(needed=CamType.wrists())
        self._fail_if_cam(CamType.shoulders())

        self.film = FilmModulator(
          in1 = 1, ## wrist d, 
          in2 = 3, ## wrist rgb, 
          do_both = False,
        )
        self.mod_downer = CNNEncoder(64, layers = self.opts["film"]["downer_cnn_layers"]) ## use default layers
        self.feat_size = self.mod_downer.flat_out_size

      case FuseConfig.Wfilm_Dfilm:
        ## both ways then concatenate
        self._check_cam(needed=CamType.wrists())
        self._fail_if_cam(CamType.shoulders())

        self.film = FilmModulator(
          in1 = 3, ## wrist rgb, 
          in2 = 1, ## wrist d, 
          do_both = True,
        )
        self.mod_downer = CNNEncoder(128, layers = self.opts["film"]["double_downer_cnn_layers"]) ## double the size of last time, 2 of them
        self.feat_size = self.mod_downer.flat_out_size

      case FuseConfig.W_D_L_R:
        self.multi_enc = MultiCamCnn(self.cam_type)
        no_cams =  len([ct for ct in CamType.main4() if ct & self.cam_type])
        ## provide at least 3 cams here, otherwise no point
        if no_cams < 3: 
          raise RuntimeError(f"[fusing_policy - (FusingPolicy)] We want at least 3 cams here")
        self.feat_size = no_cams * 128 * 2 * 2

      case FuseConfig.W_D_L_R_FILM:
        # force all 4, cant be asked to figure it out for 3 no real point for 3
        self._check_cam(needed=CamType.main4())
        ## the order of fusing is w+d then l+r then wd+lr
        ##TODO modulate wrist rgb with depth, left and right with each other

        self.wd_film = FilmModulator(
          in1 = 3, 
          in2 = 1, 
          do_both=False
        )
        self.wd_downer = CNNEncoder(64, layers = self.opts["film"]["downer_cnn_layers"])

        self.lr_film = FilmModulator(
          in1 = 3, 
          in2 = 3, 
          do_both=True
        )

        self.lr_downer = CNNEncoder(128, layers = self.opts["film"]["double_downer_cnn_layers"])

        self.feat_size = self.lr_downer.flat_out_size + self.wd_downer.flat_out_size

      case FuseConfig.W_D_L_R_ATTN:
        self.mvt = MultiViewEncoder(
          cam_type=self.cam_type,
          **self.opts["mvt"]
        )
        self.feat_size = self.mvt.embed_dim
        ## NOTE: no cam checks here the attention thing already takes care of that

      case _:
        raise ValueError(f"[fusing_policy - (FusingPolicy)] unknown value for FuseConfig: '{self.config}")

    self.final_feat_size = self.feat_size
    ## Proprio and heads:
    if self.use_proprio:
      proprio_opts: dict= self.opts["proprio_opts"]
      self.jpos_feats = JointPosEncoder(**proprio_opts) if proprio_opts else JointPosEncoder()
      self.final_feat_size += self.jpos_feats.output_size
    
    #
    self.flatten = nn.Flatten()
    self.action_head = nn.Sequential(
      nn.Linear(self.final_feat_size, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 50),
      nn.ReLU(inplace=False),
      nn.Linear(50, action_shape - 1) ## predicts 8 - 1 dim action, no pose predication here
    )

    ## conditional on setting
    self.grasp_head = nn.Sequential(
      nn.Linear(self.final_feat_size, 128),
      nn.ReLU(inplace=False),
      nn.Linear(128, 64),
      nn.ReLU(inplace = False),
      nn.Linear(64, 1),
    ) if self.is_grasp else None
  
  def _fail_if_cam(self, fail_cases: list[CamType]):
    if any([self.cam_type & ct for ct in fail_cases]):
      raise ValueError(f"[fusing_policy - (fail_if_cam)] Given a can that is not compatible with this configuration! no '{fail_cases}")

  def _check_cam(self, needed: list[CamType] | None = None, any_one: list[CamType] | None = None):

    if needed is not None and not all([self.cam_type & ct for ct in needed]):
      raise ValueError(f"[fusing_policy - (check_cam)] Policy cam_type does not include all of '{needed}' -> current: '{self.cam_type}'")

    if any_one is not None and not any([self.cam_type & ct for ct in any_one]):
      raise ValueError(f"[fusing_policy - (check_cam)] Policy cam_type does not include any of '{any_one}' -> current: '{self.cam_type}'")
    
  def _feats_to_action(self, feats, proprio = None) -> torch.Tensor:
    feats = self.flatten(feats)
    if self.use_proprio:
      jfeats, _ = self.jpos_feats(proprio)
      feats = torch.cat([feats, jfeats], dim = -1) ## cat on feature dimension

    pose = self.action_head(feats)
    if self.is_grasp:  
      grasp = self.grasp_head(feats) #type: ignore (//NOTE: this is handled)
    else: 
      grasp = torch.ones((pose.shape[0], 1)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) ## just keep open

    return torch.cat([pose, grasp], dim = 1) ## (b, 8)
  
  ## this is used whent he "demo" options is selected for dataset, so we can catch the demos randomly but process in batch size
  def _collate_demos(self, batch)-> tuple[torch.Tensor, torch.Tensor, dict]:
    ## batch: [(tensor, tensor)] for inputs, labels
    inputs, labels, loader_dict = zip(*batch) #unzip the tuple list

    ## NOTE: handle other dict entries as well
    proprio = None
    if self.use_proprio:
      proprio = [d["proprio"] for d in loader_dict]## should always exist, might be empty
      proprio = torch.cat(proprio, dim=0)

    ## concat on the batch axis, preserve order of input to label
    
    
    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0), {
      "proprio": proprio, 
      "demo_lengths": torch.LongTensor([len(i) for i in inputs]) ## needed for the lambda k thing
    }
  
  def forward(self, image, proprio=None) -> tuple[torch.Tensor, dict]:
    if proprio is None and self.use_proprio:
      raise RuntimeError(f"[fusing_policy - forward] Expecting proprio data but none given!")

    ret_dict = {"config": self.config}
    match self.config:
      case  FuseConfig.WDLR:
        return self._feats_to_action(self.conv(image), proprio), ret_dict
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
          raise RuntimeError(f"[fusing_policy - forward] Cannot be here!")   
        
        return self._feats_to_action(fused_feats, proprio), ret_dict

      case FuseConfig.DEPTH_FEATS_ATTN:
        self._check_cam(needed = [CamType.WRIST], any_one = CamType.main3())
        rgbs = image[:, :-1, :, :] ## take all rgb cams
        depth = image[:, -1, :, :].unsqueeze(dim=1) # for (B, w, h) -> (B, 1, w, h)

        fused_feats, attn_dict = self.attn_feats(rgbs, depth)
        return self._feats_to_action(fused_feats, proprio), ret_dict | attn_dict
      
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
          return self._feats_to_action(fused_feats, proprio), ret_dict | attn_dict
        else: ## self.config == WD_LR
          wd_feats = self.wd_enc(wdX)
          lr_feats = self.lr_enc(lrX)
          cated = torch.cat([wd_feats, lr_feats], dim = 1)
          return self._feats_to_action(cated, proprio), ret_dict
        
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
        return self._feats_to_action(feats, proprio), ret_dict

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
        return self._feats_to_action(cated, proprio), ret_dict
      
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
        return self._feats_to_action(cated, proprio), ret_dict
      case FuseConfig.W_D_L_R_ATTN:
        
        feats, mvt_dict = self.mvt(image)
        return self._feats_to_action(feats, proprio), ret_dict | mvt_dict

      case _:
        raise ValueError(f"[fusing_policy - (FusingPolicy)] unknown value for FuseConfig: '{self.config}")


  def train_policy(self, 
    demos: list[Demo],
    epochs: int = 200,
    minibatch_size: int = 1, ## size of the observations currently being used
    lr: float = 0.01,
    data_label: Literal["joint_velocities", "joint_positions"] = "joint_velocities",
    shuffle_data = True, 
    shuffle_obs_in_demo = False,
    model_path: Optional[str] = None,
    lambda_grasp_loss: float = 1.,
    lock_loader_seed: Optional[int] = None, 
    dataset_to_use: Literal["obs", "demo"] = "demo",
    # last_k_grasp_mask: Optional[int] = None
  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params_str = params_string(
      epochs = epochs, 
      model_path = model_path,
      shuffle_data = shuffle_data,
      minibatch_size = minibatch_size,
      lr = lr, 
      data_label = data_label, 
      shuffle_obs_in_demo = f'{shuffle_obs_in_demo} [not being used!]',
      lambda_grasp_loss = lambda_grasp_loss,
      dataset_to_use = f'{dataset_to_use} [not being used!]',
      lock_loader_seed = lock_loader_seed,
      # last_k_grasp_mask = last_k_grasp_mask,
      device = device
    )
    
    print(f"Training params: {params_str}")

    if dataset_to_use != "demo":
      raise NameError(f"[fusing_policy - train_policy] Not allowing the other dataset anymore only allow 'demo'")
    
    shuffle_obs_in_demo = None

    model = self.to(device)

    dataset = DemoDataset(demos,
      cam_type=self.cam_type,
      get_type = "cat",
      use_proprio = self.use_proprio,
      label_get=data_label
    )

    loader = DataLoader(
      dataset,
      batch_size=minibatch_size,
      shuffle=shuffle_data,
      collate_fn=self._collate_demos, ## uses parents collater, see `simple_grasp_policy._collate_demos`
      generator=torch.manual_seed(lock_loader_seed) if lock_loader_seed else None
    )

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=None)
    mse_loss = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr = lr)

    self.action_losses = []
    self.grasp_losses = []

    model.train()
    for epoch in progress(range(epochs)):
      running_pose_loss, running_grasp_loss = 0., 0.
      
      for inputs, labels, loader_dict in loader:
        if dataset_to_use == "demo":
          inputs, labels = inputs.squeeze(), labels.squeeze()

        proprio_inputs = None
        if self.use_proprio:
          proprio_inputs = loader_dict["proprio"].squeeze()
          proprio_inputs = proprio_inputs.to(device)

        inputs, labels = inputs.to(device), labels.to(device)
        
        optimiser.zero_grad()
        pred_actions, _ = model(inputs, proprio = proprio_inputs)

        ## [:, x] to preserve the batch shape (batch_size, X)
        pose_loss = mse_loss(pred_actions[:, :-1], labels[:, :-1]) ## only the pose not he gripper action

        if self.is_grasp:
          grasp_loss = bce_loss(pred_actions[:, -1], labels[:, -1])
        else:
          grasp_loss = 0
        
        loss = pose_loss + lambda_grasp_loss * grasp_loss
        
        loss.backward()
        optimiser.step()

        running_pose_loss += pose_loss.item()
        running_grasp_loss += grasp_loss.item() if self.is_grasp else 0

      loss = (running_pose_loss + lambda_grasp_loss * running_grasp_loss) / len(loader)

      self.grasp_losses.append(running_grasp_loss / len(loader))
      self.action_losses.append(running_pose_loss / len(loader))
      
    print(f"Done Training Policy on {len(demos)} Demos") 
    

    
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence

from modules.cnns.cnn_encoder import CNNEncoder
from modules.cnns.multi_cam_cnn import MultiCamCnn
from modules.cnns.vit_encoder import MultiViewEncoder

from modules.film_net import FilmModulator
from modules.cnns.cross_attn_feats import CrossAttentionFeatures

from modules.joint_pos_encoder import JointPosEncoder
from modules.cnns.fusing_encoder import FusingEncoder
from modules.policy.fusing_policy import FusingPolicy

from modules.dataset.demo_dataset import DemoDataset
from lib.fuse_config import FuseConfig


##NOTE does not currently work with proprio not sure why yet
class FusingRNNPolicy(FusingPolicy):
  def __str__(self):
    return f"fusing_rnn_policy-fuse_config:{self.fuse_config}-is_grasp:{self.is_grasp}-use_proprio:{self.use_proprio}-fusing_opts:{self.fusing_opts}-proprio_opts:{self.proprio_opts}"
  
  def __repr__(self):
    return f"FusingRNNPolicy(fuse_config={self.fuse_config}, is_grasp={self.is_grasp}, use_proprio={self.use_proprio}, config={self.config}, fusing_opts={self.fusing_opts}, proprio_opts={self.proprio_opts})"
  
  default_rnn_opts = {
    "input_size" : 512,
    "hidden_size" : 256,
    "num_layers" : 2,
    "batch_first" : True,
    "bidirectional" : False, 
  }
  def __init__(self,
    action_shape: int, 
    cam_type: CamType,
    fuse_config: FuseConfig,
    is_grasp: bool,
    fusing_opts: dict = {},
    use_proprio: bool = False, 
    proprio_opts: dict = {},
    rnn_opts: dict = {}
  ):
    
    super().__init__(
      action_shape,
      cam_type,
      fuse_config,
      is_grasp,
      fusing_opts,
      use_proprio,
      proprio_opts,
    )
    self.rnn_opts = self.default_rnn_opts | rnn_opts
    
    self.rnn_opts["input_size"] = self.feats.final_feat_size

    self.rnn = nn.LSTM(
      **self.rnn_opts
    )
    self.rnn.flatten_parameters()

    if self.use_proprio:
      raise RuntimeError(f"I give up, I think the proprio unpadding isnt correct, but it doesnt fking matter cuz im not gonna have enough time to run nor examine it so no proprio for rnn, tough")

    self.final_feat_size = self.rnn_opts["hidden_size"] + (self.jpos_feats.output_size if self.use_proprio else 0)

    ## modify these to use the out encoding size of the rnn
    self.action_head[0] = nn.Linear(self.final_feat_size, 200)
    if self.is_grasp:
      self.grasp_head[0] = nn.Linear(self.final_feat_size, 128)

    self.rnn_hidden_size = self.rnn_opts["hidden_size"]

  ## output size is the hidden size, this can later be used to do whatever
  ## THIS IS FOR TRAINING WITH ENTIRE KNOWN LENGTHS
  def forward(self,
    image,
    lengths: Optional[torch.Tensor],
    hidden_state: Optional[tuple] = None, ## of tensors (h, c)
    proprio: Optional[torch.Tensor]= None,
  ): ## image here can contain channels from differnt camears including depth
    if proprio is None and self.use_proprio:
      raise RuntimeError(f"[fusing_rnn_policy - forward] Proprioceptive training selected but data not given!")
    ## we want these of shape: (B, t, c, w, h)
    ## t is the time series that is going to be fed into the lstm, so we want some ordering now from dataset

    if lengths is None:
      # print(f"{image.shape = }")
      # image = image.unsqueeze(0)## add seq_len = 1
      # print(f"{image.shape = }")
      
      feats, ret_dict = self.feats(image) ## (1, ..) because inference
      feats = feats.view(1, 1, -1) ## (1, 1, ..) add seq_len for lstm

      rnn_out, (h, c) = self.rnn(feats, hidden_state)
      enc = h[-1]

      return self._feats_to_action(enc, proprio), ret_dict | {
        "rnn_ret": rnn_out, 
        "h": h, 
        "c": c
      }
    

    B, t, ch, w, h = image.shape
    image = image.view(B * t, ch, w, h)
    # print(f"{image.shape =}")
    
    feats, ret_dict = self.feats(image)    
    feats = feats.view(B, t, -1)
    # print(f"{feats.shape =}")

    ## Shared until this point, then inference and training differs
    ## inference:

    packed_in = pack_padded_sequence(
      feats, lengths.cpu(), batch_first=True, enforce_sorted=False
    )

    packed_out, (h_n, c_n) = self.rnn(packed_in, hidden_state)

    rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)

    flat = rnn_out.reshape(B * t, -1)
    preds = self._feats_to_action(flat, proprio)
    preds = preds.view(B, t, -1)
    
    return preds, ret_dict | {
      "rnn_ret": rnn_out, ## might be useful to have down the line 
      "h_n": h_n,
      "h_c": c_n
    } ## returns the final last time step of rnn
  
  ## Passed to DemoDataset's DataLoader, so that the mismatch shaped demos can be padded accordingly
  def _collate_demos(self, batch):
    ## batch contains [(input, labels)] where each input is a complete demo (in terms of the data in sequence rgb for example)
    ## input: (t, ch, w, h) 
    inputs, labels, loader_dict = zip(*batch)

    ## enforcing types for later
    inputs: torch.Tensor
    labels: torch.Tensor

    real_lengths = torch.LongTensor([inp.shape[0] for inp in inputs])
    inputs_padded = pad_sequence(inputs, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True)

    ## NOTE: handle other dict entries as well
    proprio_padded = None
    if self.use_proprio:
      proprio = [d["proprio"] for d in loader_dict]## should always exist, might be empty
      proprio_padded = pad_sequence(proprio, batch_first=True)

    ## need to return shape (B, t, ch, w, h) for the input and labels
    ## also returning lenths for LSTM use later

    return inputs_padded, labels_padded, {
      "demo_lengths": real_lengths,
      "proprio": proprio_padded 
      } ## padded labels are the action at every step
  
  def train_policy(self,
    demos: list[Demo],
    epochs: int = 200,
    minibatch_size: int = 1,
    lr: float = 0.01,
    data_label: Literal["joint_velocities", "joint_positions"] = "joint_velocities",
    shuffle_data=True,
    shuffle_obs_in_demo=False,
    model_path: Optional[str] = None,
    lambda_grasp_loss: float = 1, 
    lock_loader_seed: Optional[int] = None,
    dataset_to_use: Literal['obs'] | Literal['demo'] = "demo",
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
      device = device
    )
    
    print(f"Training params: {params_str}")

    if dataset_to_use != "demo":
      raise NameError(f"[rnn_grasp_policy - train_policy] Not allowing the other dataset anymore only allow 'demo'")
    
    shuffle_obs_in_demo = None

    model = self.to(device)
    dataset = DemoDataset(demos,
      cam_type=self.cam_type,
      get_type = "cat",
      label_get=data_label,
      use_proprio=self.use_proprio

    )
    loader = DataLoader(
      dataset,
      batch_size=minibatch_size,
      shuffle=shuffle_data,
      collate_fn=self._collate_demos,
      generator=torch.manual_seed(lock_loader_seed) if lock_loader_seed else None
    )
    
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=None)
    mse_loss = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr = lr)

    model.train()
    self.losses = [0 for _ in range(epochs)]
    for epoch in progress(range(epochs)):
      total_pose_loss, total_grasp_loss = 0., 0.
      
      for inputs, labels, loader_dict in loader:
        lengths = loader_dict["demo_lengths"]
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

        proprio_inputs = None
        if self.use_proprio:
          proprio_inputs = loader_dict["proprio"]
          proprio_inputs = proprio_inputs.to(device)
          
        optimiser.zero_grad()
        
        
        pred_actions, _ = model(inputs, lengths, proprio=proprio_inputs)
        B, t, ad = pred_actions.shape

        ## [:, x] to preserve the batch shape (batch_size, X)
        mask = torch.arange(t)[None, :].to(device) < lengths[:, None]
        ## compare each time index to eaech seq's length
        # mask[b, t] = True if t < lengths[b] otherwise False

        pred_pose = pred_actions[:, :-1]   # (B, T, ...)
        true_pose = labels[:, :-1]


        pose_err = mse_loss(pred_pose, true_pose)       

        ## sum over pose dims
        pose_err = pose_err.sum(dim = -1) ## (B, t)
        pose_err  = pose_err * mask.float()
        
        num_valid = mask.sum()
        pose_loss = pose_err.sum() / num_valid

        pred_grasp = pred_actions[:,  -1]  # (B, T) 
        true_grasp = labels[:, -1]
        grasp_err = bce_loss(pred_grasp, true_grasp)  
        grasp_err = grasp_err * mask.float()
        grasp_loss = grasp_err.sum() / num_valid
        ## average over valid frames
        
        loss = pose_loss + lambda_grasp_loss * grasp_loss
        
        loss.backward()
        optimiser.step()

        total_pose_loss += pose_loss.item()
        total_grasp_loss += grasp_loss.item() 

        loss = (total_pose_loss + lambda_grasp_loss * total_grasp_loss) / len(loader)

        self.losses[epoch] = loss
      
    print(f"Done Training Policy on {len(demos)} Demos") 
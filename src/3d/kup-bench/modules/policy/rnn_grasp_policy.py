from typing import Literal, Optional

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence


from rlbench.demo import Demo

from tqdm import tqdm as progress
from lib.cam_type import CamType
from lib.utils import params_string

from modules.cnns.rnn_encoder import RNNEncoder

from modules.demo_dataset import DemoDataset

## Making a separate class/file here for this differnet than `SimpleGrasp` just so it is more convenient to tweak and experiment with


class RNNGraspPolicy(nn.Module): 

  def __str__(self):
    return f"rnn_grasp_policy-merge_feats:{self.rnn_opts}"
  
  def __repr__(self):
    return f"RNNGraspPolicy(merge_feats={self.rnn_opts})"
  
  def __init__(self,
    action_shape: int, 
    cam_type: CamType,
    rnn_opts: Optional[dict] = None
  ):
    super(RNNGraspPolicy, self).__init__() ##if inherining nn.Module
    self.rnn_opts = rnn_opts
    self.cam_type = cam_type

    if rnn_opts is not None:
      self.feats_encode = RNNEncoder(self.cam_type, **rnn_opts)

    self.feats_encode = RNNEncoder(self.cam_type)

    self.feat_size = self.feats_encode.encoding_size

    self.action_head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.feat_size, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 50),
      nn.ReLU(inplace=False),
      nn.Linear(50, action_shape - 1) ## predicts 8 - 1 dim action, no pose predication here
    )

    self.grasp_head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.feat_size, 128),
      nn.ReLU(inplace=False),
      nn.Linear(128, 64),
      nn.ReLU(inplace = False),
      nn.Linear(64, 1),
    )


  def _feats_to_action(self, feats) -> torch.Tensor:
    pose = self.action_head(feats)
    grasp = self.grasp_head(feats)

    return torch.cat([pose, grasp], dim = 1) ## (b, 8)


  def forward(self, 
    image, 
    lengths: Optional[torch.Tensor] = None,
    hidden_state: Optional[tuple] = None
  ) -> tuple[torch.Tensor, dict]:
    ## image: (B, T, ch, w, h)

    ## this means inference, training will provide lengths
    if lengths is None:
      feats, infer_dict = self.feats_encode.inference_forward(image, hidden_state=hidden_state)
      return self._feats_to_action(feats), infer_dict
    
    B, t, _, _, _ = image.shape

    ## NOTE: now gives the entire sequence
    rnn_out, rnn_dict = self.feats_encode(image, lengths)
    
    flat = rnn_out.reshape(B * t, -1)
    preds = self._feats_to_action(flat)
    preds = preds.view(B, t, -1)
    
    return preds, rnn_dict
  
  ## Passed to DemoDataset's DataLoader, so that the mismatch shaped demos can be padded accordingly
  def _collate_demos(self, batch):
    ## batch contains [(input, labels)] where each input is a complete demo (in terms of the data in sequence rgb for example)
    ## input: (t, ch, w, h) 
    inputs, labels = zip(*batch)

    ## enforcing types for later
    inputs: torch.Tensor
    labels: torch.Tensor

    real_lengths = torch.LongTensor([inp.shape[0] for inp in inputs])
    # print(f"[rnn_grasp_policy - _collate_demos] {real_lengths.shape =}")
    
    inputs_padded = pad_sequence(inputs, batch_first=True) ## CHECK: if it gives (B, t, ch, w, h)
    # print(f"[rnn_grasp_policy - _collate_demos] {inputs_padded.shape = }")

    labels_padded = pad_sequence(labels, batch_first=True) ## CHECK: if it gives (B, t, ch, w, h)
    # print(f"[rnn_grasp_policy - _collate_demos] {labels_padded.shape = }")


    ## need to return shape (B, t, ch, w, h) for the input and labels
    ## also returning lenths for LSTM use later
    return inputs_padded, labels_padded, real_lengths ## padded labels are the action at every step
  
  def train_policy(self,
    demos: list[Demo],
    epochs: int = 200,
    minibatch_size: int = 1,
    lr: float = 0.01,
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
    dataset = DemoDataset(demos, cam_type=self.cam_type, get_type = "cat")
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
      
      for inputs, labels, lengths in loader:
        # print(f"Now in (train)")
        
        # print(f"{inputs.shape = }")
        # print(f"{labels.shape = }")
        # print(f"{lengths.shape = }")

        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        # print(f"{inputs.shape =}") 
        # print(f"{labels.shape =}")
        
        optimiser.zero_grad()
        
        
        pred_actions, _ = model(inputs, lengths)
        B, t, ad = pred_actions.shape

        ## [:, x] to preserve the batch shape (batch_size, X)

        mask = torch.arange(t)[None, :].to(device) < lengths[:, None]
        ## compare each time index to eaech seq's length
        # mask[b, t] = True if t < lengths[b] otherwise False

        pred_pose = pred_actions[:, :-1]   # (B, T, ...)
        true_pose = labels[:, :-1]

        pred_grasp = pred_actions[:,  -1]  # (B, T) 
        true_grasp = labels[:, -1]

        pose_err = mse_loss(pred_pose, true_pose)       
        grasp_err = bce_loss(pred_grasp, true_grasp)  

        ## sum over pose dims
        pose_err = pose_err.sum(dim = -1) ## (B, t)
        pose_err  = pose_err * mask.float()
        grasp_err = grasp_err * mask.float()

        ## average over valid frames
        num_valid = mask.sum()
        pose_loss = pose_err.sum() / num_valid
        grasp_loss = grasp_err.sum() / num_valid

        
        loss = pose_loss + lambda_grasp_loss * grasp_loss
        
        loss.backward()
        optimiser.step()

        total_pose_loss += pose_loss.item()
        total_grasp_loss += grasp_loss.item()

        loss = (total_pose_loss + lambda_grasp_loss * total_grasp_loss) / len(loader)

        self.losses[epoch] = loss
        # N = len(loader)
        # print(f"Epoch {epoch}: PoseLoss={total_pose_loss/N:.4f}, GraspLoss={total_grasp_loss/N:.4f}")
      
    print(f"Done Training Policy on {len(demos)} Demos") 
    

    
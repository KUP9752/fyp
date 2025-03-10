import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm as progress

class Policy(nn.Module):
  def __init__(self, action_shape):
    super(Policy, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
      nn.ReLU(inplace=False),
    )
    flat_size = 2 * 2 * 128
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(flat_size, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 200),
      nn.ReLU(inplace=False),
      nn.Dropout(0.2),
      nn.Linear(200, 50),
      nn.ReLU(inplace=False),
      nn.Linear(50, action_shape)
    )
    
  def forward(self, image):
    feats = self.conv(image)
    return self.fc(feats)
  
  def train(self, 
            demos: list,
            epochs: int = 100,
            batch_size: int = 2,
            lr: float = 0.001,
            model_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loader = DataLoader(demos, batch_size = batch_size, shuffle = True);
    model = self.to(device)
    
    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr = lr)
    
    model.train()
    self.losses = [0 for _ in range(epochs)]
    for epoch in progress(range(epochs)):
      running_loss = 0
      for obs in loader:
        inputs, labels = obs.wrist_rgb, obs.joint_velocities
        inputs, labels = inputs.to(device), labels.to(device)
        optimiser.zero_grad()
        pred_actions = model(inputs)
        loss = loss_fn(pred_actions, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
      loss = running_loss / len(loader)
      self.losses[epoch] = loss
    print(f"Done Training Policy on Demos")
    
    if model_path:
      torch.save(self.state_dict(), f"{model_path}")

class Agent(object):

    def __init__(self, action_shape):
      self.action_shape = action_shape
      self.policy = Policy(action_shape)

    def ingest(self, demos):
      self.policy.train(demos)

    def act(self, obs):
      # arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
      # gripper = [1.0]  # Always open
      # return np.concatenate([arm, gripper], axis=-1)
      self.policy.eval()
      with torch.no_grad():
        pred = self.policy(obs.wrist_rgb)
        

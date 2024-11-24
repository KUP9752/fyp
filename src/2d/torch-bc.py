import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pickle
from tqdm import tqdm as progress
## Attempting behavioral cloning with pytorch

class AgentNetwork(nn.Module):
  def __init__(self, stateDim: int, actionDim: int):
    super(AgentNetwork, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(stateDim, 128),
      nn.ReLU(),
      nn.Linear(128, actionDim)
    )
    
  def forward(self, x):
    return self.fc(x)
  
def train_on_behaviour(dataFilepath: str) -> AgentNetwork:
  with open(dataFilepath, "rb") as f:
    data = pickle.load(f)
  
  states, actions = zip(*data)

  
  policy = AgentNetwork(4, 4)
  lossFunc = nn.MSELoss()
  optimiser = optim.Adam(policy.parameters(), lr = 0.01)
  
  dataset = TensorDataset(torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32))
  
  loader = DataLoader(dataset, batch_size=32, shuffle=True)
  
  print(f"Training on {len(dataset)} points")
  
  for epoch in progress(range(100)):
    for stateBatch, actionBatch in loader:
      predActions = policy(stateBatch)
      loss = lossFunc(predActions, actionBatch)
      optimiser.zero_grad()
      loss.backward()
      optimiser.step()
  
  print(f"Done training!")
  print(f"Saving...")
  torch.save(policy.state_dict(), "agent-network.pth")
  
  print(f"Saved!")
  
  return policy

  
train_on_behaviour("1k-points.pkl")

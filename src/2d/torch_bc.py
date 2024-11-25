import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pickle
from tqdm import tqdm as progress
from typing import Literal

from my_types import State, Action
## Attempting behavioral cloning with pytorch

class AgentNetwork(nn.Module):
  def __init__(self, stateDim: int, actionDim: int):
    super(AgentNetwork, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(stateDim, 128),
      nn.ReLU(),
      nn.Linear(128, actionDim),
    )
    
  def forward(self, x):
    return self.fc(x)
  
def train_on_behaviour(modelName: str = None, dataFilepath: str = None, data: list[tuple[State, Action]] = None, overwriteDevice: Literal['cpu', 'cuda'] | None = None ) -> AgentNetwork:
  if data is None and dataFilepath is None:
    raise ValueError("Must provide either 'dataFilePath' or 'data', dataFilePath takes priority if provided")
  
  if dataFilepath:
    with open(dataFilepath, "rb") as f:
      data = pickle.load(f)
  
  states, actions = zip(*data)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  if overwriteDevice:
    device = torch.device(overwriteDevice)
    
  print(f"Device to use: {device}")
  
  policy = AgentNetwork(4, 4).to(device)
  lossFunc = nn.BCEWithLogitsLoss()
  optimiser = optim.Adam(policy.parameters(), lr = 0.01)
  
  dataset = TensorDataset(torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32))
  
  loader = DataLoader(dataset, batch_size=32, shuffle=True)
  
  print(f"Training on {len(dataset)} points")
  policy.train()
  trainingLoss = 0
  
  for epoch in progress(range(100)):
    for stateBatch, actionBatch in loader:
      stateBatch, actionBatch = stateBatch.to(device), actionBatch.to(device)
      
      optimiser.zero_grad()
      predActions = policy(stateBatch)
      loss = lossFunc(predActions, actionBatch)
      loss.backward()
      optimiser.step()
      trainingLoss += loss.item()
      
    print(f" Loss: {trainingLoss / len(loader)}")
    
  print(f"Done training!")
  
  if modelName:
    print(f"Saving...")
    torch.save(policy.state_dict(), f"./src/2d/models/{modelName}")
    print(f"Saved!")
  
  
  return policy

# def play_game(model: AgentNetwork) -> None:
  

if __name__ == "__main__":
  print(f"'torch_bc' [main]")
  print(f"Check for cuda; {torch.cuda.is_available() = }")
  





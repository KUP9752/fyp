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
      nn.Linear(128, actionDim),
    )
    
  def forward(self, x):
    return self.fc(x)
  
def train_on_behaviour(dataFilepath: str, modelName: str) -> AgentNetwork:
  with open(dataFilepath, "rb") as f:
    data = pickle.load(f)
  
  states, actions = zip(*data)

  
  policy = AgentNetwork(4, 4)
  lossFunc = nn.BCEWithLogitsLoss()
  optimiser = optim.Adam(policy.parameters(), lr = 0.01)
  
  dataset = TensorDataset(torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32))
  
  loader = DataLoader(dataset, batch_size=32, shuffle=True)
  
  print(f"Training on {len(dataset)} points")
  policy.train()
  trainingLoss = 0
  
  for epoch in progress(range(100)):
    for stateBatch, actionBatch in loader:
      optimiser.zero_grad()
      predActions = policy(stateBatch)
      loss = lossFunc(predActions, actionBatch)
      loss.backward()
      optimiser.step()
      trainingLoss += loss.item()
      
    print(f"Epoch {epoch} Loss: {trainingLoss / len(loader)}")
    
  
  print(f"Done training!")
  print(f"Saving...")
  torch.save(policy.state_dict(), modelName)
  
  print(f"Saved!")
  
  return policy

# def play_game(model: AgentNetwork) -> None:
  

if __name__ == "__main__":
  train_on_behaviour("1k-targets.pkl", "agent-network-1k.pth")





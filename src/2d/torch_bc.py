import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pickle
from tqdm import tqdm as progress
from typing import Literal, Self

from my_types import State, Action4, Action2
## Attempting behavioral cloning with pytorch

class AgentNetwork(nn.Module):
  def __init__(self, stateDim: int, actionDim: int):
    super(AgentNetwork, self).__init__()
    self.stateDim = stateDim
    self.actionDim = actionDim
    
  def forward(self, x):
    return self.fc(x)
  
  def training_init(self, 
                    data: list = None, 
                    dataFilePath: str = None,
                    overwriteDevice: Literal['cpu', 'cuda'] | None = None) -> list:
    if data is None and dataFilePath is None:
      raise ValueError("Must provide either 'dataFilePath' or 'data', dataFilePath takes priority if provided")
    
    if dataFilePath:
      with open(dataFilePath, "rb") as f:
        data = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if overwriteDevice:
      device = torch.device(overwriteDevice)
      
    return data, device
  
  def training_save_model(self, modelName: str = None) -> None:
    if modelName:
      print(f"Saving...")
      torch.save(self.state_dict(), f"./src/2d/models/{modelName}")
      print(f"Saved!")
    
  
  
## Regression Task
class AgentNetwork_Regression(AgentNetwork):
  def __init__(self):
    super(AgentNetwork_Regression, self).__init__(stateDim=4, actionDim=2)
    self.fc = nn.Sequential(
      nn.Linear(self.stateDim, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, self.actionDim), ## output dx, dy
    )
    
  ## static method creates the model and trains it
  def train_on_behaviour(self, 
                         modelName: str = None, 
                         dataFilepath: str = None, 
                         data: list[tuple[State, Action2]] = None, 
                         overwriteDevice: Literal['cpu', 'cuda'] | None = None ) -> Self:
      
    ## !! Data is inherently of type list[tuple[State, Action4]] must be converted for this
    
    data, device = super().training_init(data, dataFilepath, overwriteDevice)
    print(f"Device to use: {device}")
    
    states, actions = zip(*data) 
    
    
    policy = self.to(device)
    lossFunc = nn.MSELoss()
    optimiser = optim.Adam(policy.parameters(), lr = 0.001)
    
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
    
    super().training_save_model(modelName)
    
    
    return policy
  
  
## Classification task
class AgentNetwork_Classification(AgentNetwork):
  def __init__(self):
    super(AgentNetwork_Classification, self).__init__(stateDim=4, actionDim=4)
    self.fc = nn.Sequential(
      nn.Linear(self.stateDim, 128),
      nn.ReLU(),
      nn.Linear(128, self.actionDim),
    )
  
  ## static method creates the model and trains it
  def train_on_behaviour(self, modelName: str = None,
                         dataFilepath: str = None,
                         data: list[tuple[State,Action4]] = None,
                         overwriteDevice: Literal['cpu','cuda'] | None = None ) -> Self:
    data, device = super().training_init(data, dataFilepath, overwriteDevice)
    
    states, actions = zip(*data)
      
    print(f"Device to use: {device}")
    
    policy = self.to(device)
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
    
    super().training_save_model(modelName)
    
    return policy
    
  

# def play_game(model: AgentNetwork) -> None:
  

if __name__ == "__main__":
  print(f"'torch_bc' [main]")
  print(f"Check for cuda; {torch.cuda.is_available() = }")
  





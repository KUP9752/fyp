import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms

import pandas as pd
from PIL import Image

import os

import pickle
from tqdm import tqdm as progress
from typing import Literal, Self

from my_types import State, Action4, Action2
## Attempting behavioral cloning with pytorch

def printc(condition: bool | None, text: str) -> None:
  if condition:
    print(text)

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
                    overwriteDevice: Literal['cpu', 'cuda'] | None = None,
                    ) -> list:
    if data is None and dataFilePath is None:
      raise ValueError("Must provide either 'dataFilePath' or 'data', dataFilePath takes priority if provided")
    
    if dataFilePath:
      with open(dataFilePath, "rb") as f:
        data = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if overwriteDevice:
      device = torch.device(overwriteDevice)
      
    return data, device
  
  # def training_save_model(self, modelName: str = None, doPrints: bool = False) -> None:
  #   if modelName:
  #     printc(doPrints, f"Saving...")
  #     torch.save(self.state_dict(), f"./models/{modelName}")
  #     printc(doPrints, f"Saved!")
    
  
class PositionDataset(Dataset):
  def __init__(self, imagesDir: str, imageCoordsPath: str, transform = None):
    
    self.imagesDir = imagesDir
    self.imageCoords = pd.read_pickle(imageCoordsPath)
    self.transform = transform
    
  def __len__(self):
    return len(self.imageCoords)
  
  def __getitem__(self, idx):
    imageName = self.imageCoords.iloc[idx].name
    imagePath = os.path.join(self.imagesDir, imageName)
    image = Image.open(imagePath)
    label = self.imageCoords.iloc[idx]
    
    if self.transform:
      image = self.transform(image)
    
    return image, torch.tensor(label, dtype=torch.float32)
  
class PositionPredictor(nn.Module):
  def __init__(self, 
               lossFunc = nn.MSELoss(),
               lr: float = 0.001,
               epochs: int = 100,
               batchSize: int = 32):
    super(PositionPredictor, self).__init__()
    self.losses = None
    self.lossFunc = lossFunc
    self.batchSize = batchSize
    self.lr = lr
    self.epochs = epochs
    
    self.cnn = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d((4, 4))  # Reduce to fixed size
    )
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(64 * 4 * 4, 128),
      nn.ReLU(),
      nn.Linear(128, 4)  # Output 4 values: agent_x, agent_y, target_x, target_y
    )
    
  def forward(self, x):
    x_cnn= self.cnn(x)
    return self.fc(x_cnn)
  
  def train_on_images(self, 
                      imagesDir: str, #directory of the images to train on
                      imageCoordsPath: str, # DataFrame image-name -> State
                      modelPath: str = None, 
                      overwriteDevice: Literal['cpu', 'cuda'] | None = None,
                      doPrints: bool = False) -> Self:
    if overwriteDevice:
      device = torch.device(overwriteDevice)
    else:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
      transforms.ToTensor(),
    ])
    
    trainingData = PositionDataset(imagesDir, imageCoordsPath, transform)
    loader = DataLoader(trainingData, batch_size=self.batchSize, shuffle=True)
    
    model = self.to(device)
    optimiser = optim.Adam(model.parameters(), lr = self.lr)
    
    printc(doPrints, f"Training on {len(trainingData)} points")
    
    model.train()
    self.losses = [0 for _ in range(self.epochs)]
    
    for epoch in progress(range(self.epochs)):
      runningLoss = 0
      for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimiser.zero_grad()
        predActions = model(images)
        loss = self.lossFunc(predActions, labels)
        loss.backward()
        optimiser.step()
        runningLoss += loss.item()
        
      loss = runningLoss / len(loader)
      self.losses[epoch] = loss
      printc(doPrints, f" [{epoch}/{self.epochs}] Loss: {loss}")
      
      
      
    printc(doPrints, f"Done training!")
    
    if modelPath:
      printc(doPrints, f"Saving...")
      torch.save(self.state_dict(), f"{modelPath}")
      printc(doPrints, f"Saved!")
    self.loss = runningLoss
    
    return model
    
## Regression Task
class AgentNetwork_Regression(AgentNetwork):
  # batchSize: int
  # lossFunc: nn.Loss
  # lr: float = 0.001
  def __init__(self, 
               npl: int = 64, # neurons per layer
               lossFunc = nn.MSELoss(),
               lr: float = 0.001,
               epochs: int = 100,
               batchSize: int = 32):
    super(AgentNetwork_Regression, self).__init__(stateDim=4, actionDim=2)
    self.losses = None
    self.lossFunc = lossFunc
    self.batchSize = batchSize
    self.lr = lr
    self.epochs = epochs
    
    
    ## currently 1 input 1 hidden 1 output
    self.fc = nn.Sequential(
      nn.Linear(self.stateDim, npl),
      nn.ReLU(),
      nn.Linear(npl, npl),
      nn.ReLU(),
      nn.Linear(npl, self.actionDim), ## output dx, dy
    )
    
  
  def preprocess_data(data: list[tuple[State, Action4]]) -> list[tuple[State, Action2]]:
    ## (left - right, up - down) for (dx, dy)
    
    return [(state, (int(action[3]) - int(action[2]), int(action[0]) - int(action[1]))) for state, action in data]
  
  ## static method creates the model and trains it
  def train_on_behaviour(self, 
                         modelPath: str = None, 
                         dataFilepath: str = None, 
                         data: list[tuple[State, Action4]] = None, 
                         overwriteDevice: Literal['cpu', 'cuda'] | None = None,
                         doPrints: bool = False) -> Self:
      
    ## !! Data is inherently of type list[tuple[State, Action4]] must be converted for this
    
    data, device = super().training_init(data, dataFilepath, overwriteDevice)
    
    ## process data to be [(State, Action2)]
    
    data = AgentNetwork_Regression.preprocess_data(data)
    
    
    printc(doPrints, f"Device to use: {device}")
    
    states, actions = zip(*data) ## Action2 at this point
    
    
    policy = self.to(device)
    optimiser = optim.Adam(policy.parameters(), lr = self.lr)
    
    dataset = TensorDataset(torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32))
    
    loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True)
    
    printc(doPrints, f"Training on {len(dataset)} points")
    policy.train()
    self.losses = [0 for _ in range(self.epochs)]
    
    for epoch in progress(range(self.epochs)):
      runningLoss = 0
      for stateBatch, actionBatch in loader:
        stateBatch, actionBatch = stateBatch.to(device), actionBatch.to(device)
        
        optimiser.zero_grad()
        predActions = policy(stateBatch)
        loss = self.lossFunc(predActions, actionBatch)
        loss.backward()
        optimiser.step()
        runningLoss += loss.item()
        
      loss = runningLoss / len(loader)
      self.losses[epoch] = loss
      printc(doPrints, f" [{epoch}/{self.epochs}] Loss: {loss}")
      
      
      
    printc(doPrints, f"Done training!")
    
    if modelPath:
      printc(doPrints, f"Saving...")
      torch.save(self.state_dict(), f"{modelPath}")
      printc(doPrints, f"Saved!")
    self.loss = runningLoss
    
    return policy
  
  def get_training_losses(self) -> list[float]:
    if self.losses is None:
      raise ValueError("Not trained yet!")
    return self.losses
  
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
  def train_on_behaviour(self, modelPath: str = None,
                         dataFilepath: str = None,
                         data: list[tuple[State,Action4]] = None,
                         epochs: int = 100,
                         overwriteDevice: Literal['cpu','cuda'] | None = None ) -> Self:
    data, device = super().training_init(data, dataFilepath, overwriteDevice)
    
    states, actions = zip(*data)
      
    print(f"Device to use: {device}")
    
    policy = self.to(device)
    lossFunc = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(policy.parameters(), lr = 0.01)
    
    dataset = TensorDataset(torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32))
    
    loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True)
    
    print(f"Training on {len(dataset)} points")
    policy.train()
    trainingLoss = 0
    
    for epoch in progress(range(epochs)):
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
    
    if modelPath:
      print(f"Saving...")
      torch.save(self.state_dict(), f"{modelPath}")
      print(f"Saved!")
    
    return policy
    
  

# def play_game(model: AgentNetwork) -> None:
  

if __name__ == "__main__":
  print(f"'torch_bc' [main]")
  print(f"Check for cuda; {torch.cuda.is_available() = }")
  





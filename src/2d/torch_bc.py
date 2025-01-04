import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms

import pandas as pd
from PIL import Image

import re
import os
import random

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
  def __init__(self, imagesDir: str, labelsPath: str, frac:float = 1.0, transform = None):
    
    self.imagesDir = imagesDir
    self.imageLabels = pd.read_pickle(labelsPath)
    self.imageLabels = self.imageLabels.sample(frac=frac)
    self.transform = transform
    
    raise ValueError(f"No Layers configured for this, not using this for now")
    
  def __len__(self):
    return len(self.imageLabels)
  
  def __getitem__(self, idx):
    imageName = self.imageLabels.iloc[idx].name
    imagePath = os.path.join(self.imagesDir, imageName)
    image = Image.open(imagePath)
    toExtract = ["agent_x", "agent_y", "target_x", "target_y"]
    label = label = self.imageLabels.loc[self.imageLabels.index[idx], toExtract].values.astype(np.float32)
    
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
    self.transform = transforms.Compose([
      transforms.ToTensor(),
    ])
    
    
  def forward(self, x):
    x = self.cnn(x)
    # x = x.view(x_cnn.size(0), -1)  # Flatten
    return self.fc(x)
  
  def transform_image(self, image):
    if self.transform:
      return self.transform(image)
    raise ValueError("No transform set, means model hasn't been trained yet")
  
  def train_on_behaviour(self, 
                      imagesDir: str, #directory of the images to train on
                      imageLabelsPath: str, # DataFrame image-name -> State
                      modelPath: str = None, 
                      overwriteDevice: Literal['cpu', 'cuda'] | None = None,
                      sizeFrac: float = 1.0,
                      doPrints: bool = False) -> Self:
    if overwriteDevice:
      device = torch.device(overwriteDevice)
    else:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainingData = PositionDataset(imagesDir, imageLabelsPath, frac = sizeFrac, transform=self.transform)
    loader = DataLoader(trainingData, batch_size=self.batchSize, shuffle=True)
    
    model = self.to(device)
    optimiser = optim.Adam(model.parameters(), lr = self.lr)
    
    printc(doPrints, f"Training on {len(trainingData)} points")
    
    model.train()
    self.losses = [0 for _ in range(self.epochs)]
    
    for epoch in progress(range(self.epochs)):
      runningLoss = 0
      for images, labels in loader:
        printc(doPrints, f"images shape: {images.shape}")
        printc(doPrints, f"labels shape: {labels.shape}")
        
        images, labels = images.to(device), labels.to(device)
        
        optimiser.zero_grad()
        predActions = model(images)
        printc(doPrints, f"predActions shape: {predActions.shape}")
        
        loss = self.lossFunc(predActions, labels)
        loss.backward()
        optimiser.step()
        runningLoss += loss.item()
        
      loss = runningLoss / len(loader)
      self.losses[epoch] = loss
      printc(True, f" [{epoch}/{self.epochs}] Loss: {loss}")
      
      
      
    printc(doPrints, f"Done training!")
    
    if modelPath:
      printc(doPrints, f"Saving...")
      torch.save(self.state_dict(), f"{modelPath}")
      printc(doPrints, f"Saved!")
    self.loss = runningLoss
    
    return model


class SequentialDataset(Dataset):
  def __init__(self, 
    imagesDir: str,
    nFrames: int,
    labelsPath: str,
    frac: float = 1.0,
    transform = None,
    closeness: dict[int, float] = False,
    N: int = 1000,
  ):
    self.nFrames = nFrames
    self.transform = transform
    self.imagesDir = imagesDir
    print(f"{labelsPath = }")
    data = pd.read_pickle(labelsPath)
    
    ## if closeness is not None, balance the data according to closeness column values
    if closeness:
      ## index is the first number, [0] is the string being searched
      data["imageNo"] = data.index.map(lambda s: int(re.search(r"ss-(\d+)-close-(\d+)-seq-(\d+).png", s)[1])) 
      
      ## if a closeness to weights is given
      ## sample the indices as a fraction of the total given by the percentage, N = 1k by default
      sampledIndices = {c : np.random.choice(np.arange(0, 1000), size=int(frac * N)) for c, frac in closeness.items()}
      gs = [data[(data["closeness"] == c) & (data["imageNo"].isin(sampledIndices[c]))] for c, _ in closeness.items()]
      self.imageLabels = pd.concat(gs)
    else:
      ## TODO this is wrong, beacuse we frac
      ## automatically preserves the sequences as everything is kept.
      self.imageLabels = self.imageLabels.sample(frac=frac)
        
    
  def __len__(self):
    return len(self.imageLabels)
  
  def __getitem__(self, idx):
    idxName = self.imageLabels.iloc[idx].name
    s = re.search(r"ss-(\d+)-close-(\d+)-seq-(\d+).png", idxName)
    index = int(s[1])
    closeness = int(s[2])
    
    ## get all the images with this index:
    seq = self.imageLabels[self.imageLabels.index.str.contains(f"ss-{index}-close-{closeness}-seq-")]
    ## assuming this gets the values sequentially, choose consequitve `nFrames` of them
    
    ## if sequence is shorter than chosen frames take it all and pad with 0 tensors.
    if len(seq) < self.nFrames:
      chosen = seq

    else:
      i = random.randint(0, len(seq) - self.nFrames)
      chosen = seq[i: i + self.nFrames]
    
    images = [Image.open(os.path.join(self.imagesDir, imageName)) for imageName in chosen.index]
    ## NOTE: sequence to one, [images] -> Action2, or seq-seq: [images] -> [Action2] currently seq-one
    label = chosen.loc[chosen.index[-1], "action"]
    images = [self.transform(image) for image in images] if self.transform else images
    
    nMissing = self.nFrames - len(images)
    if nMissing > 0:
      ## leave the zeros_like at the front
      images = [torch.zeros_like(images[0]) for _ in range(nMissing)] + images
    
    
    return torch.cat(images), torch.tensor(label, dtype=torch.float32)


class CNN_RegressionSequences(nn.Module):
  def __init__(self, 
               nFrames: int,
               lossFunc = nn.MSELoss(),
               lr: float = 0.001,
               epochs: int = 100,
               batchSize: int = 32):
    super(CNN_RegressionSequences, self).__init__()
    self.losses = None
    self.lossFunc = lossFunc
    self.batchSize = batchSize
    self.lr = lr
    self.epochs = epochs
    self.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    self.nFrames = nFrames
    self.cnn = nn.Sequential(
      nn.Conv2d(3 * nFrames, 16, kernel_size=5, stride=2, padding=2), # 400 x 300
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 200 x 150
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 100 x 75
      nn.ReLU(),
    )
    
    flattenedSize = 64 * (800 // (2**3)) * (600 // (2**3)) #  Width and height divided by stride (2^3 = 8)
    
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(flattenedSize, 128),
      # nn.Linear(32 * 200 * 150, 128),
      nn.ReLU(),
      nn.Linear(128, 2), #( dx, dy)
    )
    
  def forward(self, x):
    x = self.cnn(x)  # CNN feature extractor
    x = self.fc(x)  # Fully connected layers
    return x
  
  def transform_image(self, image):
    if self.transform:
      return self.transform(image)
    raise ValueError("No transform set, means model hasn't been trained yet")
  
  def train_on_behaviour(self, 
                      imagesDir: str, #directory of the images to train on
                      imageLabelsPath: str, # DataFrame image-name -> State
                      modelPath: str = None, 
                      overwriteDevice: Literal['cpu', 'cuda'] | None = None,
                      sizeFrac: float = 1.0,
                      doPrints: bool = False) -> Self:
    if overwriteDevice:
      device = torch.device(overwriteDevice)
    else:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## closeWeights: {closeness: weight}
    closeWeights = {
      0: 0.0,
      1: 0.0,
      2: 0.0,
      3: 0.3,
      4: 0.3, 
      5: 0.6, 
    }
    
    # python play.py train -m cnn-regr -d ./datasets/screenshots-obs-seq-1k-withcols-closeness  -f .\models\closeness-3conv-k5s2p2-obs\1-1-1weights -e 20
    
    trainingData = SequentialDataset(
      imagesDir,
      self.nFrames,
      imageLabelsPath,
      frac = sizeFrac,
      N = 1000,
      transform=self.transform,
      closeness=closeWeights,
    )
    ## NOTE: not shuffling does not seem to work at all, will try again? shuffling needs a betgteer combination of fractions
    loader = DataLoader(trainingData, batch_size=self.batchSize, shuffle=True)
    
    model = self.to(device)
    optimiser = optim.Adam(model.parameters(), lr = self.lr, weight_decay=1e-4)
    print(f"Using device: {device}")
    printc(doPrints, f"Training on {len(trainingData)} points")
    printc(True, f"Training on {len(trainingData)} points")
    
    model.train()
    self.losses = [0 for _ in range(self.epochs)]
    
    for epoch in progress(range(self.epochs)):
      runningLoss = 0
      for images, labels in loader:
        printc(doPrints, f"images shape: {images.shape}")
        printc(doPrints, f"labels shape: {labels.shape}")
        
        images, labels = images.to(device), labels.to(device)
        
        optimiser.zero_grad()
        predActions = model(images)
        printc(doPrints, f"predActions shape: {predActions.shape}")
        
        loss = self.lossFunc(predActions, labels)
        loss.backward()
        optimiser.step()
        runningLoss += loss.item()
        
      loss = runningLoss / len(loader)
      self.losses[epoch] = loss
      printc(True, f" [{epoch}/{self.epochs}] Loss: {loss}")
      
      
      
    printc(doPrints, f"Done training!")
    
    if modelPath:
      printc(doPrints, f"Saving...")
      torch.save(self.state_dict(), f"{modelPath}")
      printc(doPrints, f"Saved!")
    self.loss = runningLoss
    
    return model
    
class MovementDataset(Dataset):
  def __init__(self, imagesDir: str,
    labelsPath: str,
    frac: float = 1.0,
    transform = None,
    closeness: dict[int, float] = False,
    N: int = 1000,
    preserveMovementSequences: bool = False
  ):
    
    self.transform = transform
    self.imagesDir = imagesDir
    print(f"{labelsPath = }")
    data = pd.read_pickle(labelsPath)
    
    ## if closeness is not None, balance the data according to closeness column values
    if preserveMovementSequences:
      if closeness:
        ## index is the first number, [0] is the string being searched
        data["imageNo"] = data.index.map(lambda s: int(re.search(r"ss-(\d+)-close-(\d+)-seq-(\d+).png", s)[1])) 
        
        ## if a closeness to weights is given
        if closeness:
          ## sample the indices as a fraction of the total given by the percentage, N = 1k by default
          sampledIndices = {c : np.random.choice(np.arange(0, 1000), size=int(frac * N)) for c, frac in closeness.items()}
          gs = [data[(data["closeness"] == c) & (data["imageNo"].isin(sampledIndices[c]))] for c, _ in closeness.items()]
          self.imageLabels = pd.concat(gs)
        else:
          ## automatically preserves the sequences as everything is kept.
          self.imageLabels = self.imageLabels.sample(frac=frac)
    else:
      ## if a closeness to weights is given
      if closeness:
        gs = [data[data["closeness"] == c].sample(frac=frac) for c, frac in closeness.items()]
        self.imageLabels = pd.concat(gs)
      else:
        self.imageLabels = self.imageLabels.sample(frac=frac)
        
    
  def __len__(self):
    return len(self.imageLabels)
  
  def __getitem__(self, idx):
    imageName = self.imageLabels.iloc[idx].name
    imagePath = os.path.join(self.imagesDir, imageName)
    image = Image.open(imagePath)
    label = self.imageLabels.loc[self.imageLabels.index[idx], "action"]
    
    if self.transform:
      image = self.transform(image)
      
    return image, torch.tensor(label, dtype=torch.float32)
class CNN_Regression(nn.Module):
  def __init__(self, 
               lossFunc = nn.MSELoss(),
               lr: float = 0.001,
               epochs: int = 100,
               batchSize: int = 32):
    super(CNN_Regression, self).__init__()
    self.losses = None
    self.lossFunc = lossFunc
    self.batchSize = batchSize
    self.lr = lr
    self.epochs = epochs
    self.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    
    self.cnn = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), # 400 x 300
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 200 x 150
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 100 x 75
      nn.ReLU(),
    )
    
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(64 * 100 * 75, 128),
      # nn.Linear(32 * 200 * 150, 128),
      nn.ReLU(),
      nn.Linear(128, 2), #( dx, dy)
    )
    
    ## Kinda works for 50x50 (./models/big-closeness-3-conv-k5-s2-p2)
    # self.cnn = nn.Sequential(
    #   nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), # 400 x 300
    #   nn.ReLU(),
    #   nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 200 x 150
    #   nn.ReLU(),
    #   nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 100 x 75
    #   nn.ReLU(),
    # )
    
    # self.fc = nn.Sequential(
    #   nn.Flatten(),
    #   nn.Linear(64 * 100 * 75, 128),
    #   # nn.Linear(32 * 200 * 150, 128),
    #   nn.ReLU(),
    #   nn.Linear(128, 2), #( dx, dy)
    # )
    
    ## KINDA WORKS FOR 20x20 (./models/closeness-4-conv-k5-s-p2)
    # 800 x 600
    # self.cnn = nn.Sequential(
    #   nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), # 400 x 300
    #   nn.ReLU(),
    #   nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 200 x 150
    #   nn.ReLU(),
    #   nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 100 x 75
    #   nn.ReLU(),
    #   nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), # 50 x 37
    #   nn.ReLU(),
    # )
    
    # self.fc = nn.Sequential(
    #   nn.Flatten(),
    #   # nn.Linear(64 * 100 * 75, 128),
    #   nn.Linear(243200, 128), ## no idea why this size ngl
    #   # nn.Linear(32 * 200 * 150, 128),
    #   nn.ReLU(),
    #   nn.Linear(128, 2), #( dx, dy)
    # )

    
    
  def forward(self, x):
    x = self.cnn(x)  # CNN feature extractor
    x = self.fc(x)  # Fully connected layers
    return x
  
  def transform_image(self, image):
    if self.transform:
      return self.transform(image)
    raise ValueError("No transform set, means model hasn't been trained yet")
  
  def train_on_behaviour(self, 
                      imagesDir: str, #directory of the images to train on
                      imageLabelsPath: str, # DataFrame image-name -> State
                      modelPath: str = None, 
                      overwriteDevice: Literal['cpu', 'cuda'] | None = None,
                      sizeFrac: float = 1.0,
                      doPrints: bool = False) -> Self:
    if overwriteDevice:
      device = torch.device(overwriteDevice)
    else:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## closeWeights: {closeness: weight}
    closeWeights = {
      0: 0.0,
      1: 0.0,
      2: 0.0,
      3: 0.2,
      4: 0.2, 
      5: 0.5, 
    }
    
    # python play.py train -m cnn-regr -d ./datasets/screenshots-obs-seq-1k-withcols-closeness  -f .\models\closeness-3conv-k5s2p2-obs\1-1-1weights -e 20
    
    trainingData = MovementDataset(imagesDir,
      imageLabelsPath,
      frac = sizeFrac,
      transform=self.transform,
      closeness=closeWeights,
      preserveMovementSequences=True
    )
    ## NOTE: not shuffling does not seem to work at all, will try again? shuffling needs a betgteer combination of fractions
    loader = DataLoader(trainingData, batch_size=self.batchSize, shuffle=True)
    
    model = self.to(device)
    optimiser = optim.Adam(model.parameters(), lr = self.lr, weight_decay=1e-4)
    print(f"Using device: {device}")
    printc(doPrints, f"Training on {len(trainingData)} points")
    printc(True, f"Training on {len(trainingData)} points")
    
    model.train()
    self.losses = [0 for _ in range(self.epochs)]
    
    for epoch in progress(range(self.epochs)):
      runningLoss = 0
      for images, labels in loader:
        printc(doPrints, f"images shape: {images.shape}")
        printc(doPrints, f"labels shape: {labels.shape}")
        
        images, labels = images.to(device), labels.to(device)
        
        optimiser.zero_grad()
        predActions = model(images)
        printc(doPrints, f"predActions shape: {predActions.shape}")
        
        loss = self.lossFunc(predActions, labels)
        loss.backward()
        optimiser.step()
        runningLoss += loss.item()
        
      loss = runningLoss / len(loader)
      self.losses[epoch] = loss
      printc(True, f" [{epoch}/{self.epochs}] Loss: {loss}")
      
      
      
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
    
  
  # def preprocess_data(data: list[tuple[State, Action4]]) -> list[tuple[State, Action2]]:
  #   ## (right - left, up - down) for (dx, dy)
    
  #   return [(state, (int(action[3]) - int(action[2]), int(action[0]) - int(action[1]))) for state, action in data]
  
  ## static method creates the model and trains it
  def train_on_behaviour(self, 
                         modelPath: str = None, 
                         dataFilepath: str = None, 
                         data: list[tuple[State, Action2]] = None, 
                         overwriteDevice: Literal['cpu', 'cuda'] | None = None,
                         sizeFrac: float = 1.0,
                         doPrints: bool = False) -> Self:
      
    data, device = super().training_init(data, dataFilepath, overwriteDevice)
    
    print(f"Device to use: {device}")
    
    if sizeFrac < 1.0:
      data = random.sample(data, int(sizeFrac * len(data)))
    
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
                         sizeFrac: float = 1.0,
                         overwriteDevice: Literal['cpu','cuda'] | None = None ) -> Self:
    data, device = super().training_init(data, dataFilepath, overwriteDevice)
    
    if sizeFrac < 1.0:
      data = random.sample(data, int(sizeFrac * len(data)))
    
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
    
if __name__ == "__main__":
  print(f"'torch_bc' [main]")
  print(f"Check for cuda; {torch.cuda.is_available() = }")
  





import pygame
from pygame.color import Color
from PIL import Image

import pandas as pd
import pickle
import random
import time
from typing import Literal, Type, Callable, TypeVar

from my_types import State, Action4, Movement, Action2


SPEED = 2.0
FPS = 60
MOV_SPEED = 5
BLOCK_SIZE = 20 # per side

WIDTH, HEIGHT = 800, 600
X_BOUND = WIDTH - BLOCK_SIZE
Y_BOUND = HEIGHT - BLOCK_SIZE

MOVEMENT: dict = {
    pygame.K_UP: (0, -MOV_SPEED),
    pygame.K_DOWN: (0, MOV_SPEED),
    pygame.K_LEFT: (-MOV_SPEED, 0),
    pygame.K_RIGHT: (MOV_SPEED, 0)
  }

## Movement Behaviour to be used by the 'learn_game' 
def auto_policy(agent: pygame.Rect, target: pygame.Rect) -> Action2:
  dx = target.x - agent.x
  dy = target.y - agent.y
  mag = (dx**2 + dy**2)**0.5
  
  if mag > 0:
    dx /= mag
    dy /= mag    
  
  dx = int(dx * MOV_SPEED)
  dy = int(dy * MOV_SPEED)
  
  agent.move_ip(dx, dy)
  
  return dx, dy

## File to save the demonstration data to train on
def learn_game(filepath: str = None,
               moveAgent: Callable[[pygame.Rect, pygame.Rect], Action2] = auto_policy,
               n = 10) -> list[tuple[State, Action4]]:

  pygame.init()
  clock = pygame.time.Clock()

  ## Setup Screen
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("2D Canvas")

  agent = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  target = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)


  isRunning = True
  
  data: list[tuple[State, Action2]] = []
  
  targetCount = 1 ## one target at the start
  startTime = time.perf_counter()

  while isRunning:
    ## White Background
    screen.fill(Color("white"))
      
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        isRunning = False
      if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        print(f"Agent(x={agent.x}, y={agent.y}) Target(x={target.x}, y={target.y})")
        
          
    ## Game Logic
    ## draw the squares, agend is BLUE, target is RED
    pygame.draw.rect(screen, Color("blue"), agent)
    pygame.draw.rect(screen, Color("red"), target)
    
    ## record the action given to the robot in this coord system
    state = agent.x, agent.y, target.x, target.y
    
    action: Action2 = moveAgent(agent, target)
    
    ## When collided restart the target, so the game continuosly runs
    if agent.colliderect(target):
      targetCount += 1
      target.x = random.randint(0, X_BOUND)
      target.y = random.randint(0, Y_BOUND)
      # agent.x = random.randint(0, X_BOUND)
      # agent.y = random.randint(0, Y_BOUND)
    
      
    ## save the data from this frame  
    data.append((state, action))
    
    ## Finish learning when n targets are reached
    if targetCount == n:
      isRunning = False
    
    pygame.display.update()
    clock.tick(FPS)

  pygame.quit()
  
  ## Once the Game Ends save the data
  elapsedTime = time.perf_counter() - startTime
  print(f"Elapsed Time: {elapsedTime} for {n} targets -> (state, action) datapoints")
  print(f"Writing to file {filepath}")
  
  if filepath:
    with open(f"./datasets/{filepath}", "wb") as f:
      pickle.dump(data, f)
      
  return data
  
from torch_bc import AgentNetwork, AgentNetwork_Classification, AgentNetwork_Regression, CNN_Regression, PositionPredictor
from torch import nn



def move_arrowkeys(agent: pygame.Rect, target: pygame.Rect) -> None:
  keys = pygame.key.get_pressed()
  
  for key, (dx, dy) in MOVEMENT.items():
    if keys[key]:
      agent.move_ip(dx, dy)
  
  
def move_classification(agent: pygame.Rect, target: pygame.Rect, model: AgentNetwork_Classification) -> None:
  state = agent.x, agent.y, target.x, target.y
  state = torch.tensor(state, dtype=torch.float32)
  action = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT:  False }
  
  model.eval()
  with torch.no_grad():
    pred = model(state) ## currently returns 4 floats, do some post processing
    ## if below a certain threshold reject it
    print(f"{pred = }")
    print(f"{target = }")
    print(f"{agent = }")
    
    
    ## Thresholding didn't seem to be working consistenty with BCEWithLogitsLoss
    # pred = torch.where(pred > 0.5, pred, torch.tensor(0.0))
    # print(f"After filter {pred = }")
    
    ## map into key pairs
    if pred[0] > pred[1]:
      action[pygame.K_UP] = True
    elif pred[1] > pred[0]:
      action[pygame.K_DOWN] = True
      
    if pred[2] > pred[3]:
      action[pygame.K_LEFT] = True
    elif pred[3] > pred[2]:
      action[pygame.K_RIGHT] = True
  
  for key, (dx, dy) in MOVEMENT.items():
    if action[key]:
      agent.move_ip(dx, dy)

def move_regression(agent: pygame.Rect, target: pygame.Rect, model: AgentNetwork_Regression) -> None:
  state = agent.x, agent.y, target.x, target.y
  state = torch.tensor(state, dtype=torch.float32)
  action = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT:  False }
  print(f"{state = }")
  
  model.eval()
  with torch.no_grad():
    pred = model(state) 
    print(f"{pred = }")
    print(f"{target = }")
    print(f"{agent = }")
    
    ## map into key pairs
    dx, dy = pred[0], pred[1]
    
    thresh = 0.3## threshold for the movement, 0.5 works well for 1k, 0.25 for 500, 0.05 for 250 otherwise they can get stuck
    # old key sytem:
    if dx > thresh: 
      action[pygame.K_RIGHT] = True
    elif dx < -thresh:
      action[pygame.K_LEFT] = True
      
    if dy > thresh: 
      action[pygame.K_UP] = True
    elif dy < -thresh:
      action[pygame.K_DOWN] = True
    
    for key, (dx, dy) in MOVEMENT.items():
      if action[key]:
        agent.move_ip(dx, dy)

def create_image_data(n: int, ssFolder: str) -> None:
  pygame.init()

  ## Setup Screen
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("2D Canvas")

  coords: dict[str, Movement] = {}
  clock = pygame.time.Clock()
  isRunning = True
  
  agent = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  target = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  
  targetCount = 0
  frameCount = 0
  while isRunning:
    ## White Background
    screen.fill(Color("white"))
     
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        isRunning = False
      if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        print(f"Agent(x={agent.x}, y={agent.y}) Target(x={target.x}, y={target.y})")
        
    
    ## Game Logic
    ## draw the squares, agend is BLUE, target is RED
    pygame.draw.rect(screen, Color("blue"), agent)
    pygame.draw.rect(screen, Color("red"), target)
    
    imageName = f"ss-{targetCount}-{frameCount}.png"
    action: Action2 = auto_policy(agent, target)
    coords[imageName] = {
      "agent_x": agent.x,
      "agent_y": agent.y,
      "target_x": target.x,
      "target_y": target.y,
      "action": action
    }
    pygame.image.save(screen, f"{ssFolder}/{imageName}")
    
    if agent.colliderect(target):
      targetCount += 1
      frameCount += 1
      target.x = random.randint(0, X_BOUND)
      target.y = random.randint(0, Y_BOUND)
      # agent.x = random.randint(0, X_BOUND)
      # agent.y = random.randint(0, Y_BOUND)
    
    if targetCount >= n:
      isRunning = False
      
    frameCount += 1
    pygame.display.update()
    clock.tick(FPS)
    # Old Action4 version
    # coords[imageName] = {
    #   "agent_x": agent.x,
    #   "agent_y": agent.y,
    #   "target_x": target.x,
    #   "target_y": target.y,
    #   "movement": (movement[pygame.K_UP], movement[pygame.K_DOWN], movement[pygame.K_LEFT], movement[pygame.K_RIGHT])
    # }
    # pygame.time.wait(1000)
  
  df = pd.DataFrame.from_dict(coords, orient="index", columns=[
    "agent_x", "agent_y", "target_x", "target_y", "action"])
  print(df)
  df.to_pickle(f"{ssFolder}/ss-info.pkl")
  pygame.quit()
  
    
def move_cnn(agent: pygame.Rect, target: pygame.Rect, image: Image, model: CNN_Regression ) -> None:
  model.eval()
  with torch.no_grad():
    ## moved the image tranformation to the model, I think this makes the most sense, coupling these things
    x = model.transform_image(image)
    x = x.unsqueeze(0) ## add the batch dimension to make [1,3,600,800], otherwise model complains
    pred = model(x)[0]
  print(f"agent: ({agent.x}, {agent.y}) target: ({target.x}, {target.y})")
  print(f"{pred = }")
  
  print(f"pred: dx: {pred[0]} | dy: {pred[1]}")
  action = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT:  False }
  ## map into key pairs
  dx, dy = pred[0], pred[1]
  
  thresh = 0.00## threshold for the movement, 0.5 works well for 1k, 0.25 for 500, 0.05 for 250 otherwise they can get stuck
  agent.move_ip(int(dx * MOV_SPEED), int(dy * MOV_SPEED))
  # # old key sytem:
  # if dx > thresh: 
  #   action[pygame.K_RIGHT] = True
  # elif dx < -thresh:
  #   action[pygame.K_LEFT] = True
    
  # if dy > thresh: 
  #   action[pygame.K_UP] = True
  # elif dy < -thresh:
  #   action[pygame.K_DOWN] = True
  
  
  
  # for key, (dx, dy) in MOVEMENT.items():
  #   if action[key]:
  #     agent.move_ip(dx, dy)
      
  
T = TypeVar("T")
def load_model(loadModelFromFile: str, 
               modelType: Type[T] = None,
               device: Literal["cpu", "cuda"] = "cpu"
               ) -> None:
  
  if modelType is None:
    raise ValueError("Must provide 'modelType' when loading model from file")
    
  print(f"Loading type {modelType}")
    
  with open(loadModelFromFile, "rb") as f:
    model = modelType() ## initialise model class before loading weights
    model.load_state_dict(torch.load(f, map_location=torch.device(device)))
    return model
      
MODEL_TYPES = {
  "auto_policy": lambda s: None,
  "move_regression": lambda s: load_model(s, modelType=AgentNetwork_Regression),
  "move_classification": lambda s: load_model(s, modelType=AgentNetwork_Classification),
  "move_cnn": lambda s: load_model(s, modelType=CNN_Regression),
  "move_arrowkeys": lambda s: None
}
      
MOVES= ["auto_policy", "move_regression", "move_classification", "move_arrowkeys", "move_cnn"]
def play_game(
              moveAgent: Literal["auto_policy", "move_regression", "move_classification", "move_arrowkeys", "move_cnn"],
              loadModelFromFile: str = None, 
            ) -> None:
  if not loadModelFromFile:
    raise ValueError("Must provide 'loadModelFromFile'")
  pygame.init()
  clock = pygame.time.Clock()

  ## Setup Screen
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("2D Canvas")

  agent = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  target = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)

  isRunning = True
  
  model = MODEL_TYPES[moveAgent](loadModelFromFile)

  while isRunning:
    ## White Background
    screen.fill(Color("white"))
      
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        isRunning = False
          
    ## Game Logic
    ## draw the squares, agend is BLUE, target is RED
    pygame.draw.rect(screen, Color("blue"), agent)
    pygame.draw.rect(screen, Color("red"), target)
    
    # print(f"Real Agent pos: {agent.x, agent.y} Target pos: {target.x, target.y}")
    
    match moveAgent:
      case "auto_policy":
        auto_policy(agent, target)
      case "move_regression":
        move_regression(agent, target, model)
      case "move_classification":
        move_classification(agent, target, model)
      case "move_arrowkeys":
        move_arrowkeys(agent, target, model)
      case "move_cnn":
        image = Image.frombytes(mode="RGB", size=(WIDTH, HEIGHT), data=pygame.image.tobytes(screen, "RGB"))
        image.save("temp.png")
        move_cnn(agent, target, image, model)
    ## When collided restart the target, so the game continuosly runs
    if agent.colliderect(target):
      target.x = random.randint(0, X_BOUND)
      target.y = random.randint(0, Y_BOUND)
    
    pygame.display.update()
    clock.tick(FPS)

  pygame.quit()


import torch

if __name__ == "__main__":
  print(f"'canvas' [main]")
  # create_image_data(10, "./datasets/screenshots-10")
  # print(f"Up -> {pygame.K_UP}")
  # print(f"DOWN -> {pygame.K_DOWN}")
  # print(f"LEFT -> {pygame.K_LEFT}")
  # print(f"RIGHT -> {pygame.K_RIGHT}")
  
  # with open("agent-network-1k.pth", "rb") as f:
  #   model = AgentNetwork(4, 4)
  #   model.load_state_dict(torch.load(f))
  #   model.eval() ## set to evaluation mode as the training is complete
  #   play_game(model)
  learn_game("1k-targets.pkl", auto_policy, n=1000)

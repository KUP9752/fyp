import pygame
from pygame.color import Color

import pickle
import random
import time
from typing import Type, Callable, TypeVar

from my_types import State, Action4, Movement

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
def auto_policy(state: State) -> Movement:
  agent_x, agent_y,target_x, target_y = state
  
  movement = {
    pygame.K_UP: False,
    pygame.K_DOWN: False,
    pygame.K_LEFT: False,
    pygame.K_RIGHT: False
  }
  
  ## To remove the jitter adjust the boundary of the condition
  match agent_x:
    case _ if agent_x >= target_x and agent_x < target_x + BLOCK_SIZE:
      movement[pygame.K_LEFT] = False
      movement[pygame.K_RIGHT] = False
    case _ if agent_x >= target_x + BLOCK_SIZE:
      movement[pygame.K_LEFT] = True
    case _ if agent_x < target_x:
      movement[pygame.K_RIGHT] = True
      
  match agent_y:
    case _ if agent_y >= target_y and agent_y < target_y + BLOCK_SIZE:
      movement[pygame.K_UP] = False
      movement[pygame.K_DOWN] = False
    case _ if agent_y >= target_y + BLOCK_SIZE:
      movement[pygame.K_UP] = True
    case _ if agent_y < target_y:
      movement[pygame.K_DOWN] = True
      
  
    
  return movement

def human_interaction(state: State) -> Movement:
  return pygame.key.get_pressed()

## File to save the demonstration data to train on
def learn_game(filepath: str = None, agentMovement: Callable[[State], Movement] = auto_policy, n = 10) -> list[tuple[State, Action4]]:

  pygame.init()
  clock = pygame.time.Clock()

  ## Setup Screen
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("2D Canvas")

  agent = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  target = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)


  isRunning = True
  
  data: list[tuple[State, Action4]]= []
  
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
    action = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT:  False }
    
    keys = agentMovement(state)
    
    for key, (dx, dy) in MOVEMENT.items():
      if keys[key]:
        action[key] = True
        agent.move_ip(dx, dy) # this is a move 'in-place', doesn't alter the object
    
    
    ## When collided restart the target, so the game continuosly runs
    if agent.colliderect(target):
      targetCount += 1
      target.x = random.randint(0, X_BOUND)
      target.y = random.randint(0, Y_BOUND)
    
      
    ## save the data from this frame  
    data.append((state, tuple(action.values())))
    
    ## Finish learning when n targets are reached
    if targetCount == n:
      isRunning = False
    
    pygame.display.update()
    clock.tick(60)

  pygame.quit()
  
  ## Once the Game Ends save the data
  elapsedTime = time.perf_counter() - startTime
  print(f"Elapsed Time: {elapsedTime} for {n} targets -> (state, action) datapoints")
  print(f"Writing to file {filepath}")
  
  if filepath:
    with open(f"./datasets/{filepath}", "wb") as f:
      pickle.dump(data, f)
      
  return data
  
from torch_bc import AgentNetwork, AgentNetwork_Classification, AgentNetwork_Regression
from torch import nn

T = TypeVar("T")

def move_arrowkeys(agent: pygame.Rect, target: pygame.Rect, model: AgentNetwork) -> None:
  keys = pygame.key.get_pressed()
  
  for key, (dx, dy) in MOVEMENT.items():
    if keys[key]:
      agent.move_ip(dx, dy)
  
  
  
def move_classification(agent: pygame.Rect, target: pygame.Rect, model: AgentNetwork_Classification) -> None:
  state = agent.x, agent.y, target.x, target.y
  state = torch.tensor(state, dtype=torch.float32)
  action = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT:  False }
  
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
  
  with torch.no_grad():
    pred = model(state) 
    print(f"{pred = }")
    print(f"{target = }")
    print(f"{agent = }")
    
    ## map into key pairs
    dx, dy = pred[0], pred[1]
    
    thresh = 0.05## threshold for the movement, 0.5 works well for 1k, 0.25 for 500, 0.05 for 250 otherwise they can get stuck
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

def play_game(
              move_agent: Callable[[pygame.Rect, pygame.Rect, AgentNetwork], None],
              model: AgentNetwork = None, 
              loadModelFromFile: str = None, 
              modelType: Type[T] = None, 
            ) -> None:
  if not model and not loadModelFromFile:
    raise ValueError("Must provide either 'model' or 'loadModelFromFile', loadModelFromFile takes priority if provided !!modelType must be also provided!!")
  
  if loadModelFromFile:
    if modelType is None:
      raise ValueError("Must provide 'modelType' when loading model from file")
    
    with open(loadModelFromFile, "rb") as f:
      model = modelType() ## initialise model class before loading weights
      model.load_state_dict(torch.load(f))
  
  pygame.init()
  clock = pygame.time.Clock()

  ## Setup Screen
  WIDTH, HEIGHT = 800, 600
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("2D Canvas")

  agent = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  target = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)

  isRunning = True

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
    
    move_agent(agent, target, model)
    
    ## When collided restart the target, so the game continuosly runs
    if agent.colliderect(target):
      target.x = random.randint(0, X_BOUND)
      target.y = random.randint(0, Y_BOUND)
    
    pygame.display.update()
    clock.tick(60)

  pygame.quit()


import torch

if __name__ == "__main__":
  print(f"'canvas' [main]")
  # with open("agent-network-1k.pth", "rb") as f:
  #   model = AgentNetwork(4, 4)
  #   model.load_state_dict(torch.load(f))
  #   model.eval() ## set to evaluation mode as the training is complete
  #   play_game(model)
  # learn_game("10-targets.pkl", human_interaction, n=10)

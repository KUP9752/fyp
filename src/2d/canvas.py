import pygame
from pygame.color import Color

import pickle
import random
import time
from typing import Callable, type

type State = tuple[int, int, int, int]
type Action = tuple[bool, bool, bool, bool]
type Movement = dict[pygame.key, bool]

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
def learn_game(filepath: str, movBehaviour: Callable[[State], Movement],n = 10000) -> None:

  pygame.init()
  clock = pygame.time.Clock()

  ## Setup Screen
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("2D Canvas")

  agent = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  target = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)


  isRunning = True
  
  data: list[tuple[State, Action]]= []
  
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
    
    keys = movBehaviour((agent.x, agent.y), (target.x, target.y))
    
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
  
  with open(filepath, "wb") as f:
    pickle.dump(data, f)
  
from torch_bc import AgentNetwork

def play_game(model: AgentNetwork) -> None:
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
    
    state = agent.x, agent.y, target.x, target.y
    state = torch.tensor(state, dtype=torch.float32)
    action = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT:  False }
    
    
    # print(f"{pygame.K_UP = }")
    # print(f"{pygame.K_DOWN = }")
    # print(f"{pygame.K_LEFT = }")
    # print(f"{pygame.K_RIGHT = }")
    
    with torch.no_grad():
      pred = model(state) ## currently returns 4 floats, do some post processing
      ## if below a certain threshold reject it
      print(f"{pred = }")
      print(f"{target = }")
      print(f"{agent = }")
      
      
      ## Thresholding didn't seem to be necessary with BCEWithLogitsLoss
      # pred = torch.where(pred > 0.4, pred, torch.tensor(0.0))
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
      
      # if pred[0] <= 0:
      #   action[pygame.K_UP] = False
      # if pred[1] <= 0:
      #   action[pygame.K_DOWN] = False
      # if pred[2] <= 0:
      #   action[pygame.K_LEFT] = False
      # if pred[3] == 0:
      #   action[pygame.K_RIGHT] = False
      
      
      # print(f"{pred = }")
      print(f"{action = }")
      # raise Exception("Done")
    
    
    # print(f"{action = }")
    
        
    
    
    
    for key, (dx, dy) in MOVEMENT.items():
      if action[key]:
        agent.move_ip(dx, dy)
    
    ## When collided restart the target, so the game continuosly runs
    if agent.colliderect(target):
      target.x = random.randint(0, X_BOUND)
      target.y = random.randint(0, Y_BOUND)
    
    pygame.display.update()
    clock.tick(60)

  pygame.quit()


import torch

if __name__ == "__main__":
  # with open("agent-network-1k.pth", "rb") as f:
  #   model = AgentNetwork(4, 4)
  #   model.load_state_dict(torch.load(f))
  #   model.eval() ## set to evaluation mode as the training is complete
  #   play_game(model)
  

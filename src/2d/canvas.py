import pygame
from pygame.color import Color
from PIL import Image

import heapq
import pandas as pd
import pickle
import random
import time
from typing import Literal, Type, Callable, TypeVar

from my_types import State, Action4, Movement, Action2


SPEED = 2.0
FPS = 60
MOV_SPEED = 5
BLOCK_SIZE = 50 # per side

OBS_MAX_SIZE = 500 # per side
OBS_MIN_SIZE = 50 # per side

WIDTH, HEIGHT = 800, 600
X_BOUND = WIDTH - BLOCK_SIZE
Y_BOUND = HEIGHT - BLOCK_SIZE

MOVEMENT: dict = {
    pygame.K_UP: (0, -MOV_SPEED),
    pygame.K_DOWN: (0, MOV_SPEED),
    pygame.K_LEFT: (-MOV_SPEED, 0),
    pygame.K_RIGHT: (MOV_SPEED, 0)
  }


## Clamped Movement
def move_ip_clamped(rect: pygame.Rect, dx: int, dy: int) -> None:
  rect.move_ip(dx, dy)
  rect.clamp_ip(0, 0, WIDTH, HEIGHT)


## Non-in-place movement
def move_obs(rect: pygame.Rect, dx: int, dy: int, obstacles: list[pygame.Rect]) -> pygame.Rect:
  clone = rect.move(dx, dy)
  
  collides = clone.collidelistall(obstacles)
  for i in collides:
    obs = obstacles[i]
    
    if dx > 0:
      if rect.right <= obs.left:
        clone.right = obs.left
    elif dx < 0:
      if rect.left >= obs.right:
        clone.left = obs.right
    if dy > 0:
      ## strict top
      if rect.bottom <= obs.top:
        clone.bottom = obs.top
    elif dy < 0:
      ## strict bottom
      if rect.top >= obs.bottom:
        clone.top = obs.bottom
          
  return clone.clamp(0, 0, WIDTH, HEIGHT)
## Restrictive movement with obstaclles
def move_ip_obs(rect: pygame.Rect, dx: int, dy: int, obstacles: list[pygame.Rect]) -> None:
  clone = rect.move(dx, dy)
  
  collides = clone.collidelistall(obstacles)
  for i in collides:
    obs = obstacles[i]
    
    if dx > 0:
      if rect.right <= obs.left:
        clone.right = obs.left
    elif dx < 0:
      if rect.left >= obs.right:
        clone.left = obs.right
    if dy > 0:
      ## strict top
      if rect.bottom <= obs.top:
        clone.bottom = obs.top
    elif dy < 0:
      ## strict bottom
      if rect.top >= obs.bottom:
        clone.top = obs.bottom
          
  clone.clamp_ip(0, 0, WIDTH, HEIGHT)
  rect.x, rect.y = clone.x, clone.y

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
  
  # move_ip_clamped(agent, dx, dy)
  
  return dx, dy


## With obstacles
def auto_policy_obstacles(agent: pygame.Rect, target: pygame.Rect, obstacles: list[pygame.Rect]) -> Action2:
  ## Manhattan heuristic
  def heuristic(robot: tuple[int, int], goal: tuple[int, int], penalty: int = 0) -> int:
    from math import sqrt
    return sqrt((robot[0] - goal[0])**2 + (robot[1] - goal[1])**2) + penalty
    
    
  stepSize = MOV_SPEED
  
  directions = [
    (0, stepSize), (stepSize, 0), (0, -stepSize), (-stepSize, 0),
    (stepSize, stepSize), (stepSize, -stepSize), (-stepSize, stepSize), (-stepSize, -stepSize)
  ] 
  
  open_set = []
  start = agent.x, agent.y
  goal = target.x, target.y
  heapq.heappush(open_set, (0, start))
  
  ## (x, y) -> (x, y), (dx, dy) coordinate we came from + the step that got us here
  cameFrom: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
  g_score = {start: 0}
  f_score = {start: heuristic(start, goal)}
  
  clone = agent.copy() ## essentially a copy
  
  while open_set:
    _, current = heapq.heappop(open_set)
    clone.x, clone.y = current
    if clone.colliderect(target):
      # Reconstruct the path
      path = []
      while current in cameFrom:
        x, y = current
        current = cameFrom[current]
        path.append((x - current[0], y - current[1]))  # Add the step that got us here
      # path.append(start)
      return path[::-1] if path else [(0,0)]  # Reverse the path, if no path then we must already be on the target
    
    isValidMove = False
    for dx, dy in directions:
      neighbour = clone.move(dx, dy)
      n_coords = neighbour.x, neighbour.y
      # Check if neighbor is within bounds
      if neighbour.collidelistall(obstacles):
        
        continue
      
      tentative_g_score = g_score[current] + 1  # Cost of moving to a neighbor
      if n_coords not in g_score or tentative_g_score < g_score[n_coords]:
        # Update scores and add to the open set
        cameFrom[n_coords] = current
        g_score[n_coords] = tentative_g_score
        f_score[n_coords] = tentative_g_score + heuristic(n_coords, goal)
        heapq.heappush(open_set, (f_score[n_coords], n_coords))
        
  raise ValueError("No path found, but a path exists!")



def generate_obstacles(agent: pygame.Rect, target: pygame.Rect) -> list[pygame.Rect]:
  gridSize = BLOCK_SIZE * 2 ## NOTE: could be changed to change the finness of the grid and the obstacles
  cols, rows = WIDTH // gridSize, HEIGHT // gridSize
  
  pathGrid = [[0 for _ in range(cols)] for _ in range(rows)] 
  
  start = (agent.x // gridSize, agent.y // gridSize)
  goal = (target.x // gridSize, target.y // gridSize)
  
  curr = start
  
  while curr != goal:
    x, y = curr
    pathGrid[y][x] = 1
    weights = [0.1, 0.1, 0.1, 0.1] ##
    gx, gy = goal
    
    dx, dy = gx - x, gy - y## negative dx goal is left, positive dy goal is above
    
    if dx > 0:
      weights[3] = abs(dx) / (abs(dx) + abs(dy))
    if dx < 0:
      weights[2] = abs(dx) / (abs(dx) + abs(dy))
    if dy > 0:
      weights[1] = abs(dy) / (abs(dx) + abs(dy))
    if dy < 0:
      weights[0] = abs(dy) / (abs(dx) + abs(dy))
      
    curr = random.choices([(x, max(y - 1, 0)), (x, min(y + 1, rows - 1)), (max(x - 1, 0), y), (min(x + 1, cols - 1), y)], weights=weights, k=1)[0]
    
  obstacles = []
  print(f"Traced Path:")
  from pprint import pprint
  pprint(pathGrid)
  for i in range(rows):
    for j in range(cols):
      if pathGrid[i][j] == 0:
        prob = 50
        ## the chosen path is just around this block
        if pathGrid[i][min(j + 1, cols - 1)] or pathGrid[i][max(j - 1, 0)] or pathGrid[min(i + 1, rows - 1)][j] or pathGrid[max(i - 1, 0)][j]:
          prob  = 70
        
        if prob >= random.randint(0, 100):
          obs = pygame.Rect(j * gridSize, i * gridSize, gridSize, gridSize)
          if obs.collidelistall([agent, target]):
            continue
          obstacles.append(obs)
  return obstacles
  

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
    screen .fill(Color("white"))
      
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


## Movement helper to map back to buttons
def _move_buttons(action: dict[pygame.key, bool]) -> Action2:
  dx, dy = 0, 0
  for key, (cx, cy) in MOVEMENT.items():
    if action[key]:
      dx += cx
      dy += cy
      
  return dx, dy


def move_arrowkeys(agent: pygame.Rect, target: pygame.Rect) -> Action2:
  keys = pygame.key.get_pressed()
  
  return _move_buttons(keys)
  
  
def move_classification(agent: pygame.Rect, target: pygame.Rect, model: AgentNetwork_Classification) -> Action2:
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
  
  return _move_buttons(action)

def move_regression(agent: pygame.Rect, target: pygame.Rect, model: AgentNetwork_Regression) -> Action2:
  state = agent.x, agent.y, target.x, target.y
  state = torch.tensor(state, dtype=torch.float32)
  print(f"{state = }")
  
  model.eval()
  with torch.no_grad():
    pred = model(state) 
    print(f"{pred = }")
    print(f"{target = }")
    print(f"{agent = }")
    
    ## map into key pairs
    dx, dy = int(pred[0]), int(pred[1])
  return dy, dx
    
def create_random_loc_image_data(n: int, ssFolder: str) -> None:
  pygame.init()

  ## Setup Screen
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("2D Canvas")

  coords: dict[str, Movement] = {}
  clock = pygame.time.Clock()
  isRunning = True
  
  
  def get_agent_with_phase_bounds(phase: int, target: pygame.Rect) -> pygame.Rect:
    match close:
      ## 1. within 1 rect of target
      case 0:
        x_low, x_high = target.x - BLOCK_SIZE, target.x + 2 * BLOCK_SIZE ## 2 because count from top left
        y_low, y_high = target.y - BLOCK_SIZE, target.y + 2 * BLOCK_SIZE 
      ## 2. Close, within 3 rects of target
      case 1:
        x_low, x_high = target.x - 3 * BLOCK_SIZE, target.x + 4 * BLOCK_SIZE ## 4 because count from top left
        y_low, y_high = target.y - 3 * BLOCK_SIZE, target.y + 4 * BLOCK_SIZE
      ## 3. entire canvas  
      case 2:  
        x_low, x_high, y_low, y_high = 0, X_BOUND, 0, Y_BOUND
        
    ## But also respect the bounds of the canvas
    x_low = max(x_low, 0)
    x_high = min(x_high, X_BOUND)
    y_low = max(y_low, 0)
    y_high = min(y_high, Y_BOUND)
    return pygame.Rect(random.randint(x_low, x_high), random.randint(y_low, y_high), BLOCK_SIZE, BLOCK_SIZE)
        
  for close in range(0, 3):
    for i in range(n): ## make n points for each closeness phase
      ## White Background
      screen.fill(Color("white"))
      target = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
      agent = get_agent_with_phase_bounds(close, target)
      
      #  Not getting overlapping images might be hurting the model's ability to move when close to the target ,keep these in.
      # if agent.colliderect(target):
      #   i -= 1
      #   print(f"touching at {i}")
      #   continue
      
      ## Game Logic
      ## draw the squares, agend is BLUE, target is RED
      pygame.draw.rect(screen, Color("blue"), agent)
      pygame.draw.rect(screen, Color("red"), target)
      
      imageName = f"ss-{i}-close-{close}.png"
      action: Action2 = auto_policy(agent, target)
      coords[imageName] = {
        "agent_x": agent.x,
        "agent_y": agent.y,
        "target_x": target.x,
        "target_y": target.y,
        "action": action,
        "closeness": close
      }
      
      pygame.display.update()
      
      pygame.image.save(screen, f"{ssFolder}/{imageName}")
    
  df = pd.DataFrame.from_dict(coords, orient="index", columns=["agent_x", "agent_y", "target_x", "target_y", "action", "closeness"])
  print(df)
  df.to_pickle(f"{ssFolder}/ss-info.pkl")
  pygame.quit()
  
  
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
    
      
def move_cnn(agent: pygame.Rect, target: pygame.Rect, image: Image, model: CNN_Regression ) -> Action2:
  model.eval()
  with torch.no_grad():
    ## moved the image tranformation to the model, I think this makes the most sense, coupling these things
    x = model.transform_image(image)
    x = x.unsqueeze(0) ## add the batch dimension to make [1,3,600,800], otherwise model complains
    pred = model(x)[0]
  print(f"agent: ({agent.x}, {agent.y}) target: ({target.x}, {target.y})")
  dx, dy = pred[0], pred[1]
  print(f"pred: dx: {dx} | dy: {dy}")
  # map into key pairs
  dx, dy = int(pred[0] * 10), int(pred[1] * 10)
  return dx, dy


def move_cnn_buttons(agent: pygame.Rect, target: pygame.Rect, image: Image, model: CNN_Regression ) -> Action2:
  model.eval()
  with torch.no_grad():
    ## moved the image tranformation to the model, I think this makes the most sense, coupling these things
    x = model.transform_image(image)
    x = x.unsqueeze(0) ## add the batch dimension to make [1,3,600,800], otherwise model complains
    pred = model(x)[0]
  print(f"agent: ({agent.x}, {agent.y}) target: ({target.x}, {target.y})")
  dx, dy = pred[0], pred[1]
  print(f"pred: dx: {dx} | dy: {dy}")
  
  # map into key pairs
  action = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT:  False }
  threshold = 0
  
  dx = dx if abs(dx) > threshold else 0
  dy = dy if abs(dy) > threshold else 0
  
  if dx > 0:
    action[pygame.K_RIGHT] = True
  elif dx < 0:
    action[pygame.K_LEFT] = True
    
  if dy > 0:
    action[pygame.K_DOWN] = True 
  elif dy < 0:
    action[pygame.K_UP] = True
  
  return _move_buttons(action)  
  
  
  
# Works slightly different than play game, so delegate here when playing the game with "auto_policy_obs"
def _auto_obs_game() -> None:
  pygame.init()
  clock = pygame.time.Clock()
  
   ## Setup Screen
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("2D Canvas")
  bg = pygame.Surface((WIDTH, HEIGHT))
  bg.fill(Color("white"))
  
  agent = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  target = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  obstacles = generate_obstacles(agent, target)
  # obs = obstacles[0]
  # obs.unionall_ip(obstacles[1:])
  
  isRunning = True
  
  while isRunning:
    # for obs in obstacles:
    #   pygame.draw.rect(screen, Color("black"), obs)
    # pygame.draw.rect(screen, Color("blue"), agent)
    # pygame.draw.rect(screen, Color("red"), target)
    # pygame.display.update()
    
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        isRunning = False
        
    screen.blit(bg, (0, 0))
    for obs in obstacles:
      pygame.draw.rect(screen, Color("black"), obs)
    pygame.draw.rect(screen, Color("blue"), agent)
    pygame.draw.rect(screen, Color("red"), target)
    
    dx, dy = auto_policy_obstacles(agent, target, obstacles)[0]
    move_ip_obs(agent, dx, dy, obstacles)
    # for dx, dy in auto_policy_obstacles(agent, target, obstacles):
    #   move_ip_obs(agent, dx, dy, obstacles)
    #   pygame.display.update()
    #   clock.tick(60)
      
    if agent.colliderect(target):
      target.x = random.randint(0, X_BOUND)
      target.y = random.randint(0, Y_BOUND)
      obstacles = generate_obstacles(agent, target)
      
    pygame.display.update()
    clock.tick(60)
        
    
  pygame.quit()
  
  
T = TypeVar("T")
def load_model(loadModelFromFile: str, 
               modelType: Type[T] = None,
               device: Literal["cpu", "cuda"] = None
               ) -> None:
  
  if modelType is None:
    raise ValueError("Must provide 'modelType' when loading model from file")
    
  print(f"Loading type {modelType}")
  
  if not device:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
  with open(loadModelFromFile, "rb") as f:
    model = modelType() ## initialise model class before loading weights
    model.load_state_dict(torch.load(f, map_location=torch.device(device)))
    return model

MODEL_TYPES = {
  "auto_policy": lambda s: None,
  "auto_policy_obs": lambda s: None,
  "move_regression": lambda s: load_model(s, modelType=AgentNetwork_Regression),
  "move_classification": lambda s: load_model(s, modelType=AgentNetwork_Classification),
  "move_cnn": lambda s: load_model(s, modelType=CNN_Regression),
  "move_cnn_buttons": lambda s: load_model(s, modelType=CNN_Regression),
  "move_arrowkeys": lambda s: None
}
      
MOVES= ["auto_policy", "auto_policy_obs", "move_regression", "move_classification", "move_arrowkeys", "move_cnn", "move_cnn_buttons"]
def play_game(
              moveAgent: Literal["auto_policy","auto_policy_obs", "move_regression", "move_classification", "move_arrowkeys", "move_cnn"],
              loadModelFromFile: str = None, 
            ) -> None:
  if moveAgent == "auto_policy_obs":
    return _auto_obs_game()
  if not loadModelFromFile:
    raise ValueError("Must provide 'loadModelFromFile'")
  
  
  pygame.init()
  clock = pygame.time.Clock()

  ## Setup Screen
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  bg = pygame.Surface((WIDTH, HEIGHT))
  bg.fill(Color("white"))
  pygame.display.set_caption("2D Canvas")
  
  agent = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  target = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  obstacles = generate_obstacles(agent, target)
  
  isRunning = True
  
  model = MODEL_TYPES[moveAgent](loadModelFromFile)

  while isRunning:
    ## White Background and Display
    screen.blit(bg, (0, 0))
    for obs in obstacles:
      pygame.draw.rect(screen, Color("black"), obs)
    pygame.draw.rect(screen, Color("blue"), agent)
    pygame.draw.rect(screen, Color("red"), target)
      
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        isRunning = False
          
    ## Game Logic
    ## draw the squares, agend is BLUE, target is RED
    # print(f"Real Agent pos: {agent.x, agent.y} Target pos: {target.x, target.y}")
      
    match moveAgent:
      case "auto_policy":
        dx, dy = auto_policy(agent, target)
      ## Slighlty special case, stop the frame by frame movement and move the agent until meeting target
      # case "auto_policy_obs":
      case "move_regression":
        dx, dy = move_regression(agent, target, model)
      case "move_classification":
        dx, dy = move_classification(agent, target, model)
      case "move_arrowkeys":
        dx, dy = move_arrowkeys(agent, target)
      case "move_cnn":
        image = Image.frombytes(mode="RGB", size=(WIDTH, HEIGHT), data=pygame.image.tobytes(screen, "RGB"))
        # pygame.image.save(screen, "temp.png")
        # image = Image.open("temp.png")
        # image.save("temp.png")
        dx, dy = move_cnn(agent, target, image, model)
      case "move_cnn_buttons":
        image = Image.frombytes(mode="RGB", size=(WIDTH, HEIGHT), data=pygame.image.tobytes(screen, "RGB"))
        dx, dy = move_cnn_buttons(agent, target, image, model)
    
    move_ip_obs(agent, dx, dy, obstacles)
    
    
    ## When collided restart the target, so the game continuosly runs
    if agent.colliderect(target):
      target.x = random.randint(0, X_BOUND)
      target.y = random.randint(0, Y_BOUND)
      obstacles = generate_obstacles(agent, target)
    
    pygame.display.update()
    clock.tick(FPS)
    # isRunning = False
    # pygame.image.save(screen, "temp.png")
    
    # pygame.time.wait(100000)   

  pygame.quit()


import torch

if __name__ == "__main__":
  print(f"'canvas' [main]")
  # create_image_data(100, "./datasets/screenshots-big-100")
  # create_random_loc_image_data(10_000, "./datasets/screenshots-rand-10k-withcols-closeness")
  # print(f"Up -> {pygame.K_UP}")
  # print(f"DOWN -> {pygame.K_DOWN}")
  # print(f"LEFT -> {pygame.K_LEFT}")
  # print(f"RIGHT -> {pygame.K_RIGHT}")
  
  # with open("agent-network-1k.pth", "rb") as f:
  #   model = AgentNetwork(4, 4)
  #   model.load_state_dict(torch.load(f))
  #   model.eval() ## set to evaluation mode as the training is complete
  #   play_game(model)
  # learn_game("1k-targets.pkl", auto_policy, n=1000)

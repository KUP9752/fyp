import pygame
from pygame.color import Color

import pickle
import random
import time

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


def auto_policy(state: tuple[int, int], target: tuple[int, int]) -> dict[pygame.key, bool]:
  agent_x, agent_y = state
  target_x, target_y = target
  
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

## File to save the demonstration data to train on
def learn_game(filepath: str, n = 10000) -> None:

  pygame.init()
  clock = pygame.time.Clock()

  ## Setup Screen
  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  pygame.display.set_caption("2D Canvas")

  agent = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)
  target = pygame.Rect(random.randint(0, X_BOUND), random.randint(0, Y_BOUND), BLOCK_SIZE, BLOCK_SIZE)


  isRunning = True
  
  data = []
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
    
    keys = auto_policy((agent.x, agent.y), (target.x, target.y))
    # keys = pygame.key.get_pressed() ## teaching by human demonstration
    
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
  
  
  
  
  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Train a model on teh demonstration data
def fit_model(filepath: str) -> RandomForestClassifier:
  
  with open(filepath, "rb") as f:
    data = pickle.load(f)
  
  
  print(f"{data = }")
  
  X, y = zip(*data) # states, (demonstrated) actions

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12) ## random state shuffles the data

  model = RandomForestClassifier()
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  
  print(accuracy_score(y_test, y_pred))
  return model
  

def play_game(model: RandomForestClassifier) -> None:
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
    
    pred = model.predict([state])[0]
    
    
    action = {key: movement for key, movement in zip(MOVEMENT.keys(), pred)}
    print(f"{action = }")
    
    
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


learn_game("10k-points.pkl", n = 10000)
# model = fit_model("demonstration.pkl")
# play_game(model)



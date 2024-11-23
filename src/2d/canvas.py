import pygame
from pygame.color import Color

import random

pygame.init()
## Setup Screen
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Canvas")



MOV_SPEED = 5

BLOCK_SIZE = 20 # per side

xBound = WIDTH - BLOCK_SIZE
yBound = HEIGHT - BLOCK_SIZE

agent = pygame.Rect(random.randint(0, xBound), random.randint(0, yBound), BLOCK_SIZE, BLOCK_SIZE)
target = pygame.Rect(random.randint(0, xBound), random.randint(0, yBound), BLOCK_SIZE, BLOCK_SIZE)

isRunning = True

movement: dict = {
  pygame.K_UP: (0, -MOV_SPEED),
  pygame.K_DOWN: (0, MOV_SPEED),
  pygame.K_LEFT: (-MOV_SPEED, 0),
  pygame.K_RIGHT: (MOV_SPEED, 0)
}


while isRunning:
  ## White Background
  screen.fill(Color("white"))
    
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      isRunning = False
  
  ## Game Logic
  pygame.draw.rect(screen, Color("blue"), agent)
  pygame.draw.rect(screen, Color("red"), target)
  
  
  keys = pygame.key.get_pressed()
  for key, (dx, dy) in movement.items():
    if keys[key]:
      print(f"Moving {dx}, {dy}")
      # agent.move(dx, dy)
  #     agent.move_ip(dx, dy) # this is a move 'in-place', doesn't alter the object
  
  
  
  if agent.colliderect(target):
    print("Collision!")
  
  ## draw the squares, agend is BLUE, target is RED
  

  
  
  pygame.display.update()
  


import pygame
from pygame.color import Color

pygame.init()
## Setup Screen
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Canvas")


BLOCK_SIZE = 20 # per side

agent = pygame.Rect(100, 100, BLOCK_SIZE, BLOCK_SIZE)
target = pygame.Rect(500, 100, BLOCK_SIZE, BLOCK_SIZE)

isRunning = True

while isRunning:
    screen.fill(Color("white"))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False

    pygame.draw.rect(screen, Color("blue"), agent)
    pygame.draw.rect(screen, Color("red"), target)

    pygame.display.update()
  


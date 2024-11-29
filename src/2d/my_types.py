import pygame

type State = tuple[int, int, int, int]

## ([-1, 1], [-1, 1]) for delta_y and delta_x
type Action2 = tuple[int, int]
## Up, Down, Left, Right
type Action4 = tuple[bool, bool, bool, bool]

type Movement = dict[pygame.key, bool]

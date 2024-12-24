import pygame

# agent_x, agent_y, target_x, target_y
type State = tuple[int, int, int, int]

## ([-1, 1], [-1, 1]) for delta_x and delta_y
type Action2 = tuple[int, int]
## Up, Down, Left, Right
type Action4 = tuple[bool, bool, bool, bool]

type Movement = tuple[float, float]

import pygame

type State = tuple[int, int, int, int]
type Action = tuple[bool, bool, bool, bool]
type Movement = dict[pygame.key, bool]

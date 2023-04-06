from .block import Block, BlockType

import numpy as np
from enum import Enum

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    UP = 4
    DOWN = 5

    def vector(self) -> np.ndarray:
        if self == Direction.NORTH:
            return np.array([0, 1, 0])
        elif self == Direction.EAST:
            return np.array([1, 0, 0])
        elif self == Direction.SOUTH:
            return np.array([0, -1, 0])
        elif self == Direction.WEST:
            return np.array([-1, 0, 0])
        elif self == Direction.UP:
            return np.array([0, 0, 1])
        elif self == Direction.DOWN:
            return np.array([0, 0, -1])

class World:
    DIM = 15
    ORIGIN = np.full(3, DIM // 2)

    def __init__(self):
        init_array = np.full((World.DIM, World.DIM, World.DIM), None)
        self.vGrid = np.vectorize(Block)

        self.blocks = self.vGrid(init_array)

    def shift(self, robotDir: Direction):
        # perform the shift
        self.blocks = np.roll(self.blocks, shift=-robotDir.vector(), axis=(0, 1, 2))

        # fill in the rolled blocks
        init_array = np.full((World.DIM, World.DIM), None)
        newBlocks = self.vGrid(init_array)

        def map_func(x):
            if x == 0:
                return slice(0, World.DIM)
            elif x == -1:
                return slice(0, 1)
            elif x == 1:
                return slice(World.DIM-1, World.DIM)
        
        idx = robotDir.vector()
        m = tuple(map(map_func, idx))
        view = np.squeeze(self.blocks[m])
        view[:, :] = newBlocks
    
    def in_world(pos):
       return np.min(pos) >= 0 and np.max(pos) < World.DIM
    
    def copy(self):
        newWorld = World()
        for i in range(newWorld.blocks.shape[0]):
            for j in range(newWorld.blocks.shape[1]):
                for k in range(newWorld.blocks.shape[2]):
                    newWorld.blocks[i,j,k] = self.blocks[i,j,k].copy()
        
        return newWorld
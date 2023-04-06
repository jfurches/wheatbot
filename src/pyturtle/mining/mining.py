from ..base import TurtleState
from enum import Enum
import numpy as np

class BlockType(Enum):
    # None represents an unknown block
    AIR = 0
    BLOCK = 1
    ORE = 2
    COAL = 3

class Block:
    ORE_REWARD = 5

    def __init__(self, type):
        self.type = type
    
    def is_solid(self):
        return self.type and self.type in [BlockType.BLOCK, BlockType.ORE, BlockType.COAL]
    
    def is_known(self):
        return self.type is not None
    
    def get_mining_reward(self):
        if self.type == BlockType.ORE:
            return Block.ORE_REWARD
        
        return 0
        
    def copy(self):
        return Block(self.type)


class MiningState(TurtleState):
    pass
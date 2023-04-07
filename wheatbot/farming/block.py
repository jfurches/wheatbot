from enum import Enum
from functools import cache
import numpy as np

class BlockType(Enum):
    '''Enum of the different block types we care about for this task
    
    They are mapped to the hex color which represents them in world image
    files
    '''
    AIR = ""
    WHEAT = "#FFC90E"
    WATER = "#00A2E8"
    DIRT = "#B97A57"
    GRASS = "#22B14C"
    FARMLAND = "#FF7F27"
    CHEST = "#FFDF8A"
    COMPOSTER = "#8B8F4C"
    FENCE = "#663310"

    @cache
    @staticmethod
    def from_color(color: str) -> "BlockType":
        for bt in BlockType.__members__.values():
            if color.upper() == bt.value.upper():
                return bt
        
        return None
    
    def color(self) -> str:
        return self.value

class Block:
    "Represents a block"

    solid_types = [BlockType.DIRT, BlockType.GRASS, BlockType.FARMLAND,
                   BlockType.CHEST, BlockType.COMPOSTER, BlockType.FENCE]
    def __init__(self, type: BlockType):
        self.type = type
    
    def is_solid(self) -> bool:
        return self.type in self.solid_types
    
    def tick(self):
        pass

    def copy(self):
        return Block(self.type)

class Wheat(Block):
    MAX_AGE = 7
    MIN_LIGHT = 9

    def __init__(self, age: int=0):
        super().__init__(BlockType.WHEAT)

        self.age = age
    
    def tick(self, light):
        if self.is_mature() or light < Wheat.MIN_LIGHT:
            return
        
        prob = 1/(np.floor(25/10) + 1)
        if np.random.random() < prob:
            self.age += 1
    
    def get_seeds(self) -> int:
        if self.is_mature():
            return 1
        
        else:
            seeds = [0, 1, 2, 3]
            p = [0.0787, 0.3149, 0.4198, 0.1866]

            return np.random.choice(seeds, p=p)
    
    def is_mature(self) -> bool:
        return self.age >= Wheat.MAX_AGE
    
    def copy(self):
        return Wheat(self.age)
    
class Composter(Block):
    FULL = 7
    SEED_PROB = 0.3

    def __init__(self, level: int=0):
        super().__init__(BlockType.COMPOSTER)
    
        self.level = level
    
    def add_seeds(self, n: int):
        "Adds seeds to the composter"

        to_add = np.random.binomial(n, Composter.SEED_PROB)
        self.level = np.min([Composter.FULL, self.level + to_add])
    
    def is_full(self):
        return self.level >= Composter.FULL
    
    def empty(self):
        "Empties the composter (which would give +1 bone meal)"
        self.level = 0
    
    def copy(self):
        return Composter(self.level)
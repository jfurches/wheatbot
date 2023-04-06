import numpy as np
from PIL import Image
import itertools
import logging

from .block import Block, BlockType, Composter, Wheat

class Sky:
    data = np.array([
        # start, level
        [0, 15],
        [12041, 14],
        [12210, 13],
        [12377, 12],
        [12542, 11],
        [12705, 10],
        [12867, 9],
        [13027, 8],
        [13188, 7],
        [13348, 6],
        [13509, 5],
        [13670, 4],
        [22331, 5],
        [22492, 6],
        [22653, 7],
        [22813, 8],
        [22974, 9],
        [23134, 10],
        [23297, 11],
        [23460, 12],
        [23624, 13],
        [23791, 14],
        [23961, 15],
        [24000, 15]
    ])

    @staticmethod
    def get_light(time: int):
        t = time % 24000
        mask = (t >= Sky.data[:-1, 0]) & (t < Sky.data[1:, 0])
        idx = np.flatnonzero(mask)[0]
        retval = Sky.data[idx, 1]
        return retval

class FarmingWorld:
    randomTickSpeed = 3
    spawnColor = "#A349A4"

    def __init__(self, blocks: np.ndarray, special_blocks: dict, time=0):
        self.blocks = blocks
        self.time = time

        self.spawn_points = special_blocks.get('spawn_points', [])
        self.composters = special_blocks.get('composters', [])
        self.chests = special_blocks.get('chests', [])

        n_wheat = 0
        mean_wheat_pos = np.zeros(3)
        for (x, y, z) in itertools.product(*list(map(range, blocks.shape))):
            block = blocks[x, y, z]
            if isinstance(block, Wheat):
                mean_wheat_pos = (np.array([x, y, z]) + n_wheat * mean_wheat_pos) / (n_wheat + 1)
                n_wheat += 1
        
        self.mean_wheat_pos = mean_wheat_pos
    
    def light_level(self):
        return Sky.get_light(self.time)
    
    def tick(self, n=1):
        for _ in range(n):
            self.time += 1
            light = self.light_level()

            # do random chunk ticks, see https://minecraft.fandom.com/wiki/Tick#Random_tick
            for chunk in self.subchunks():
                vol = chunk.size / 16**3
                for _ in range(FarmingWorld.randomTickSpeed):
                    if np.random.rand() < vol:
                        block_idx = np.random.randint(0, chunk.size)
                        block = chunk.flat[block_idx]
                        if isinstance(block, Wheat):
                            block.tick(light)
                        elif block is not None:    
                            block.tick()

    def subchunks(self):
        nx, ny, nz = map(int, np.ceil(np.array(self.blocks.shape) / 16))
        for x, y, z in itertools.product(range(nx), range(ny), range(nz)):
            view = self.blocks[
                16*x:np.min([self.blocks.shape[0], 16*(x+1)]),
                16*y:np.min([self.blocks.shape[1], 16*(y+1)]),
                16*z:np.min([self.blocks.shape[2], 16*(z+1)]),
            ]

            yield view
        
    def in_world(self, pos: np.ndarray) -> bool:
        return (pos < np.array(self.blocks.shape)).all() and (pos >= 0).all()
    
    def block(self, pos: np.ndarray) -> Block:
        if not self.in_world(pos):
            return None
        
        return self.blocks[pos[0], pos[1], pos[2]]
    
    def nearest_chest(self, pos: np.ndarray) -> np.ndarray:
        return min(self.chests, key=lambda x: np.linalg.norm(x - pos, ord=1))
    
    def nearest_composter(self, pos: np.ndarray) -> np.ndarray:
        return min(self.composters, key=lambda x: np.linalg.norm(x - pos, ord=1))
    
    @classmethod
    def from_image(cls, img: Image, wheat_age=0):
        "Convenience method to generate farm worlds from pixel images"

        def rgb2hex(r, g, b) -> str:
            return '#{:02x}{:02x}{:02x}'.format(r, g, b).upper()

        width, height = img.size
        pixels = img.convert('RGBA').load()

        n_layers = width // height
        dx, dy, dz = height + width % height, height, n_layers

        logging.debug(f'World size: {dx}, {dy}, {dz}')

        special_blocks = {'composters': [], 'spawn_points': [], 'chests': []}

        grid = np.full((dx, dy, dz), None, dtype=Block)
        for x, y, z in itertools.product(range(dx), range(dy), range(dz)):
            r, g, b, _ = pixels[x + dy*z, y]
            color = rgb2hex(r, g, b).upper()

            logging.debug(f'({x}, {y}, {z}) Color: {color}')

            if color == FarmingWorld.spawnColor:
                logging.debug('Spawn point')
                special_blocks['spawn_points'].append(np.array([x, y, z]))
                continue

            bt = BlockType.from_color(color)
            logging.debug(f'Block Type: {bt}')

            if bt is BlockType.WHEAT:
                # by explicitly setting 'wheat age' to None, randomly initialize the age
                if wheat_age is None:
                    age = np.random.randint(0, Wheat.MAX_AGE+1)
                    logging.debug(f"Random init wheat to {age}")
                else:
                    age = wheat_age

                block = Wheat(age)
            elif bt is BlockType.COMPOSTER:
                block = Composter()
                special_blocks['composters'].append(np.array([x, y, z]))
            else:
                block = Block(bt)
                
                if bt is BlockType.CHEST:
                    special_blocks['chests'].append(np.array([x, y, z]))
            
            grid[x, y, z] = block
        
        return cls(grid, special_blocks)
    
    def copy(self):
        special_blocks = {
            'spawn_points': self.spawn_points,
            'composters': self.composters,
            'chests': self.chests
        }

        return FarmingWorld(self.blocks.copy(), special_blocks, self.time)
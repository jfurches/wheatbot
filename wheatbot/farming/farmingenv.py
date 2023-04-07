from typing import Optional, Mapping, Any, Tuple, Dict
import os
import logging
import itertools
from glob import glob

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
import numpy as np
from scipy.stats import norm
import pygame
from PIL import Image

from ..common.inventory import Inventory, ItemStack
from .world import FarmingWorld, Wheat, BlockType, Block

WORLDS_DIR = os.path.join(os.path.dirname(__file__), 'worlds')
RESOURCE_FOLDER = os.path.join(os.path.dirname(__file__), 'resources')

DEFAULT_CONFIG = {
    # Generic attributes
    'gamma': 0.99,
    'world': 'farm2.png',
    'wheat_age': 0,
    'fuel': 80,
    'max_timesteps': 1000,

    # Rewards
    'timestep_reward': 0,
    'field_reward': 0.1,
    'field_reward_radius': 2,
    'harvest_reward': 1,
    'chest_reward': 10,
    'scale_chest_reward': 0.1,
    'chest_dist_penalty': 1,

    # Potential-based reward shaping
    'chest_pbrs_type': 'r',
    'chest_pbrs_strength': 1,
    'chest_pbrs_scale': 5,
    'scale_chest_pbrs': False,

    'field_pbrs_type': 'r',
    'field_pbrs_strength': 1,
    'field_pbrs_scale': 5
}

ObsType = Dict[str, Any]
ActType = int
InfoType = Dict[str, Any]

class FarmingEnv(gym.Env):
    '''The default environment where the agent must learn to
    collect wheat and bring it back to a chest for collection.
    
    The action space is a `Discrete` set of 6 actions, with some
    actions being invalid depending on conditions.

    The observation space is made up of world attributes, the time
    limit, fuel, as well as relative positions to key points and
    properties about the block it is looking at. Because we also
    return the action mask of valid actions, the observation looks like

    >>> Dict({
    ...    'action_mask': Box(0, 1, n_actions),
    ...    'observations': Dict(...)
    ... })

    Config Params:
        gamma: The agent's gamma value, used for reward shaping
        world: The image file corresponding to a world to load.
            Defaults to `world2.png`
        wheat_age: The initial age of the wheat from 0 (new) to
            7 (harvestable). Pass `None` for randomly generated wheat.
        fuel: The amount of fuel the agent has. In computercraft,
            only moving forward consumes fuel. Default 80
        max_timesteps: The maximum timesteps before the environment is
            halted. Default 1000
        
        timestep_reward: A reward to give on each timestep. Default 0
        field_reward: Reward for the first time the agent gets within
            `field_reward_radius` blocks of the center of the wheat field.
            Default 0.1
        field_reward_radius: The radius for the field reward to activate.
            Default 2.
        harvest_reward: Reward for harvesting mature (age=7) wheat.
            Default 1
        chest_reward: Reward for bringing back any wheat to the chest.
            Default 10
        scale_chest_reward: An additional bonus for the agent that scales
            the amount of wheat it returned. Default 0.1
        chest_dist_penalty: A penalty for the agent that scales the number
            of blocks it finishes an episode from the chest. Default 1
        
        chest_pbrs_type: Type of potential function to use. Default `r`
        chest_pbrs_strength: Constant multiplier of the PBRS. Default 1
        chest_pbrs_scale: Sigma value if `gaussian` PBRS is used. Default 5
        scale_chest_pbrs: If true, the chest PBRS is multiplied by the amount
            of currently held wheat. Default false

        field_pbrs_type: Same as above in `chest_pbrs_type`
        field_pbrs_strength: Same as above in `chest_pbrs_strength`
        field_pbrs_scale: Same as above in `chest_pbrs_scale`
    '''

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 2
    }

    actions = ['no-op', 'move-forward', 'turn-left', 
               'turn-right', 'harvest', 'interact-chest']
    actions_idx = {a: i for i, a in enumerate(actions)}
    action_space = Discrete(len(actions))

    block_idxs = [None] + list(BlockType.__members__.values())


    def __init__(self, env_config: Optional[Mapping[str, Any]], *,
                 render_mode: Optional[str] = None):
        '''Create the environment from the given configuration'''
        self.config = DEFAULT_CONFIG | (env_config or {})

        self.gamma = self.config['gamma']

        self.real_obs_space = spaces.Dict({
            'timesteps_remaining': Box(low=0, high=1),
            'world_time': Box(low=0, high=1),
            'light_level': Box(low=0, high=1),
            'fuel': Box(low=0, high=self.config['fuel']),
            'wheat': Box(low=0, high=np.inf),
            # 'position': Box(low=-np.inf, high=np.inf, shape=(3,)),
            'direction': Box(low=-1, high=1, shape=(3,)),
            'chest_displacement': Box(low=-np.inf, high=np.inf, shape=(3,)),
            'field_displacement': Box(low=-np.inf, high=np.inf, shape=(3,)),
            'facing': Box(low=0, high=1, shape=(len(self.block_idxs),)),
            'wheat_age': Box(low=0, high=1)
        })

        self.observation_space = spaces.Dict({
            'action_mask': Box(low=0, high=1, shape=(self.action_space.n,), dtype=int),
            'observations': self.real_obs_space
        })

        # the turtle's attributes
        self._pos: np.ndarray
        self._dir: np.ndarray
        self._fuel: int
        self._inventory: Inventory

        # world
        self._world: FarmingWorld
        self._timesteps: int

        self._reached_field = False

        # scorekeeping
        self._scores: dict
        self._last_obs: dict
        self._last_full_obs: dict

        # rendering
        self.render_mode = render_mode
        self.window = None
        self.window_size: tuple
        self.block_size = 8
        
        # resources for rendering
        self.textures: Dict[str, pygame.Surface] = None
        self.fonts: Dict[str, pygame.font.Font] = None

        # Clock is the pygame clock, bg_canvas is a rectangle
        # that will contain all the elements of the scene that
        # don't change, so we can save ourselves a significant
        # amount of rendering
        self.clock: pygame.time.Clock = None
        self.bg_canvas: pygame.Surface = None
        self.sprite_groups: Dict[str, pygame.sprite.Group | pygame.sprite.Sprite] = None

    def reset(self, seed=None, options=None) -> Tuple[ObsType, InfoType]:
        '''Resets the environment'''
        super().reset(seed=seed, options=options)

        # Reload the world
        # todo: support arbitrary image URLs
        img = Image.open(os.path.join(WORLDS_DIR, self.config['world']))
        self._world = FarmingWorld.from_image(
            img,
            wheat_age=self.config['wheat_age']
        )

        self._world_size = np.array(self._world.blocks.shape)

        self._timesteps = self.config['max_timesteps']
        self._reached_field = False

        # Reset the turtle position, fuel, and inventory
        pos_idx = self.np_random.choice(len(self._world.spawn_points))
        self._pos = self._world.spawn_points[pos_idx].copy()
        self._dir = np.array([1, 0, 0], dtype=int)

        self._fuel = self.config['fuel']
        self._inventory = Inventory()

        # Reset the scores
        self._scores = {
            'wheat_harvested': 0,
            'wheat_collected': 0
        }

        # Rendering
        self.window_size = (self._get_frame_width(), self._get_frame_height())

        obs, info = self._get_obs()
        self._last_obs = obs

        self._render_frame()

        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, InfoType]:
        '''Advances the world one timestep, which may correspond to
        several minecraft ticks'''

        looking_at = self._world.block(self._pos + self._dir) or Block(BlockType.AIR)
        action = FarmingEnv.actions[action]
        reward = self.config['timestep_reward']

        info = {
            'chest_reward': 0,
            'harvest_reward': 0,
            'field_reward': 0
        }

        old_pos = self._pos.copy()

        if action == 'no-op':
            pass

        elif action == 'move-forward':
            if not looking_at.is_solid():
                self._pos += self._dir
                self._fuel -= 1

                if not self._reached_field and \
                    np.linalg.norm(self._pos - self._world.mean_wheat_pos, ord=1) <= self.config['field_reward_radius']:
                    self._reached_field = True
                    field_reward = self.config['field_reward']
                    info['field_reward'] = field_reward
                    reward += field_reward

        elif action == 'turn-left':
            m = np.array(
                [[0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]], dtype=int)
            
            self._dir = np.dot(m, self._dir)
        
        elif action == 'turn-right':
            m = np.array(
                [[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]], dtype=int)
            
            self._dir = np.dot(m, self._dir)
        
        elif action == 'harvest':
            harvest_reward_amount = self.config['harvest_reward']

            if isinstance(looking_at, Wheat):
                # Reward the bot and add to inventory
                if looking_at.age == Wheat.MAX_AGE:
                    harvest_reward = harvest_reward_amount
                    self._inventory.add_item(ItemStack('minecraft:wheat', 1))
                    self._scores['wheat_harvested'] += 1
                else:
                    harvest_reward = -harvest_reward_amount

                # Reset the wheat block
                looking_at.age = 0
                reward += harvest_reward
                info['harvest_reward'] = harvest_reward
            
            # else no-op

        elif action == 'interact-chest':
            if looking_at.type == BlockType.CHEST:
                # Remove the wheat from inventory
                idx, item = self._inventory.find_item('minecraft:wheat', return_idx=True)
                scale_reward = self.config['scale_chest_reward']

                if idx >= 0:
                    self._inventory.items[idx] = ItemStack.empty()
                    self._fuel += self.config['fuel']
                    self._scores['wheat_collected'] += item.amount

                    chest_reward = self.config['chest_reward']
                    if scale_reward:
                        chest_reward += scale_reward * item.amount
                else:
                    chest_reward = 0


                reward += chest_reward
                info['chest_reward'] = chest_reward

        reward += self._pbrs(old_pos)

        self._timesteps -= 1
        self._world.tick(10)
        obs, info = self._get_obs(info)
        self._last_obs = obs
        done = self._fuel == 0 or self._scores['wheat_collected'] > 0 or self._timesteps == 0

        if done:
            # Here we add an additional penalty of how far the bot is from the chest.
            # We won't normalize the distance since that could make the configuration
            # parameter unintuitive
            dist = np.linalg.norm(self._pos - self._world.nearest_chest(self._pos), ord=1) - 1
            dist = np.sqrt(dist)
            dist *= self.config['chest_dist_penalty']
            reward -= dist

        # done = self._fuel == 0 or self._timesteps == 0
        truncated = False

        self._render_frame()

        return obs, reward, done, truncated, info
    
    def _pbrs_func(self, x: np.ndarray, y: np.ndarray,
                   form='r', a=1, sigma=1):
        d = np.linalg.norm(x - y, ord=1)
        dx, dy = self._world.blocks.shape[:2]
        if form == '1/r':
            return a / d
        elif form == 'gaussian':
            return a * norm.pdf(d, scale=sigma)
        elif form == 'r':
            return -a * d / np.sqrt(dx ** 2 + dy ** 2)

    def _pbrs(self, old_pos: np.ndarray) -> float:
        '''Computes the potential-based reward shaping for the chest
        and the wheat field'''

        reward = 0

        # Reward shaping to encourage exploration towards the
        # chest when we have wheat
        #   PBRS: r = gamma * f(s') - f(s)
        def chest_f(x):
            form = self.config.get('chest_pbrs_type')
            a = self.config['chest_pbrs_strength']
            sigma = self.config['chest_pbrs_scale']
            chest = self._world.nearest_chest(x)

            return self._pbrs_func(x, chest, form=form, a=a, sigma=sigma)
    
        def field_f(x):
            form = self.config.get('field_pbrs_type')
            a = self.config['field_pbrs_strength']
            sigma = self.config['field_pbrs_scale']

            return self._pbrs_func(x, self._world.mean_wheat_pos, form=form, a=a, sigma=sigma)
        
        # Encourage going to the chest
        wheat = self._inventory.find_item('minecraft:wheat').amount
        chest_pbrs = self.gamma * chest_f(self._pos) - chest_f(old_pos)

        if self.config['scale_chest_pbrs']:
            # Goal gets more prominent the more wheat we hold. This
            # also means it won't be active if the agent doesn't
            # have any wheat
            chest_pbrs *= wheat

        reward += chest_pbrs
        self._scores['chest_pbrs'] = chest_pbrs
        
        # Encourage going towards the wheat field
        field_pbrs = self.gamma * field_f(self._pos) - field_f(old_pos)
        reward += field_pbrs
        self._scores['field_pbrs'] = field_pbrs
        
        return reward

    def _get_obs(self, info = None) -> Tuple[ObsType, InfoType]:
        info = self._scores.copy() | (info or {})

        fuel = self._fuel / self.config['fuel']
        wheat = self._inventory.find_item('minecraft:wheat').amount or 0

        info['world_time'] = self._world.time
        info['fuel'] = fuel
        info['wheat'] = wheat
        info['pos_x'] = self._pos[0]
        info['pos_y'] = self._pos[1]
        info['pos_z'] = self._pos[2]
        info['chest_dist'] = np.linalg.norm(self._pos - self._world.nearest_chest(self._pos), ord=1)
        info['field_dist'] = np.linalg.norm(self._pos - self._world.mean_wheat_pos, ord=1)

        # front block
        looking_at = self._world.block(self._pos + self._dir) or Block(BlockType.AIR)
        front_block_type = FarmingEnv.block_idxs.index(looking_at.type)
        fbt_onehot = np.zeros(len(FarmingEnv.block_idxs), dtype=int)
        fbt_onehot[front_block_type] = 1
        info['facing'] = str(looking_at.type)

        state = {
            'timesteps_remaining': np.array(self._timesteps) / self.config.get('max_timesteps'),
            'world_time': np.array(self._world.time % 24000) / 24000,
            'light_level': np.array(self._world.light_level()) / 15,
            'fuel': np.array(fuel) / self.config['fuel'],
            'wheat': np.array(wheat),
            # 'position': self._pos,
            'direction': self._dir,
            'chest_displacement': (np.array(self._world.nearest_chest(self._pos)) - self._pos) / self._world_size,
            'field_displacement': (self._world.mean_wheat_pos - self._pos) / self._world_size,
            'facing': fbt_onehot,
            'wheat_age': np.array(
                1.0 * looking_at.is_mature() if isinstance(looking_at, Wheat) \
                                             else 0
            )
        }

        obs = {
            'action_mask': self._get_action_mask(),
            'observations': state
        }

        return obs, info
    
    def _get_action_mask(self) -> np.ndarray:
        '''Returns a mask representing the valid actions'''

        mask = np.ones(self.action_space.n, dtype=int)
        looking_at = self._world.block(self._pos + self._dir) or Block(BlockType.AIR)

        if looking_at.is_solid():
            mask[FarmingEnv.actions_idx['move-forward']] = 0

        # if not isinstance(looking_at, Wheat) or not looking_at.is_mature():
        #     mask[FarmingEnv.actions_idx['harvest']] = 0
        
        if looking_at.type != BlockType.CHEST and self._inventory.find_item('minecraft:wheat').amount > 0:
            mask[FarmingEnv.actions.index('interact-chest')] = 0
        
        return mask

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def _render_check(self):
        if self.window is None:
            pygame.init()
            
            if self.render_mode == 'human':
                pygame.display.init()
                pygame.display.set_caption('Farming RL training')
                self.window = pygame.display.set_mode(self.window_size)
            else:
                pygame.display.init()
                self.window = pygame.display.set_mode(self.window_size, flags=pygame.HIDDEN)
        
        if self.textures is None:
            self.textures = self._load_textures(os.path.join(RESOURCE_FOLDER, 'textures'))
            self.fonts = self._load_fonts(os.path.join(RESOURCE_FOLDER, 'fonts'))

            WheatSprite.importTextures(self.textures)
            TurtleSprite.importTextures(self.textures)
            WheatItemSprite.importTextures(self.textures, self.fonts)
            SeedsItemSprite.importTextures(self.textures, self.fonts)

        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        if self.bg_canvas is None:
            self.bg_canvas = self._render_bg()
        
        if self.sprite_groups is None:
            self.sprite_groups = {
                'ui': self._get_ui(),
                'wheat': self._get_all_wheat(),
                'turtle': TurtleSprite(self._pos)
            }

    def _render_frame(self):
        if self.render_mode is None:
            return

        self._render_check()

        canvas = pygame.Surface(self.window_size)

        for sprite_group in self.sprite_groups.values():
            sprite_group.update(self._last_obs['observations'], self)

        canvas.blit(self.bg_canvas, (0, 0))
        for wheat in self.sprite_groups['wheat']:
            canvas.blit(wheat.surf, wheat.rect)
        
        turtle = self.sprite_groups['turtle']
        canvas.blit(turtle.surf, turtle.rect)

        for ui_el in self.sprite_groups['ui']:
            canvas.blit(ui_el.surf, ui_el.rect)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def _get_frame_width(self) -> int:
        return self._world.blocks.shape[0] * self.block_size
    
    def _get_frame_height(self) -> int:
        return self._world.blocks.shape[1] * self.block_size
    
    def _load_fonts(self, folder: str) -> Dict[str, pygame.font.Font]:
        font_map = {}

        logging.info(f'Looking for fonts in {folder}')

        for file in glob(f"{folder}/*.ttf"):
            font = pygame.font.Font(file, 12)
            name = os.path.splitext(os.path.basename(file))[0]

            logging.info(f'Loaded font {name}')

            font_map[name] = font
        
        return font_map
    
    def _load_textures(self, folder: str) -> Dict[str, pygame.Surface]:
        texture_map = {}

        logging.info(f'Looking for textures in {folder}')

        for file in glob(os.path.join(folder, '*.png')):
            texture = pygame.image.load(file)
            texture = pygame.transform.scale(texture, (self.block_size, self.block_size)).convert_alpha()
            name = os.path.splitext(os.path.basename(file))[0]

            logging.info(f'Loaded texture {name}')

            texture_map[name] = texture
        
        return texture_map
    
    def _get_texture(self, block: Block) -> pygame.Surface:
        '''Returns the texture for a given block'''
        if block is None or block.type is None:
            return None

        if isinstance(block, Wheat):
            type_map = f'wheat{block.age}'
        else:
            type_map = {
                BlockType.WATER: 'water',
                BlockType.CHEST: 'chest',
                BlockType.COMPOSTER: 'composter',
                BlockType.GRASS: 'grass',
                BlockType.FENCE: 'fence',
                BlockType.FARMLAND: 'farmland'
            }.get(block.type)

        if type_map is None:
            logging.error(f'BlockType {block.type} not in sprites!')

        return self.textures[type_map]

    def _render_bg(self):
        '''Renders all the static elements of the environment that don't change'''
        w, h = self._get_frame_width(), self._get_frame_height()
        surf = pygame.Surface((w, h))

        ignore = [BlockType.WHEAT]
        for (x, y, z) in itertools.product(*list(map(range, self._world.blocks.shape))):
            block = self._world.block(np.array([x, y, z]))

            if block is None or block.type in ignore:
                continue

            texture = self._get_texture(block)

            if texture:
                rect = texture.get_rect(x = x * self.block_size, y = y * self.block_size)
                surf.blit(texture, rect)

        return surf

    def _get_all_wheat(self) -> pygame.sprite.Group:
        '''Creates a sprite group of all wheat blocks'''
        sprites = pygame.sprite.Group()

        for (x, y, z) in itertools.product(*list(map(range, self._world.blocks.shape))):
            pos = np.array([x, y, z])
            block = self._world.block(pos)

            if isinstance(block, Wheat):
                wheat = WheatSprite(pos, block.age)
                sprites.add(wheat)

        return sprites
    
    def _get_ui(self) -> pygame.sprite.Group:
        '''Creates a sprite group of all UI elements'''
        sprites = pygame.sprite.Group()
        fuel = FuelSprite((2, 2), 250, 25)
        sprites.add(fuel)

        wheat = WheatItemSprite((2, 30))
        sprites.add(wheat)

        seeds = SeedsItemSprite((44, 30))
        sprites.add(seeds)

        return sprites


class WheatSprite(pygame.sprite.Sprite):
    '''UI sprite that renders held wheat'''
    textures = []

    def __init__(self, pos: np.ndarray, age=0):
        super(WheatSprite, self).__init__()

        self.pos = pos
        self.age = age

        self.surf = WheatSprite.textures[self.age]
        self.rect = self.surf.get_rect(
            x = self.pos[0] * self.surf.get_width(),
            y = self.pos[1] * self.surf.get_height()
        )
    
    def update(self, obs: ObsType, env: FarmingEnv):
        block = env._world.block(self.pos)
        if block and isinstance(block, Wheat):
            self.age = block.age
            self.surf = WheatSprite.textures[self.age]
    
    @classmethod
    def importTextures(cls, tmap: dict[str, pygame.Surface]):
        cls.textures = [tmap[f'wheat{i}'] for i in range(Wheat.MAX_AGE + 1)]
        logging.info(f'Imported {len(cls.textures)} wheat textures')

class TurtleSprite(pygame.sprite.Sprite):
    '''Sprite that represents the turtle/agent'''
    texture = None

    def __init__(self, pos: np.ndarray):
        super().__init__()

        self.pos = pos
        self.surf = TurtleSprite.texture.copy()
        self.__update_rect() 
    
    def update(self, obs: ObsType, env: FarmingEnv):
        self.pos = env._pos
        self.__update_rect()        

    def __update_rect(self):
        self.rect = self.surf.get_rect(
            x = self.pos[0] * self.surf.get_width(),
            y = self.pos[1] * self.surf.get_height()
        )

    @classmethod
    def importTextures(cls, tmap: dict[str, pygame.Surface]):
        cls.texture = tmap['turtle']

class FuelSprite(pygame.sprite.Sprite):
    '''UI bar showing remaining fuel'''
    def __init__(self, pos: Tuple[float, float], width, height):
        super().__init__()

        self.pos = pos
        self.dims = (width, height)
        self.fuel = 1

        self.rect = pygame.Rect(pos, self.dims)
        self.endRect = pygame.Rect((pos[0] + width - 1, pos[1]), (1, height))
        self.surf = pygame.Surface(self.dims)
        self.surf.set_colorkey((0, 0, 0))
        self.__update_surf()

    def update(self, obs: ObsType, env: FarmingEnv):
        self.fuel = env._fuel / env.config['fuel']
        self.rect.update(self.pos, (self.dims[0] * self.fuel, self.dims[1]))

        self.__update_surf()
    
    def __update_surf(self):
        self.surf.fill((0, 0, 0))
        fuelColor = (255, 0, 0) if self.fuel < 0.25 else (255, 255, 255)
        self.surf.fill(fuelColor, self.rect)
        self.surf.fill((255, 255, 255), self.endRect)

class WheatItemSprite(pygame.sprite.Sprite):
    itemSize = (40, 40)

    def __init__(self, pos: Tuple[float, float]):
        super().__init__()

        self.amount = 0
        self.surf = pygame.Surface(WheatItemSprite.itemSize).convert_alpha()
        self.rect = pygame.Rect(pos, WheatItemSprite.itemSize)
    
    def update(self, obs: dict, env: FarmingEnv):
        self.amount = obs['wheat']

        self.surf.fill((0, 0, 0, 50))
        self.surf.blit(WheatItemSprite.texture, (0, 0))
        counter = WheatItemSprite.font.render(str(self.amount), True, (255, 255, 255, 255))
        self.surf.blit(counter, (20, 25))

    @classmethod
    def importTextures(cls, tmap: dict[str, pygame.Surface], fmap: dict[str, pygame.font.Font]):
        cls.texture = pygame.transform.scale(tmap['wheatitem'], WheatItemSprite.itemSize)
        cls.font = fmap['notosansbold']

class SeedsItemSprite(pygame.sprite.Sprite):
    itemSize = (40, 40)

    def __init__(self, pos: Tuple[float, float]):
        super().__init__()

        self.amount = 0
        self.surf = pygame.Surface(SeedsItemSprite.itemSize).convert_alpha()
        self.rect = pygame.Rect(pos, SeedsItemSprite.itemSize)
    
    def update(self, obs: ObsType, env: FarmingEnv):
        # self.amount = state.seeds
        self.amount = 0

        self.surf.fill((0, 0, 0, 50))
        self.surf.blit(SeedsItemSprite.texture, (0, 0))
        counter = SeedsItemSprite.font.render(str(self.amount), True, (255, 255, 255, 255))
        self.surf.blit(counter, (20, 25))

    @classmethod
    def importTextures(cls, tmap: dict[str, pygame.Surface], fmap: dict[str, pygame.font.Font]):
        cls.texture = pygame.transform.scale(tmap['seeds'], SeedsItemSprite.itemSize)
        cls.font = fmap['notosansbold']
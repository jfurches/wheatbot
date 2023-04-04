import gymnasium as gym

from .block import *
from .farmingenv import FarmingEnv
from .hierarchical_farmingenv import HierarchicalFarmingEnv
from .world import FarmingWorld

# gym.register(
#     id='FarmingEnv-v0',
#     entry_point=__package__ + ':FarmingEnv',
#     max_episode_steps=1000
# )

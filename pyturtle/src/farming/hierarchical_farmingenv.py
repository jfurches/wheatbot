from typing import Dict, Any
import logging

import numpy as np
from gymnasium import spaces

from ray.rllib.env import MultiAgentEnv

from .farmingenv import FarmingEnv

logger = logging.getLogger(__name__)

class HierarchicalFarmingEnv(MultiAgentEnv):
    tasks = ['goto-field', 'harvest-wheat', 'goto-chest']

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        self.config = env_config
        self._skip_env_checking = True
        self.flat_env = FarmingEnv(env_config)

        # placeholders
        self.current_goal: int
        self.steps_remaining_at_level: int
        self.num_high_level_steps: int
        self.low_level_agent_id: str
        self.accumulated_reward: float

        flat_obs_space = self.flat_env.observation_space
        self.low_level_observation_space = spaces.Dict({
            'action_mask': flat_obs_space['action_mask'],
            'observations': spaces.Dict({
                'obs': self.flat_env.real_obs_space,
                'goal': spaces.Discrete(len(self.tasks))
            })
        })
        self.low_level_action_space = self.flat_env.action_space

        self.high_level_observation_space = self.flat_env.real_obs_space # don't mask top level agent
        self.high_level_action_space = spaces.Discrete(len(self.tasks))

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        cur_obs, info = self.flat_env.reset(seed=seed, options=options)
        self.current_goal = None
        self.steps_remaining_at_level = None
        self.num_high_level_steps = 0
        self.accumulated_reward = 0

        self.low_level_agent_id = f'low_level_{self.num_high_level_steps}'
        return {
            'high_level_agent': cur_obs['observations']
        }, {'high_level_agent': info}

    def step(self, action_dict: Dict[str, int]):
        assert len(action_dict) == 1, 'Only 1 action allowed'
        if "high_level_agent" in action_dict:
            return self._high_level_step(action_dict["high_level_agent"])
        else:
            return self._low_level_step(list(action_dict.values())[0])

    def _high_level_step(self, action: int):
        if self.config.get('assign_subtasks', True):
            assert 0 <= action < len(self.tasks)

        logger.debug('High level agent chose goal %d', action)

        self.current_goal = action
        self.steps_remaining_at_level = self.config.get('low_level_steps', 20)
        self.num_high_level_steps += 1
        self.low_level_agent_id = f'low_level_{self.num_high_level_steps}'
        self.accumulated_reward = 0

        cur_obs = self.flat_env._last_full_obs
        obs = {self.low_level_agent_id: {
            'action_mask': cur_obs['action_mask'],
            'observations': {
                'obs': cur_obs['observations'],
                'goal': self.current_goal
            }
        }}
        reward = {self.low_level_agent_id: 0}
        done = truncated = {"__all__": False}
        return obs, reward, done, truncated, {}

    def _low_level_step(self, action: int):
        logger.debug('Low level agent chose action %d', action)
        self.steps_remaining_at_level -= 1

        # Step in the actual env
        f_obs, f_reward, f_terminated, f_truncated, info = self.flat_env.step(action)

        # Calculate low-level agent observation and reward
        obs = {self.low_level_agent_id: {
            'action_mask': self._adjust_action_mask(f_obs['action_mask']),
            'observations': {
                'obs': f_obs['observations'],
                'goal': self.current_goal
            }
        }}

        if self.config.get('assign_subtasks', True):
            # Go to field
            if self.current_goal == 0:
                low_level_reward = info['field_pbrs'] + info['field_reward']
                self.accumulated_reward += info['field_reward']
            # Harvest wheat
            elif self.current_goal == 1:
                low_level_reward = info['harvest_reward']
                self.accumulated_reward += info['harvest_reward']
            # Return to chest
            elif self.current_goal == 2:
                low_level_reward = info['chest_pbrs'] + info['chest_reward']
                self.accumulated_reward += info['chest_reward']
        else:
            low_level_reward = 0

        reward = {self.low_level_agent_id: low_level_reward}

        # Handle env termination & transitions back to higher level.
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        if f_terminated or f_truncated:
            terminated["__all__"] = f_terminated
            truncated["__all__"] = f_truncated
            logger.debug("High level final reward %f", f_reward)
            reward["high_level_agent"] = self.accumulated_reward
            obs["high_level_agent"] = f_obs['observations']
        elif self.steps_remaining_at_level == 0:
            terminated[self.low_level_agent_id] = True
            truncated[self.low_level_agent_id] = False
            reward["high_level_agent"] = self.accumulated_reward
            obs["high_level_agent"] = f_obs['observations']

        info = {
            '__common__': info
        }

        return obs, reward, terminated, truncated, info

    def _adjust_action_mask(self, mask: np.ndarray) -> np.ndarray:
        task = self.tasks[self.current_goal]
        if task == 'goto-field':
            mask[self.flat_env.actions_idx['harvest']] = 0
            mask[self.flat_env.actions_idx['interact-chest']] = 0
        elif task == 'harvest_wheat':
            mask[self.flat_env.actions_idx['interact-chest']] = 0
        elif task == 'goto-chest':
            mask[self.flat_env.actions_idx['harvest']] = 0
        
        return mask
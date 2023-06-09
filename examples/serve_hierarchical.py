import argparse
from typing import Dict, Any
from dataclasses import dataclass
import pprint

import numpy as np

from ray import serve, tune
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.policy.torch_policy import TorchPolicy
from starlette.requests import Request

from wheatbot.farming import HierarchicalFarmingEnv
from wheatbot.farming.farmingenv import ObsType, InfoType, ActType
from wheatbot.farming.world import Sky
from hierarchical_farming_ppo import ParametricDictFlattenWrapper as Flattener

class Registry:
    types: Dict[str, Any] = {}

    @classmethod
    def register(cls, type_: str):
        def inner_decorator(class_):
            cls.types[type_] = class_
            return class_
        return inner_decorator
    
    @classmethod
    def get(cls, type_: str):
        return cls.types[type_]

@Registry.register('reset')
@dataclass
class Reset:
    obs: ObsType
    info: InfoType

@Registry.register('step')
@dataclass
class Step:
    obs: ObsType
    reward: float
    done: bool
    truncated: bool
    info: InfoType

@serve.deployment
class ServeHierarchicalModel:
    def __init__(self, checkpoint_path: str, num_low_level_steps: int = 20):
        self.algo = Algorithm.from_checkpoint(checkpoint_path)

        self.num_low_level_steps = num_low_level_steps
        self.num_low_steps_remaining = 0
        self.goal = None

        cfg: AlgorithmConfig = self.algo.get_config()
        env = HierarchicalFarmingEnv(cfg.env_config)
        self.env = Flattener(env)

        self.max_fuel = cfg.env_config['fuel']

        cfg = self.algo.get_policy('low_level_policy').config
        self.num_transformers = cfg['model']['attention_num_transformer_units']
        self.memory_inference = cfg['model']['attention_memory_inference']
        self.attention_dim = cfg['model']['attention_dim']

        pprint.pprint(cfg['model'], width=1)

        self.state = None
    
    async def __call__(self, req: Request) -> Dict[str, ActType]:
        data: dict = await req.json()

        # Use the 'type' field of the request to dispatch the
        # method and destructure the data
        type_ = data.pop('type')
        data: Reset | Step = Registry.get(type_)(**data)

        data.obs = self._fix_obs(data.obs)

        if isinstance(data, Reset):
            action = self.reset(data)
        elif isinstance(data, Step):
            action = self.step(data)
        
        return {'action': action}
    
    def _dict_to_vec(self, d):
        # Flip for mc coordinates
        return np.array([d['x'], d['z'], d['y']])

    def _fix_obs(self, observation: dict) -> dict:
        print('Received obs')
        # pprint.pprint(observation, width=3)
        observation['action_mask'] = np.array(observation['action_mask'])
        obs = observation['observations']

        # Convert everything to numpy
        for k, v in obs.items():
            if k in ['field_displacement', 'chest_displacement', 'direction']:
                obs[k] = self._dict_to_vec(v)
            else:
                obs[k] = np.array(v)
        
        # Get the current light level since we don't have a light sensor. Normalize
        # the resulting value since the policy expects that
        obs['light_level'] = Sky.get_light(int(obs['world_time'])) / 15

        # Also normalize the world time
        obs['world_time'] = obs['world_time'] / 24000

        # Normalize the fuel level
        if obs['fuel'] == 'unlimited':
            obs['fuel'] = 1
        else:
            obs['fuel'] = min(1, obs['fuel'] / self.max_fuel)

        # We also need to normalize the displacements from the chest and field. Since
        # we can't just measure the world size with a turtle, we'll be lazy and try
        # assuming the env is 100x100. This should keep values well behaved
        obs['chest_displacement'] = 1e-2 * obs['chest_displacement']
        obs['field_displacement'] = 1e-2 * obs['field_displacement']

        return observation

    def reset(self, reset: Reset) -> ActType:
        self._reset_state()
        return self._get_action(reset.obs)
    
    def step(self, step: Step) -> ActType:
        return self._get_action(step.obs)
    
    def _reset_state(self):
        self.num_low_steps_remaining = 0
        self.state = [
            np.zeros([self.memory_inference, self.attention_dim], dtype=np.float32)
            for _ in range(self.num_transformers)
        ]

    def _get_action(self, obs: dict) -> ActType:
        if self.num_low_steps_remaining == 0:
            self.goal = self._high_level_action(obs)
            print('New goal:', self.goal)
        
        # Preprocess for low-level agent
        obs = {
            'low_level_agent': {
                'action_mask': obs['action_mask'],
                'observations': {
                    'goal': self.goal,
                    'obs': obs['observations']
                }
            }
        }
        obs = self.env.observation(obs)['low_level_agent']

        # print('Flattened obs:')
        # pprint.pprint(obs, width=3)

        # Keep track of state since our low level agent is attention based
        action, state_out, *_ = self.algo.compute_single_action(
            observation=obs,
            state=self.state,
            policy_id='low_level_policy'
        )

        self.state = [
            np.concatenate([self.state[i], [state_out[i]]], axis=0)[1:]
            for i in range(self.num_transformers)
        ]

        self.num_low_steps_remaining -= 1

        print('Predicted action ', action)

        return action
    
    def _high_level_action(self, obs: dict) -> int:
        # Ignore the action mask, and pass our observation
        # through the flattening preprocessor
        obs = {'high_level_agent': obs['observations']}
        obs = self.env.observation(obs)['high_level_agent']

        # We don't have to deal with rnn state since we used a 
        # feedforward high level agent
        action = self.algo.compute_single_action(
            observation=obs,
            policy_id='high_level_policy'
        )

        # Reset low level agent
        self._reset_state()

        return action

def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default='127.0.0.1')
    parser.add_argument('-p', '--port', type=int, default=8080)
    parser.add_argument('-c', '--checkpoint', type=str, required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_cli_args()

    def env_creator(config):
        env = HierarchicalFarmingEnv(config)
        env = Flattener(env)

        return env
    
    tune.register_env('FarmingEnv', env_creator)

    server = ServeHierarchicalModel.bind(args.checkpoint)
    handle = serve.run(server, host=args.address, port=args.port)

    while True:
        pass
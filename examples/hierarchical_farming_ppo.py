'''Training script to run PPO on the hierarchical environment

It shows how to setup the policies in RLLib.
'''

from collections import Counter
import argparse
import os
from typing import Dict, Any, Union

import numpy as np
import ray
from ray import tune, air
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env import MultiAgentEnv

import gymnasium as gym
from gymnasium import spaces

# pylint: disable=unused-import
from wheatbot.farming import FarmingEnv, HierarchicalFarmingEnv
from torch_action_mask_recurrent import (
    TorchAttnActionMaskModel as AttnModel,
    TorchLSTMActionMaskModel as LstmModel
)

class ParametricDictFlattenWrapper(gym.ObservationWrapper, MultiAgentEnv):
    '''Wrapper class that handles environments with an observation space of
    
    ```
    Dict({
        'high_level_agent': Dict(...),  # flattened
        'low_level_(\\d+)': Dict({
            'action_mask': Box(...),
            'observations': Dict({      # flattened
                'obs': Dict(...),
                'goal': Discrete(...)
            })
        })
    })
    ```

    We want it to become

    ```
    Dict({
        'high_level_agent': Box(...),   # flattened
        'low_level_(\\d+)': Dict({
            'action_mask': Box(...),
            'observations': Box(...)    # flattened
        })
    })
    ```

    '''
    def __init__(self, env: HierarchicalFarmingEnv):
        gym.ObservationWrapper.__init__(self, env)
        MultiAgentEnv.__init__(self)

        self.fenv: HierarchicalFarmingEnv = env.unwrapped
        assert isinstance(self.fenv, HierarchicalFarmingEnv)

        self.low_level_observation_space = spaces.Dict({
            'action_mask': self.fenv.low_level_observation_space['action_mask'],
            'observations': spaces.flatten_space(self.fenv.low_level_observation_space['observations'])
        })
        self.low_level_action_space = self.fenv.low_level_action_space

        self.high_level_observation_space = spaces.flatten_space(self.fenv.high_level_observation_space)
        self.high_level_action_space = self.fenv.high_level_action_space

    def observation(self, observation: Dict[str, Dict[str, Any]]):
        new_obs = {}

        for agent, obs in observation.items():
            if agent == 'high_level_agent':
                new_obs[agent] = spaces.flatten(
                    self.fenv.high_level_observation_space,
                    obs
                )
            else:
                new_obs[agent] = {
                    'action_mask': obs['action_mask'],
                    'observations': spaces.flatten(
                        self.fenv.low_level_observation_space['observations'],
                        obs['observations']
                    )
                }

        return new_obs

class Metrics(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies = None, episode: Union[Episode, EpisodeV2], env_index = None, **kwargs):
        for k, info in episode._last_infos.items():
            if k.startswith('low_level'):
                for k, v in info.items():
                    if k.startswith('pos_') or k.endswith('_pbrs') or k.endswith('_dist'):
                        episode.user_data.setdefault(k, []).append(v)

    def on_episode_end(self, *, worker, base_env, policies, episode: EpisodeV2, env_index, **kwargs):
        for k, info in episode._last_infos.items():
            if k.startswith('low_level'):
                episode.custom_metrics['wheat_collected'] = info['wheat_collected']
                episode.custom_metrics['wheat_harvested'] = info['wheat_harvested']

        for k in episode.user_data.keys():
            if k.startswith('pos_') or k.endswith('_pbrs') or k.endswith('_dist'):
                episode.custom_metrics[k] = np.mean(episode.user_data[k])

    def on_postprocess_trajectory(self, *,
                                  worker, episode: Episode, agent_id, policy_id, policies, 
                                  postprocessed_batch: SampleBatch, original_batches, **kwargs) -> None:
        
        actions = postprocessed_batch[SampleBatch.ACTIONS].tolist()
        if policy_id == 'low_level_policy':
            episode.hist_data['actions'] = actions
            counter = Counter(actions)

            for i, name in enumerate(FarmingEnv.actions):
                episode.custom_metrics[f'action_{name}_selected'] = counter.get(i, 0) / len(actions)
            
            chest_idx = FarmingEnv.actions.index('interact-chest')
            episode.custom_metrics['used_chest'] = 1.0 if chest_idx in actions else 0.0
        elif policy_id == 'high_level_policy':
            episode.hist_data['goals'] = actions
            counter = Counter(actions)

            for i, name in enumerate(HierarchicalFarmingEnv.tasks):
                episode.custom_metrics[f'goal_{name}_selected'] = counter.get(i, 0) / len(actions)

def get_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stop-iters', type=int, default=10000,
                        help='Number of iterations to train')
    parser.add_argument('--stop-timesteps', type=int, default=1000000,
                        help='Number of timesteps to train for')
    parser.add_argument('--stop-reward', type=float, default=1000,
                        help='Reward threshold for stopping')
    parser.add_argument('--num-cpus', type=int, default=0,
                        help='Number of CPUs to use')
    parser.add_argument('--local-mode', action='store_true',
                        help='Start ray in local mode')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    GAMMA = 0.99
    args = get_cli_args()

    def env_creator(config):
        env = HierarchicalFarmingEnv(config)
        env = ParametricDictFlattenWrapper(env)

        return env

    ray.init(num_cpus=args.num_cpus or None, num_gpus=1, local_mode=args.local_mode)
    tune.register_env('FarmingEnv', env_creator)

    chest_reward = 1
    harvest_reward = (1 - GAMMA) * chest_reward

    LOW_LEVEL_STEPS = 20

    ENV_CONFIG = {
        'gamma': GAMMA,
        'wheat_age': 7,
        'timestep_reward': 0,
        'fuel': 240,
        'max_timesteps': 240,

        'harvest_reward': harvest_reward,
        'field_pbrs_type': 'r', # [1/r, gaussian, r]
        'field_pbrs_strength': 1,
        'field_pbrs_scale': 0.1,

        'chest_reward': chest_reward + 0.1,
        'chest_dist_penalty': 0,
        'scale_chest_reward': 0.25,

        'chest_pbrs_type': 'r', # [1/r, gaussian, r]
        'chest_pbrs_strength': 0.1,
        'chest_pbrs_scale': 1,

        'low_level_steps': LOW_LEVEL_STEPS
    }

    farming_env = env_creator(ENV_CONFIG)

    def policy_mapping_fn(agent_id, *args, **kwargs):
        if agent_id.startswith("low_level_"):
            return "low_level_policy"
        else:
            return "high_level_policy"

    config = (
        APPOConfig()
            .framework('torch')
            .environment('FarmingEnv', env_config=ENV_CONFIG, disable_env_checking=True)
            .training(
                gamma=GAMMA,
                lr=1e-3,
                lr_schedule=[(0, 5e-3), (2_000_000, 1e-4)],
                clip_param=0.2,
                train_batch_size=2000
            )
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
            .rollouts(num_rollout_workers=args.num_cpus - 1)
            .callbacks(Metrics)
            .multi_agent(
                policies={
                    "high_level_policy": PolicySpec(
                        observation_space=farming_env.high_level_observation_space,
                        action_space=farming_env.high_level_action_space,
                        # config=APPOConfig.overrides(model={
                        #     'use_attention': True,
                        #     'fcnet_hiddens': [64],
                        #     'fcnet_activation': 'tanh',
                        #     'max_seq_len': 50,
                        #     'attention_num_transformer_units': 2,
                        #     'attention_memory_training': 50,
                        #     'attention_memory_inference': 50,
                        #     'attention_num_heads': 4,
                        #     'attention_head_dim': 64
                        # }),
                        # config=APPOConfig.overrides(model={
                            
                        # })
                    ),
                    "low_level_policy": PolicySpec(
                        observation_space=farming_env.low_level_observation_space,
                        action_space=farming_env.low_level_action_space,
                        config=APPOConfig.overrides(model={
                            'custom_model': AttnModel,
                            'fcnet_hiddens': [64],
                            'fcnet_activation': 'tanh',
                            'max_seq_len': LOW_LEVEL_STEPS,
                            # 'attention_use_n_prev_rewards': LOW_LEVEL_STEPS,
                            'attention_num_transformer_units': 2,
                            'attention_memory_training': LOW_LEVEL_STEPS,
                            'attention_memory_inference': LOW_LEVEL_STEPS,
                            'attention_num_heads': 4,
                            'attention_head_dim': 64
                        },
                            gamma=0.9),
                    ),
                },
                policy_mapping_fn=policy_mapping_fn,
            )
    )

    stop = {
        "timesteps_total": args.stop_timesteps,
        # "custom_metrics/used_chest_mean": 0.8,
        "custom_metrics/wheat_collected_mean": 10,
    }

    tuner = tune.Tuner(
        'APPO',
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            local_dir='~/ray_results/HierarchicalFarmingEnv',
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=2,
                checkpoint_frequency=1000,
                checkpoint_score_attribute='custom_metrics/wheat_collected_mean',
            )
        ),
    )
    tuner.fit()

    ray.shutdown()
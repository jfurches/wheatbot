'''Training script for the regular flat FarmingEnv'''

from collections import Counter
import argparse
import os
from typing import Dict, Any, Union
import pprint

import numpy as np
import ray
from ray import tune, air
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy.sample_batch import SampleBatch

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.normalize import NormalizeReward

# pylint: disable=unused-import
from wheatbot.farming import FarmingEnv
from torch_action_mask_recurrent import TorchLSTMActionMaskModel as LSTMModel, TorchAttnActionMaskModel as AttnModel

class ParametricDictFlattenWrapper(gym.ObservationWrapper):
    '''Wrapper class that handles environments with an observation space of
    
    ```
    Dict({
        'action_mask': Box(...),
        'observations': Dict(...)
    })
    ```
    '''
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            'action_mask': env.observation_space['action_mask'],
            'observations': spaces.flatten_space(self.env.observation_space['observations'])
        })

    def observation(self, obs: Dict[str, Any]):
        # pprint.pprint(obs['observations'], width=1)
        new_obs = {
            'action_mask': obs['action_mask'],
            'observations': spaces.flatten(self.env.observation_space['observations'], obs['observations'])
        }
        return new_obs

class Metrics(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies = None, episode: Union[Episode, EpisodeV2], env_index = None, **kwargs):
        info: Dict[str, Any] = episode._last_infos['agent0']
        for k, v in info.items():
            if k.startswith('pos_') or k.endswith('_pbrs') or k.endswith('_dist'):
                episode.user_data.setdefault(k, []).append(v)

    def on_episode_end(self, *, worker, base_env, policies, episode: EpisodeV2, env_index, **kwargs):
        info = episode._last_infos['agent0']
        episode.custom_metrics['wheat_collected'] = info['wheat_collected']
        episode.custom_metrics['wheat_harvested'] = info['wheat_harvested']

        for k in episode.user_data.keys():
            if k.startswith('pos_') or k.endswith('_pbrs') or k.endswith('_dist'):
                episode.custom_metrics[k] = np.mean(episode.user_data[k])

    def on_postprocess_trajectory(self, *,
                                  worker, episode: Episode, agent_id, policy_id, policies, 
                                  postprocessed_batch: SampleBatch, original_batches, **kwargs) -> None:
        
        actions = postprocessed_batch[SampleBatch.ACTIONS].tolist()
        episode.hist_data['actions'] = actions
        counter = Counter(actions)

        for i, name in enumerate(FarmingEnv.actions):
            episode.custom_metrics[f'action_{name}_selected'] = counter.get(i, 0) / len(actions)
        
        chest_idx = FarmingEnv.actions.index('interact-chest')
        episode.custom_metrics['used_chest'] = 1.0 if chest_idx in actions else 0.0

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
        env = FarmingEnv(config)
        env = ParametricDictFlattenWrapper(env)
        # env = TimeLimit(env, max_episode_steps=4000)
        # env = NormalizeReward(env, gamma=GAMMA)
        # env = RecordVideo(env, 'videos', episode_trigger=lambda x: x % 1000)

        return env

    ray.init(num_cpus=args.num_cpus or None, num_gpus=1, local_mode=args.local_mode)
    tune.register_env('FarmingEnv', env_creator)

    chest_reward = 1
    harvest_reward = (1 - GAMMA) * chest_reward

    config = (
        APPOConfig()
            .framework('torch')
            .environment('FarmingEnv', env_config={
                'gamma': GAMMA,
                'wheat_age': 7,
                'timestep_reward': 0,
                'fuel': 240,
                'max_timesteps': 240,

                'harvest_reward': harvest_reward,
                'field_pbrs_type': 'r', # [1/r, gaussian, r]
                'field_pbrs_strength': 0,
                'field_pbrs_scale': 1,

                'chest_reward': chest_reward + 0.1,
                'chest_dist_penalty': 0,
                'scale_chest_reward': 0,
                'chest_pbrs_type': 'r', # [1/r, gaussian, r]
                'chest_pbrs_strength': 0,
                'chest_pbrs_scale': 1,
            })
            .exploration(exploration_config={
                'type': 'EpsilonGreedy',
                'epsilon_timesteps': 100000
            })
            .training(
                gamma=GAMMA,
                lr=1e-3,
                kl_coeff=0.1,
                clip_param=0.2,
                model={
                    # 'custom_model': LSTMModel,
                    # 'fcnet_hiddens': [256, 256],
                    # 'fcnet_activation': 'tanh',
                    # 'lstm_cell_size': 64

                    'custom_model': AttnModel,
                    'fcnet_hiddens': [64],
                    'fcnet_activation': 'tanh',
                    'max_seq_len': 50,
                    'attention_use_n_prev_rewards': 50,
                    'attention_use_n_prev_actions': 50,
                    'attention_num_transformer_units': 5,
                    'attention_memory_training': 50,
                    'attention_memory_inference': 50,
                    'attention_num_heads': 6,
                    'attention_head_dim': 64
                },
                train_batch_size=2000
            )
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
            .rollouts(num_rollout_workers=args.num_cpus - 1)
            .callbacks(Metrics)
    )

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "custom_metrics/used_chest_mean": 0.8,
    }

    tuner = tune.Tuner(
        'APPO',
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=2
            )
        ),
    )
    tuner.fit()

    ray.shutdown()

# Todo: when we copy this for hierarchal RL, rename 'prbs' to 'pbrs', and
# update the custom metrics since each obs is now a dict with new keys, and the
# infos are also dicts of dicts
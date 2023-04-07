import numpy as np

from wheatbot.farming import FarmingEnv

def test_render():
    config = {
        'max_timesteps': 10,
        'wheat_age': None,
        'world': 'farm1.png'
    }
    env = FarmingEnv(config, render_mode='human')
    
    obs, _ = env.reset()
    done = False
    while not done:
        p = obs['action_mask']
        p[1] *= 5
        p = p.astype(float) / p.sum()
        action = np.random.choice(len(env.actions), p=p)
        obs, _, done, _, _ = env.step(int(action))
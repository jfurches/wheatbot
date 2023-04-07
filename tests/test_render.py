from wheatbot.farming import FarmingEnv

def test_render():
    config = {
        'max_timesteps': 10,
        'wheat_age': None
    }
    env = FarmingEnv(config, render_mode='human')
    
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(1)
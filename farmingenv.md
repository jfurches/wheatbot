# The Wheat Farming Environment

In this file, we'll break down the motivations and decisions behind the environment details to avoid cluttering the main README file.

## Goal

![Wheatbot environment](https://user-images.githubusercontent.com/38408451/229688875-696af56c-64de-41c8-8644-a0c79260e6ff.png)

As stated earlier, the objective of the agent is to harvest wheat and to bring it to the chest. An episode works as follows:

1. The agent spawns somewhere randomly
2. The agent can move around and harvest wheat as it pleases
3. The episode ends when (1) any wheat is returned to the chest, (2) the maximum time limit is reached, or (3) the agent runs out of fuel

## Observation Space

The observation space is a `Dict` with the following structure:

```python
Dict({
    'action_mask': Box(0, 1, n_actions),
    'observations': Dict(...)
})
```

`action_mask` is a binary mask with `1s` representing actions the agent can perform at the current timestep. `observations` is a `Dict` containing the actual observation, which looks like this (taken from `farmingenv.py`)

```python
Dict({
    'timesteps_remaining': Box(low=0, high=1),
    'world_time': Box(low=0, high=1),
    'light_level': Box(low=0, high=1),
    'fuel': Box(low=0, high=self.config['fuel']),
    'wheat': Box(low=0, high=np.inf),
    'direction': Box(low=-1, high=1, shape=(3,)),
    'chest_displacement': Box(low=-np.inf, high=np.inf, shape=(3,)),
    'field_displacement': Box(low=-np.inf, high=np.inf, shape=(3,)),
    'facing': Box(low=0, high=1, shape=(len(self.block_idxs),)),
    'wheat_age': Box(low=0, high=1)
})
```

The meaning of each of these is as follows:
- `timesteps_remaining`: The amount of time remaining before the environment terminates itself, normalized to be between 0 and 1. Shown to be necessary for learning by \[[1](https://arxiv.org/abs/1712.00378)].
- `world_time`: The current Minecraft world time, normalized to be between 0 and 1.\*
- `light_level`: The world light level, normalized to be between 0 and 1.\*
- `fuel`: The amount of remaining fuel, normalized to 0 and 1.
- `wheat`: The amount of currently held wheat in the inventory.
- `direction`: The unit vector the agent is looking at.
- `chest_displacement`: The displacement vector of the agent and the chest.
- `field_displacement`: The displacement vector of the agent and the center of the wheat field.
- `facing`: A one-hot encoded vector of the type of block the agent is looking at
- `wheat_age`: A binary feature indicating if the wheat is harvestable (age 7) or not.

\* This is not relevant for the current wheatbot task; rather, I intend(ed) to add another subtask of using a composter and obtaining seeds from the wheat plants. In that case, the time of day and world light level are relevant.

Importantly, the observation only consists of blocks perceivable to the turtle (adjacent ones) and it does not have access to the entire world, like it would if we rendered the environment from top-down and used a CNN. This means the environment is partially-observable (and technically since multiple minecraft ticks can happen between actions, this is also a semi-Markov decision process).

## Action Space

The action space is a simple `Discrete` space consisting of the actions
- `no-op`: Do nothing and wait
- `move-forward`: Move the agent one block in the direction it is looking, if possible. Consumes 1 fuel if successful
- `turn-left`: Turn the agent counterclockwise 90 degrees
- `turn-right`: Turn the agent clockwise 90 degrees
- `harvest`: Harvests wheat the bot is looking at, if possible. The agent must be looking at a wheat block
- `interact-chest`: Returns collected wheat to the chest, ending the episode, if possible. The agent must be (1) facing the chest and (2) have 1 or more wheat items in its inventory

The use of invalid action masking simplifies the action space the agent has to learn.

## Rewards

The reward functions are designed around the following set of preferences:

1. The agent should prefer to bring back wheat rather than just harvest it
2. The agent should prefer to bring back as much wheat as possible to the chest

To ensure the agent wants to obtain as much wheat as possible, harvesting wheat successfully should have a positive reward $r_h$ (`harvest_reward`). To ensure the optimal policy is to return some wheat rather than collect an _infinite_ amount of wheat, we use a tiered reward function [[2](https://arxiv.org/abs/2212.03733)] to set the reward for returning to the chest $r_c$ (`chest_reward`), as

$$r_t < \frac{1}{1-\gamma} r_h < r_c,$$

where $r_t$ (`timestep_reward`) is the timestep reward received at each timestep, often to penalize taking longer than necessary in some RL problems.

There's a few problems with this reward structure as specified:

1. The agent doesn't care whether it returns 1 or 100 or $\infty$ wheat, while we want it to return as much as possible within the time limits.
2. Returning to the chest is a sparse reward that is difficult to learn.

To fix #1, we can add a small bonus to the chest reward

$$r_c \rightarrow r_c + \beta w$$

where $\beta$ (`scale_chest_reward`) is a small factor that adds a bonus depending on the amount of wheat $w$ the agent holds. This ensures that while it's preferable to return 1 wheat instead of collect an infinite amount, returning 5 wheat is preferable to returning just 1. The agent should learn this by correlations in the terminal reward $r_c$ and the observation field `wheat`.

Putting all of that together, we have

$$ r_h < (1 - \gamma)r_c + \beta w. $$

We can also motivate the agent to go to the wheat field (enabling it to then harvest wheat) by adding a small one-time reward $r_f$ (`field_reward`) for the first time it goes within a certain radius of the center of the field.

### Potential-based reward shaping

As noted above, the chest reward and the field reward are sparse, which is difficult to learn. To fix this, we can employ potential-based reward shaping (PBRS), which provides reward hints of the form

$$r_\phi = \gamma \phi(s') - \phi(s),$$

where $\phi(s)$ is a _potential_ function that depends only on the state $s$. Using a PBRS form leaves the optimal policy unmodified.

The first PBRS function we will add is $r_{\phi_f}$ (`field_pbrs`) to motivate the agent to explore more towards the field. This can be done with a $\phi_f(||\mathbf{x}-\mathbf{x}_f||_1)$, where $\mathbf{x}_f$ is the center of the field.

We define a similar function for the chest $\phi_c(||\mathbf{x}-\mathbf{x}_c||_1)$ (`chest_pbrs`).

In practice, there are several possible choices of potential function,

- `r` (default): $\phi(d) \sim -d$
- `gaussian`: $\phi(d) \sim e^{-d^2 / 2\sigma_2}$
- `1/r`: $\phi(d) \sim 1/d$

### Rewards in Hierarchical Environment

The reward scheme above just adds all rewards to a single scalar for the agent to learn from. For a hierarchical policy, we can divide the rewards up based on the subgoal chosen by the top-level agent. This is done when `assign_subtasks` is true, and the rewards are split up as follows:

- For the `goto-field` subgoal, the subpolicy receives rewards $r_f$ (`field_reward`) and $r_{\phi_f}$ (field PBRS)
- For the `harvest-wheat` subgoal, the subpolicy receives rewards $r_h$ (`harvest_reward`)
- For the `goto-chest` subgoal, the subpolicy receives rewards $r_c$ (`chest_reward` and the scaling term) and $r_{\phi_c}$ (chest PBRS)

## References

[1] Pardo et al, 2022. Time limits in reinforcement learning. https://arxiv.org/abs/1712.00378

[2] Zhou et al, 2022. Specifying behavior preference with tiered reward functions. https://arxiv.org/abs/2212.03733
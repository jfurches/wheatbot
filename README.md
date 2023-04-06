# Wheatbot

Partially observable, hierarchical RL environment based on Minecraft

## Environment

This project is inspired by the [ComputerCraft turtle](https://www.computercraft.info/wiki/Turtle), a
Minecraft mod that introduces simplistic robots into the world. My objective is to train the turtle
to autonomously complete tasks, and then to deploy the policies in the game.

### Objective

![Wheatbot environment](https://user-images.githubusercontent.com/38408451/229688875-696af56c-64de-41c8-8644-a0c79260e6ff.png)

In what might be the most over-engineered wheat farm in Minecraft, the agent starts randomly on a wheat farm,
and it must navigate to the wheat, collect it, and bring back what it collected to a chest within a time limit
and fixed fuel budget. While much less impressive than MineDojo, there's a number of aspects that make this problem challenging:

1. **Partial Observability** The turtle, unlike most real-world robots, has extremely limited perception of its environment. It can only perceive blocks it is directly touching, and its action space is similarly limited to moving forward, turning, and manipulating blocks it is facing. This is solved using recurrent models (transformers) that can remember previous observations to better estimate the current state.

2. **Parametric Action Space** Some actions are only valid occasionally, such as moving forwards or mining certain blocks. This is implemented using invalid action masking (setting invalid logits to `-inf` so the softmax generates a probability of 0).

3. **Task Dependency** The robot must complete the subgoals in the order above to succeed at the task, and the long horizon of the problem makes it a good candidate for hierarchical reinforcement learning.

4. **Reward Function** To guarantee optimal behavior and facilitate learning, the reward function consists of multiple potential-based reward shaping functions to guide the agent, as well as hierarchical reward scaling based on the agent's `gamma` value to ensure its preferences align with the desired behavior.

Furthermore, actually deploying the agent inside a real instance of Minecraft is experience similar to deploying it in a real robot.

For more information on the environment design, see [FarmingEnv.md](farmingenv.md)

## Setup

An `env.yml` file is provided to generate a conda environment,

```python
conda env create -n wheatbot -f env.yml
```

## Training

I used RLlib to train PPO agents on the custom environment. An example of training on the regular environment ("flat") is in `src/examples/farming_ppo.py`. A training script for the hierarchical environment is given in
`src/examples/hierarchical_farming_ppo.py`.

## Deployment

Coming soon..

from collections import deque, namedtuple
import random
import textwrap
from typing import Callable, List
import numpy as np

from ..base import Agent, TurtleState, Action, Transition

import logging

import torch
import torch.nn as nn
import torch.optim as optim

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, features: int, actions: int, linear_layers: int):
        super(DQN, self).__init__()

        self.n_features = features
        self.actions = actions

        self.model = nn.Sequential(
            nn.Linear(features, linear_layers),
            nn.ReLU(),
            nn.Linear(linear_layers, actions),
            nn.Softmax(0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ICMAgent(Agent):
    def __init__(self, f: Callable[[TurtleState], torch.Tensor], features: int, actions: List[Action], linear_layers:int = 10, 
                replay_buffer = 5000, device=None, batch_size=128, gamma=0.9, eps=0.1, 
                loss=nn.SmoothL1Loss(), optimizer=optim.RMSprop, weights=None):
        super(ICMAgent, self).__init__()

        self.f = f
        self.replay = ReplayMemory(replay_buffer)
        self.device = device

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.features = features
        self.actions = actions
        self.linear_layers = linear_layers

        self.batch_size = batch_size

        self.gamma = gamma
        self.eps = eps

        self.policy_net = DQN(features, len(actions), linear_layers).to(self.device)
        self.target_net = DQN(features, len(actions), linear_layers).to(self.device)

        if weights is not None:
            self.target_net.load_state_dict(torch.load(weights))
            logging.info(f'Loaded state dict from {weights}')

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.loss = loss
        self.optimizer = optimizer(self.policy_net.parameters())

        logging.info(f"Loaded DQN agent:\n{self}")
    
    def get_action(self, state: TurtleState) -> Action:
        if random.random() < self.eps:
            action = np.random.choice(state.get_valid_actions())
            return action
        else:
            with torch.no_grad():
                logging.debug('Generating new action')
                features = self.f(state).view(1, -1)
                q_values: torch.Tensor = self.policy_net(features)
                logging.debug(f'Q-Values: {q_values}')
                # valid actions mask
                valid_mask = torch.tensor([[action.can_perform(state) for action in self.actions]], device=self.device)
                logging.debug(f'Mask: {valid_mask}')
                # set invalid actions to 0, since the predicted q values of the network are [0-1]
                # q_values[~valid_mask] = 0
                q_values[~valid_mask] = -np.inf
                idx = q_values.max(1)[1].item()

                return self.actions[idx]
    
    def handle_action(self, sample: Transition):
        # In replay memory, we modify the transition to hold an int action instead of the Action object,
        # also converting the TurtleState object to tensor feature representations
        sample = Transition(
            self.f(sample.state).view(1, -1), 
            torch.tensor([sample.action.get_id()]).view(1, -1), 
            None if sample.next_state.is_terminal() else self.f(sample.next_state).view(1, -1), 
            torch.tensor([sample.reward]).view(1, -1))

        self.replay.push(*sample)
        self.optimize()

    def optimize(self):
        if len(self.replay) < self.batch_size:
            return
        
        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.cat(batch.state).to(self.device)
        actions = torch.cat(batch.action).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        
        logging.info(f'State size: {states.size()}')
        logging.info(f'Action size: {actions.size()}')
        logging.info(f'Rewards size: {rewards.size()}')

        q_values: torch.Tensor = self.policy_net(states)
        q_values = q_values.gather(1, actions)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_q_values = (next_state_values.view(-1, 1) * self.gamma) + rewards

        loss: torch.Tensor = self.loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.clamp(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        logging.info('Updated target net')
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def __str__(self):
        return textwrap.dedent(
            """\
            DQN Agent(
                device: {device}
                replay buffer: {replay_cap}
                features: {features}
                actions: {actions}
                linear layers: {layers}
                loss: {loss}
                optimizer: {optim}
            )
            """).format(
                device=self.device, replay_cap=self.replay.capacity, features=self.features, actions=[str(a) for a in self.actions], 
                layers=self.linear_layers, loss=self.loss, optim=self.optimizer)
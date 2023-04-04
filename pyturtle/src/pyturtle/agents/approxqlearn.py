from typing import Callable
from ..base import Agent, Transition, TurtleState

import logging

import numpy as np

class ApproxQLearnAgent(Agent):
    "Does approximate Q learning with manually engineered feature function"

    def __init__(self, x: Callable[[TurtleState], np.ndarray], out_shape, alpha=0.1, gamma=0.9, eps=1e-1):
        self.x = x                  # feature function
        self.out_shape = out_shape  # shape of features
        self.w = np.zeros(out_shape)

        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.eps = eps              # e-greedy chance
    
    def get_action(self, state: TurtleState):
        actions = state.get_valid_actions()

        logging.info(f'[Agent] Valid actions: {list(map(str, actions))}')

        if len(actions) == 0:
            return None
        
        # epsilon-greedy
        if np.random.random() < self.eps:
            logging.info('Choosing random action')
            action = np.random.choice(actions)
        # select action with highest q value
        else:
            q_values = self.q_values(state)
            action = actions[np.argmax(q_values)]
        
        return action
    
    def q_values(self, state: TurtleState):
        actions = state.get_valid_actions()
        g = lambda a: np.dot(self.x(state, a), self.w)
        q_values = [g(a) for a in actions]

        return q_values
    
    def handle_action(self, sample: Transition):
        f = self.x(sample.state, sample.action)   # features of Q(s, a)
        Q = np.dot(f, self.w)       # Q(s, a)
        q_values = self.q_values(sample.next_state)
        V = np.max(q_values) if len(q_values) > 0 else 0 # V(S')

        correction = (sample.reward + self.gamma * V) - Q
        self.w += self.alpha * correction * f # weight update

        logging.info(f'[Agent] Features: {f}')
        logging.info(f'[Agent] Weights: {self.w}')
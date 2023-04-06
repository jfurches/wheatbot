from abc import ABC, abstractmethod
from ast import AST
from collections import namedtuple
from dataclasses import dataclass, field
import logging
from typing import Any, List, Tuple
from queue import PriorityQueue, Queue

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class TurtleState(ABC):
    @abstractmethod
    def sync(self, obj):
        pass

    @abstractmethod
    def get_valid_actions(self) -> list:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

class Action(ABC):
    count = 0

    def __init__(self):
        self.id = Action.count
        Action.count += 1

    @abstractmethod
    def do(self, state: TurtleState) -> Tuple[float, TurtleState]:
        "Performs the action on a state, returning a new state and the reward associated with the action, if any"
        pass

    @abstractmethod
    def can_perform(self, state: TurtleState) -> bool:
        "Returns whether or not an action can be performed from a particular state"
        pass

    def get_id(self):
        return self.id

class Agent(ABC):
    @abstractmethod
    def get_action(self, state: TurtleState) -> Action:
        "Represents the policy"
        pass
    
    @abstractmethod
    def handle_action(self, sample: Transition):
        pass

class Connector(ABC):
    @abstractmethod
    def get_new_state(self) -> Tuple[float, TurtleState]:
        pass

    @abstractmethod
    def send_action(self, action: Action):
        pass

class Turtle:
    ANIM_TICKS = 8
    
    def __init__(self, agent: Agent, connector: Connector):
        self.agent = agent
        self.connector = connector
        self.action_queue: Queue[Action] = Queue()
        _, self.state = connector.get_new_state()
    
    def tick(self):
        # send actions through the controller to a real turtle or a simulation
        lastAction = None
        newState = self.state
        while not self.action_queue.empty():
            action = self.action_queue.get()
            lastAction = action
            self.connector.send_action(action)
        
        if lastAction is not None:
            # get resulting state and any reward associated with it
            # then send that data to the agent so it can learn
            reward, newState = self.connector.get_new_state()
            sample = Transition(self.state, lastAction, newState, reward)
            self.agent.handle_action(sample)

        # finally use the new state and use the agent's policy to derive our next
        # action, then enqueue it
        self.state = self.state.sync(newState)
        nextAction = self.agent.get_action(newState)

        if nextAction is not None:
            self.action_queue.put(nextAction)

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

class AStarAction(Action, ABC):
    # FIXME: borked
    "An action that involves A* search to reach a goal, e.g. returning to a point"

    def __init__(self):
        super(AStarAction, self).__init__()

        self.actions: list[Action] = []

    @abstractmethod
    def cost(self, state: TurtleState, actions):
        pass

    @abstractmethod
    def heuristic(self, state: TurtleState):
        pass

    @abstractmethod
    def is_goal(self, state: TurtleState) -> bool:
        pass

    def value(self, state: TurtleState, actions):
        return self.cost(state, actions) + self.heuristic(state)

    def search(self, state: TurtleState) -> List[Action]:
        # data will be form of (priority, (actions, state))
        queue: Queue[PrioritizedItem] = PriorityQueue()

        curr_state = state.copy()
        queue.put(PrioritizedItem(self.value(curr_state, []), ([], curr_state)))

        while not queue.empty():
            item = queue.get()
            logging.info(f'[Search] Got new state. Queue size: {queue.qsize()}')
            actions, curr_state = item.item

            logging.info(f'[Search] Current pos: {curr_state.pos}, actions: {list(map(str, actions))}')

            if self.is_goal(curr_state):
                break

            for action in filter(lambda it: it.can_perform(curr_state), self.actions):
                actions = actions + [action]
                _, newState = action.do(curr_state, tick=False)
                v = self.value(newState, actions)
                logging.info(f'Cost of new action {action}: {v}')
                queue.put(PrioritizedItem(v, (actions, newState)))
            
            logging.getLogger().handlers[0].flush()
        
        return actions
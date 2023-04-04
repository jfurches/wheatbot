import logging
from .base import *

class SimulationConnector(Connector):
    def __init__(self, starting_state: TurtleState) -> None:
        super(Connector, self).__init__()

        self.state = starting_state
        self.last_reward = 0

    def get_new_state(self) -> Tuple[float, TurtleState]:
        return self.last_reward, self.state
    
    def send_action(self, action: Action):
        logging.info(f'[Connector] Received action {action}')
        self.last_reward, self.state = action.do(self.state)
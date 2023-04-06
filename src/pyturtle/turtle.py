from abc import ABC, abstractmethod
from block import Block, BlockType
from world import Direction, World
from inventory import Inventory, ItemStack
from typing import Tuple

import numpy as np

class Turtle:
    def __init__(self, predictor, gamma=0.9, lr=1e-2):
        self.predictor = predictor
        self.world = World()
        self.state = TurtleState(self.world)
        self.gamma = gamma
        self.lr = lr

        self.episode = []
    
    def get_action(self, state=None):
        if state == None:
            state = self.state

        q_values = []
        actions = state.get_valid_actions()
        for action in actions:
            q_values.append(self.predictor(state, action))
        
        return actions[np.argmax(q_values)]



class TurtleState:
    BASIC_FUEL_LIMIT = 20000
    ADVANCED_FUEL_LIMIT = 100000
    NO_FUEL_PENALTY = -1000

    def __init__(self, world, fuel=0, fuelLimit=BASIC_FUEL_LIMIT, inventory=Inventory(), pos=World.ORIGIN.copy()):
        # TODO: possible improvement is replacing inventory knowledge with a counter of how many 
        # fuel itemstacks there are
        self.world = world
        self.fuel = fuel
        self.fuelLimit = fuelLimit
        self.inventory = inventory
        self.pos = pos # position local to the model of the world
    
    def get_valid_actions(self):
        return list(filter(lambda a: a.can_perform(self), Turtle.actions))
    
    def copy(self):
        "Creates a clone, but does not copy the world"
        return TurtleState(self.world, self.fuel, self.fuelLimit, self.inventory.copy(), self.pos.copy())
    
    def deepcopy(self):
        "Creates a clone, also copying the world"
        return TurtleState(self.world.copy(), self.fuel, self.fuelLimit, self.inventory.copy(), self.pos.copy())

class Action(ABC):
    ALL = set()

    @abstractmethod
    def perform(self, turtle: Turtle):
        pass

    @abstractmethod
    def simulate(self, state: TurtleState) -> Tuple[int, TurtleState]:
        "Performs the action on a state, returning a new state and the reward associated with the action, if any"
        pass

    @abstractmethod
    def can_perform(self, state: TurtleState) -> bool:
        "Returns whether or not an action can be performed from a particular state"
        pass

class Refuel(Action):
    """Consumes coal in the inventory to refuel the turtle"""
    def perform(self, turtle: Turtle):
        pass

    def simulate(self, state: TurtleState):
        newState = state.copy()
        for i, item in enumerate(newState.inventory.items):
            if item.is_fuel():
                newState.fuel += item.get_refuel_amount()
                if item.amount == 1:
                    newState.inventory.items[i] = ItemStack.empty()
                else:
                    item.amount -= 1
                
                break
    
        return -1, newState
    
    def can_perform(self, state: TurtleState):
        return state.inventory.has_fuel()

class Movement(Action):
    MOVEMENT_PENALTY = -1

    def __init__(self, direction: Direction):
        self.direction = direction

    def perform(self, turtle: Turtle):
        pass
    
    def simulate(self, state: TurtleState):
        newState = state.copy()
        newState.pos += self.direction.vector()
        newState.fuel -= 1

        # penalize the robot for running out of fuel
        reward = TurtleState.NO_FUEL_PENALTY if newState.fuel == 0 else Movement.MOVEMENT_PENALTY

        return reward, newState

    def can_perform(self, state: TurtleState):
        newPos = state.pos + self.direction.vector()

        # check if new position is outside our world border
        if not state.world.in_world(newPos):
            return False
        
        # can't move through solid blocks
        if state.world.blocks[newPos].is_solid():
            return False
        
        # can't move if we have no fuel
        if state.fuel <= 0:
            return False
        
        return True

class Mine(Action):
    def __init__(self, direction: Direction):
        self.direction = direction
    
    def perform(self, turtle: Turtle):
        pass

    def simulate(self, state: TurtleState):
        newState = state.deepcopy()
        blockPos = newState.pos + self.direction.vector
        block = newState.world.blocks[blockPos]
        newState.world.blocks[blockPos] = Block.empty()
        reward = 0

        # we mined fuel, account for it
        if block.type == BlockType.COAL:
            fuelItem = newState.inventory.get_item(Inventory.FUEL_SLOT)
            # add to existing itemstack
            if newState.inventory.has_fuel():
                fuelItem.amount += 1
            # have to make a new one
            else:
                newState.inventory.items[Inventory.FUEL_SLOT] = ItemStack.coal(1)

            reward = 0
        else:
            reward = block.get_mining_reward()
        
        return reward, newState
    
    def can_perform(self, state: TurtleState) -> bool:
        blockPos = state.pos + self.direction.vector
        return state.world.in_world(blockPos) and state.world.blocks[blockPos].type != BlockType.AIR
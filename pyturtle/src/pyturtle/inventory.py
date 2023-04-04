from dataclasses import dataclass
from typing import Optional

ItemType = str

@dataclass
class ItemStack:
    REFUEL_TYPES = {
        'minecraft:coal': 80
    }

    itemtype: Optional[ItemType]
    amount: int
    
    def is_fuel(self):
        return self.itemtype in ItemStack.REFUEL_TYPES and self.amount > 0
    
    def get_refuel_amount(self):
        ItemStack.REFUEL_TYPES.get(self.itemtype, 0)
    
    def copy(self):
        return ItemStack(self.itemtype, self.amount)

    @staticmethod
    def empty():
        return ItemStack(None, 0)
    
    @staticmethod
    def coal(amount):
        return ItemStack('minecraft:coal', amount)

class Inventory:
    CAPACITY = 16
    FUEL_SLOT = 0

    def __init__(self, items = None):
        self.items = items or [ItemStack.empty() for _ in range(Inventory.CAPACITY)]
    
    def get_item(self, i: int):
        return self.items[i]
    
    def find_item(self, itemtype: ItemType, return_idx = False) -> ItemStack:
        for idx, item in enumerate(self.items):
            if item.itemtype == itemtype:
                if return_idx:
                    return idx, item
                else:
                    return item
        
        if return_idx:
            return -1, ItemStack.empty()
        else:
            return ItemStack.empty()

    def has_fuel(self):    
        fuelItem = self.get_item(Inventory.FUEL_SLOT)
        return fuelItem.is_fuel()
    
    def use_fuel(self):
        if self.has_fuel():
            fuelItem = self.get_item(Inventory.FUEL_SLOT)
            if fuelItem.amount == 1:
                self.items[Inventory.FUEL_SLOT] = ItemStack.empty()
            else:
                fuelItem.amount -= 1
        else:
            raise Exception('No fuel to consume')

    def copy(self):
        return Inventory(items=[item.copy() for item in self.items])

    def add_item(self, item: ItemStack):
        for slot, invitem in enumerate(self.items):
            if slot == Inventory.FUEL_SLOT:
                continue

            if invitem.itemtype is None:
                self.items[slot] = item.copy()
                return

            elif invitem.itemtype == item.itemtype:
                self.items[slot] = ItemStack(item.itemtype, item.amount + invitem.amount)
                return

        raise Exception('No space in inventory')

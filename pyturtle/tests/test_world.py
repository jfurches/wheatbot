import unittest

from .context import pyturtle

from pyturtle.world import World, Direction
from pyturtle.block import Block, BlockType

class WorldTest(unittest.TestCase):

    def test_init(self):
        world = World()
        ids = set()

        for i in range(5):
            for j in range(5):
                for k in range(5):
                    ids.add(id(world.blocks[i, j, k]))
        
        # test all the blocks are different objects
        self.assertEqual(len(ids), World.DIM ** 3)
    
    def test_shift(self):
        world = World()
        world.blocks[2, 2, 2].type = BlockType.COAL

        block_id = id(world.blocks[2, 2, 2])

        world.shift(Direction.UP)

        self.assertEqual(world.blocks[2, 2, 2].type, BlockType.AIR)
        self.assertEqual(world.blocks[2, 2, 1].type, BlockType.COAL)
        self.assertEqual(id(world.blocks[2, 2, 1]), block_id)

        world.shift(Direction.WEST)

        self.assertEqual(world.blocks[2, 2, 1].type, BlockType.AIR)
        self.assertEqual(world.blocks[3, 2, 1].type, BlockType.COAL)

if __name__ == '__main__':
    unittest.main()
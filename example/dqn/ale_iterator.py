__author__ = 'sxjscience'

from ale_python_interface import ALEInterface
import numpy
from replay_memory import ReplayMemory
from defaults import *

class ALEIterator(object):
    def __init__(self, rom_path=ALEDefaults.ROM_PATH, display_screen=ALEDefaults.DISPLAY_SCREEN,
                 rows=ALEDefaults.RESIZED_ROWS, cols=ALEDefaults.RESIZED_COLS):
        self.ale = self.load_from_rom(rom_path=rom_path, display_screen=display_screen)
        self.rows = rows
        self.cols = cols
        self.replay_memory = ReplayMemory()

    def load_from_rom(self, rom_path, display_screen):
        rng = numpy.random.RandomState(123456)
        ale = ALEInterface()
        ale.setInt('random_seed', rng.randint(1000))
        ale.setBool('display_screen', display_screen)
        ale.setFloat('repeat_action_probability', 0)
        ale.loadROM(rom_path)
        return ale


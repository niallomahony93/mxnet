__author__ = 'sxjscience'

from ale_python_interface import ALEInterface
import numpy

_basic_config = {}
_basic_config['display_screen'] = False

def load_from_rom(rom_path='breakout.bin', config=_basic_config):
    rng = numpy.random.RandomState(123456)
    ale = ALEInterface()
    ale.setInt('random_seed', rng.randint(1000))
    ale.setBool('display_screen', config['display_screen'])
    ale.setFloat('repeat_action_probability', 0)
    ale.loadROM(rom_path)
    return ale

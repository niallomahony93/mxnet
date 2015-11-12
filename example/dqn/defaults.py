__author__ = 'sxjscience'
import mxnet as mx
import numpy.random

_ctx = mx.cpu()
_numpy_rng = numpy.random.RandomState(10000)
def get_ctx():
    return _ctx

def get_numpy_rng():
    return _numpy_rng

class ALEDefaults:
    ROM_PATH = "./roms/breakout.bin"
    DISPLAY_SCREEN = False
    FRAME_SKIP = 4
    RESIZED_ROWS = 84
    RESIZED_COLS = 84
    EXPLORATION_PROB_START = 1.0
    EXPLORATION_PROB_MIN = .1
    EXPLORATION_PROB_DECAY = 1E-6
    STEPS_PER_EPOCH = 250000
    EPOCHS = 200
    STEPS_PER_TEST = 125000
    MINIBATCH_SIZE = 32

class ReplayMemoryDefaults:
    REPLAY_MEMORY_SIZE = 1000000
    REPLAY_START_SIZE = 50000
    SLICE_LENGTH = 4
    UPDATE_FREQUENCY = 4

class LossDefaults:
    CLIP_DELTA = 1.0

class OptimizerDefaults:
    LEARNING_RATE = .00025
    RMS_DECAY = .95
    RMS_EPSILON = .01
    FREEZE_INTERVAL = 10000
    RESIZE_METHOD = 'scale'
    MAX_START_NULLOPS = 30
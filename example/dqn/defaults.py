__author__ = 'sxjscience'
import mxnet as mx
import numpy.random

_ctx = mx.cpu()
_numpy_rng = numpy.random.RandomState(10000)
def get_ctx():
    return _ctx

def get_numpy_rng():
    return _numpy_rng

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 250000
    EPOCHS = 200
    STEPS_PER_TEST = 125000
    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "./roms/breakout.bin"
    FRAME_SKIP = 4
    # ----------------------
    # Optimization Parameters:
    # ----------------------
    UPDATE_RULE = 'deepmind_rmsprop'
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95 # (Rho)
    RMS_EPSILON = .01

    CLIP_DELTA = 1.0
    EXPLORATION_EPSILON_START = 1.0
    EXPLORATION_EPSILON_MIN = .1
    EXPLORATION_EPSILON_DECAY = 1000000
    SLICE_LENGTH = 4
    UPDATE_FREQUENCY = 4
    REPLAY_MEMORY_SIZE = 1000000
    MINIBATCH_SIZE = 32
    FREEZE_INTERVAL = 10000
    REPLAY_START_SIZE = 50000
    RESIZE_METHOD = 'scale'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    MAX_START_NULLOPS = 30
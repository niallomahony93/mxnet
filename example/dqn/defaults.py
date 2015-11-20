__author__ = 'sxjscience'
import mxnet as mx
import numpy.random
import os
_ctx = mx.cpu()
_numpy_rng = numpy.random.RandomState(10000)
def get_ctx():
    return _ctx

def get_numpy_rng():
    return _numpy_rng

class IteratorDefaults:
    ROM_PATH = os.path.dirname(os.path.realpath(__file__)) + "/roms/breakout.bin"
    DISPLAY_SCREEN = True
    FRAME_SKIP = 4
    SLICE_LENGTH = 4
    RESIZED_ROWS = 84
    RESIZED_COLS = 84
    EXPLORATION_PROB_START = 1.0
    EXPLORATION_PROB_MIN = .1
    EXPLORATION_PROB_DECAY = 1E-6
    EPOCH_MAX_STEP = 250000
    EPOCHS = 10000000
    STEPS_PER_TEST = 125000
    BATCH_SIZE = 32
    RESIZE_MODE = 'crop'
    CROP_OFFSET = 8
    DISCOUNT = 0.95

class ReplayMemoryDefaults:
    REPLAY_MEMORY_SIZE = 1000000
    REPLAY_START_SIZE = 100

class LossDefaults:
    CLIP_DELTA = 1.0

class OptimizerDefaults:
    LEARNING_RATE = .0002
    RMS_DECAY = .99
    RMS_EPSILON = 1E-6
    MAX_START_NULLOPS = 30

class DQNDefaults:
    SHORTCUT_INTERVAL = 1
    SAVE_INTERVAL = 10000
    SAVE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/model"
    SAVE_PREFIX = 'dqn'
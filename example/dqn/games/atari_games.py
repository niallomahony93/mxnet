__author__ = 'sxjscience'

from ale_python_interface import ALEInterface
import mxnet as mx
import numpy
import cv2
import logging
import os
from ..replay_memory import ReplayMemory

logger = logging.getLogger(__name__)

_numpy_rng = numpy.random.RandomState(123456)

_dirname = os.path.dirname(os.path.realpath(__file__))
_default_rom_path = os.path.join(_dirname, "roms", "breakout.bin")

def ale_load_from_rom(rom_path, display_screen):
    rng = _numpy_rng
    ale = ALEInterface()
    ale.setInt('random_seed', rng.randint(1000))
    ale.setBool('display_screen', display_screen)
    ale.setFloat('repeat_action_probability', 0)
    ale.loadROM(rom_path)
    return ale


class AtariGame(object):
    def __init__(self,
                 rom_path=_default_rom_path,
                 frame_skip=4, history_length=4,
                 resize_mode='scale', resized_rows=84, resized_cols=84, crop_offset=8,
                 display_screen=False, max_null_op=30,
                 replay_memory_size=1000000, replay_start_size=100):
        self.total_reward = 0
        self.current_step = 0
        self.rng = _numpy_rng
        self.ale = ale_load_from_rom(rom_path=rom_path, display_screen=display_screen)
        self.action_set = self.ale.getMinimalActionSet()
        self.resize_mode = resize_mode
        self.resized_rows = resized_rows
        self.resized_cols = resized_cols
        self.crop_offset = crop_offset
        self.frame_skip = frame_skip
        self.history_length = history_length
        self.max_null_op = max_null_op
        self.screen_buffer_length = 2
        self.screen_buffer = numpy.empty((self.screen_buffer_length,
                                          self.ale.getScreenDims()[1], self.ale.getScreenDims()[0]),
                                         dtype='uint8')
        self.replay_memory = ReplayMemory(rows=resized_rows, cols=resized_cols,
                                          history_length=history_length,
                                          memory_size=replay_memory_size,
                                          replay_start_size=replay_start_size)
        self.start()

    def start(self):
        self.ale.reset_game()
        null_op_num = self.rng.randint(self.screen_buffer_length,
                                       max(self.max_null_op + 1, self.screen_buffer_length + 1))
        for i in range(null_op_num):
            self.ale.act(0)
            self.ale.getScreenGrayscale(self.screen_buffer[i % self.screen_buffer_length, :, :])
        self.total_reward = 0
        self.current_step = 0

    def get_observation(self):
        image = self.screen_buffer.max(axis=0)
        if 'crop' == self.resize_mode:
            original_rows, original_cols = image.shape
            new_resized_rows = int(round(
                float(original_rows) * self.resized_cols / original_cols))
            resized = cv2.resize(image, (self.resized_cols, new_resized_rows),
                                 interpolation=cv2.INTER_LINEAR)
            crop_y_cutoff = new_resized_rows - self.crop_offset - self.resized_rows
            img = resized[crop_y_cutoff:
            crop_y_cutoff + self.resized_rows, :]
            return img
        else:
            return cv2.resize(image, (self.resized_cols, self.resized_rows),
                              interpolation=cv2.INTER_LINEAR)

    def current_state(self):
        return self.replay_memory.latest_slice()

    def play(self, a):
        self.current_step += 1
        reward = 0.0
        action = self.action_set[a]
        for i in xrange(self.frame_skip):
            reward += self.ale.act(action)
            self.ale.getScreenGrayscale(self.screen_buffer[i % self.screen_buffer_length, :, :])
        self.total_reward += reward
        ob = self.get_observation()
        terminate_flag = self.ale.game_over()
        self.replay_memory.append(ob, a, numpy.clip(reward, -1, 1), terminate_flag)
        return reward, terminate_flag

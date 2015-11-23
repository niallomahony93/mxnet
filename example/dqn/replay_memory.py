__author__ = 'sxjscience'
import numpy
from defaults import *

class ReplayMemory(object):
    def __init__(self, rows, cols, slice_length, memory_size=ReplayMemoryDefaults.REPLAY_MEMORY_SIZE,
                 replay_start_size=ReplayMemoryDefaults.REPLAY_START_SIZE,
                 state_dtype='uint8', action_dtype='uint8'):
        self.rng = get_numpy_rng()
        self.states = numpy.zeros((memory_size, rows, cols), dtype=state_dtype)
        self.actions = numpy.zeros(memory_size, dtype=action_dtype)
        self.rewards = numpy.zeros(memory_size, dtype='float32')
        self.terminate_flags = numpy.zeros(memory_size, dtype='bool')
        self.memory_size = memory_size
        self.replay_start_size = replay_start_size
        self.slice_length = slice_length
        self.top = 0
        self.size = 0

    def latest_slice(self):
        return self.states.take(numpy.arange(self.top - self.slice_length, self.top), axis=0, mode="wrap")\
            .reshape((1,) + (self.slice_length, self.states.shape[1], self.states.shape[2]))

    def append(self, img, action, reward, terminate_flag):
        self.states[self.top, :, :] = img
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminate_flags[self.top] = terminate_flag
        self.top = (self.top + 1) % self.memory_size
        if self.size < self.memory_size:
            self.size += 1

    def sample(self, batch_size):
        assert self.replay_start_size >= batch_size and self.replay_start_size >= self.slice_length
        assert(0 <= self.size <= self.memory_size)
        assert(0 <= self.top <= self.memory_size)
        if self.size <= self.replay_start_size:
            raise ValueError("Size of the effective samples of the ReplayMemory must be bigger than "
                             "start_size! Currently, size=%d, start_size=%d" %(self.size, self.replay_start_size))
        states = numpy.zeros((batch_size, self.slice_length, self.states.shape[1], self.states.shape[2]),
                             dtype=self.states.dtype)
        actions = numpy.zeros(batch_size, dtype=self.actions.dtype)
        rewards = numpy.zeros(batch_size, dtype='float32')
        terminate_flags = numpy.zeros(batch_size, dtype='bool')
        next_states = numpy.zeros((batch_size, self.slice_length, self.states.shape[1], self.states.shape[2]),
                                  dtype=self.states.dtype)
        counter = 0
        while counter < batch_size:
            index = self.rng.randint(low=self.top - self.size + 1, high=self.top - self.slice_length + 1)
            transition_indices = numpy.arange(index, index + self.slice_length)
            initial_indices = transition_indices - 1
            end_index = index + self.slice_length - 1
            if numpy.any(self.terminate_flags.take(transition_indices[0:-1], mode='wrap')):
                # Check if terminates in the middle of the sample!
                continue
            states[counter] = self.states.take(initial_indices, axis=0, mode='wrap')
            actions[counter] = self.actions.take(end_index, mode='wrap')
            rewards[counter] = self.rewards.take(end_index, mode='wrap')
            terminate_flags[counter] = self.terminate_flags.take(end_index, mode='wrap')
            next_states[counter] = self.states.take(transition_indices, axis=0, mode='wrap')
            counter += 1
        return states, actions, rewards, next_states, terminate_flags

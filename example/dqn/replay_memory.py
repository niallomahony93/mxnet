__author__ = 'sxjscience'
import numpy
from defaults import *

class ReplayMemory(object):
    def __init__(self, rows, cols, memory_size=ReplayMemoryDefaults.REPLAY_MEMORY_SIZE,
                 slice_length=ReplayMemoryDefaults.SLICE_LENGTH, state_dtype='uint8', action_dtype='int32'):
        self.states = numpy.zeros((memory_size, rows, cols), dtype=state_dtype)
        self.actions = numpy.zeros(memory_size, dtype=action_dtype)
        self.rewards = numpy.zeros(memory_size, dtype='float32')
        self.terminate_flags = numpy.zeros(memory_size, dtype='bool')
        self.memory_size = memory_size
        self.slice_length = slice_length
        self.top = 0
        self.size = 0

    def append(self, img, reward, terminate_flag):
        self.states[self.top, :, :] = img
        self.rewards[self.top] = img
        self.terminate_flags[self.top] = terminate_flag
        self.top = (self.top + 1) % self.memory_size
        if self.size < self.memory_size:
            self.size += 1

    def sample(self, minibatch_size):
        if self.size <= minibatch_size or self.size <= self.slice_length:
            raise ValueError("Size of the effective samples of the ReplayMemory must be bigger than "
                             "minibatch_size and slice_length! Currently, minibatch=%d, ReplayMemory=%d, slice_length=%d"
                                      %(minibatch_size, self.size, self.slice_length))
        assert(0 <= self.size <= self.memory_size)
        assert(0 <= self.top <= self.memory_size)
        rng = get_numpy_rng()
        states = numpy.zeros((minibatch_size, self.slice_length, self.states.shape[1], self.states.shape[2]),
                             dtype=self.states.dtype)
        actions = numpy.zeros(minibatch_size, dtype=self.actions.dtype)
        rewards = numpy.zeros(minibatch_size, dtype='float32')
        terminate_flags = numpy.zeros(minibatch_size, dtype='bool')
        next_states = numpy.zeros((minibatch_size, self.slice_length, self.states.shape[1], self.states.shape[2]),
                                  dtype=self.states.dtype)
        counter = 0
        while counter < minibatch_size:
            index = rng.randint(low=self.top - self.size, high=self.top - self.slice_length + 1)
            initial_indices = numpy.arange(index, index + self.slice_length)
            transition_indices = initial_indices + 1
            end_index = index + self.slice_length - 1
            if numpy.any(self.terminate_flags.take(initial_indices[0:-1], mode='wrap')):
                # Check if terminates in the middle of the sample!
                continue
            states[counter] = self.states.take(initial_indices, axis=0, mode='wrap')
            actions[counter] = self.actions.take(end_index, mode='wrap')
            rewards[counter] = self.rewards.take(end_index, mode='wrap')
            terminate_flags[counter] = self.terminate_flags.take(end_index, mode='wrap')
            next_states[counter] = self.states.take(transition_indices, axis=0, mode='wrap')
        return states, actions, rewards, next_states, terminate_flags

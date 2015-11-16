__author__ = 'sxjscience'

from ale_python_interface import ALEInterface
import numpy
from replay_memory import ReplayMemory
from defaults import *
import cv2
import mxnet as mx

class ALEIterator(mx.io.DataIter):
    def __init__(self, is_train=True,
                 rom_path=IteratorDefaults.ROM_PATH,
                 slice_length=IteratorDefaults.SLICE_LENGTH,
                 resized_rows=IteratorDefaults.RESIZED_ROWS,
                 resized_cols=IteratorDefaults.RESIZED_COLS,
                 epochs=IteratorDefaults.EPOCHS,
                 epoch_max_step=IteratorDefaults.EPOCH_MAX_STEP,
                 batch_size=IteratorDefaults.BATCH_SIZE,
                 discount=IteratorDefaults.DISCOUNT,
                 exploration_prob_start=IteratorDefaults.EXPLORATION_PROB_START,
                 exploration_prob_decay=IteratorDefaults.EXPLORATION_PROB_DECAY,
                 exploration_prob_min=IteratorDefaults.EXPLORATION_PROB_MIN,
                 frame_skip=IteratorDefaults.FRAME_SKIP,
                 display_screen=IteratorDefaults.DISPLAY_SCREEN):
        super(ALEIterator, self).__init__()
        self.ale = self.load_from_rom(rom_path=rom_path, display_screen=display_screen)
        self.action_set = self.ale.getMinimalActionSet()
        self.rows = resized_rows
        self.cols = resized_cols
        self.slice_length = slice_length
        self.is_train = is_train
        self.actor = None
        self.critic = None
        self.epochs = epochs
        self.epoch_max_step = epoch_max_step
        self.current_step = 0
        self.batch_size = batch_size
        self.discount = discount
        self.exploration_prob_start = exploration_prob_start
        self.exploration_prob_decay = exploration_prob_decay
        self.exploration_prob_min = exploration_prob_min
        self.frame_skip = frame_skip
        self.current_exploration_prob = self.exploration_prob_start
        self.screen_buffer_length = 2
        self.screen_buffer = numpy.empty((self.screen_buffer_length,
                                       self.ale.getScreenDims()[0], self.ale.getScreenDims()[1]),
                                      dtype='uint8')
        self.start_lives = self.ale.lives()
        self.epoch_reward = 0
        self.data = mx.ndarray.empty((self.batch_size, self.slice_length, self.rows, self.cols))
        self.action_reward = mx.ndarray.empty((self.batch_size, 2))
        self.replay_memory = ReplayMemory(rows=resized_rows, cols=resized_cols, slice_length=slice_length)

    def load_from_rom(self, rom_path, display_screen):
        rng = get_numpy_rng()
        ale = ALEInterface()
        ale.setInt('random_seed', rng.randint(1000))
        ale.setBool('display_screen', display_screen)
        ale.setFloat('repeat_action_probability', 0)
        ale.loadROM(rom_path)
        return ale

    def init_training(self, actor=None, critic=None):
        assert self.is_train
        self.actor = actor
        self.critic = critic
        while self.replay_memory.size < self.replay_memory.replay_start_size:
            self.act()

    @property
    def provide_data(self):
        return [('data', (self.batch_size, self.slice_length, self.rows, self.cols))]

    @property
    def provide_label(self):
        if self.is_train:
            return [('dqn_action_reward', (self.batch_size, 2))]
        else:
            return []

    def reset(self):
        self.ale.reset_game()
        self.start_lives = self.ale.lives()
        self.epoch_reward = 0

    def next(self):
        if self.iter_next():
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel(), pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def act(self, specific_action=None):
        rng = get_numpy_rng()
        # 1. Select an action based on the actor and the exploration probability
        if specific_action is not None:
            action = specific_action
        else:
            if rng.choice([True, False], p=[self.current_exploration_prob, 1 - self.current_exploration_prob]):
                action = rng.randint(0, len(self.action_set))
            else:
                action = numpy.argmax(self.actor.predict(self.replay_memory.latest_slice()), axis=1)
        # 2. Repeat the action for a fixed number of times and receive the image, reward and termination flag
        reward = 0
        for i in range(self.frame_skip):
            reward += self.ale.act(self.action_set[action])
            if i >= self.frame_skip - self.screen_buffer_length:
                self.ale.getScreenGrayscale(self.screen_buffer[i + self.screen_buffer_length - self.frame_skip, :, :])
        reward = numpy.clip(reward, -1, 1)
        self.epoch_reward += reward
        img = self.get_observation()
        terminate_flag = (self.ale.lives() < self.start_lives) or self.ale.game_over()
        self.replay_memory.append(img, action, reward, terminate_flag)

    def iter_next(self):
        if self.ale.game_over():
            return False
        else:
            self.current_step += 1
            if self.current_step > self.epoch_max_step:
                return False
            if self.is_train:
                # 1. Play the game for a single step and update the replay memory
                self.act()
                # 2. Sample a random batch from the replay memory
                self.data[:], actions, rewards, new_states, terminate_flags = self.replay_memory.sample(self.batch_size)
                # 3. Calculate the reward target
                rewards[terminate_flags.nonzero()] += \
                    self.discount*numpy.max(self.critic.predict(new_states[terminate_flags.nonzero(), ...]), axis=1)
                self.action_reward[:, 0] = actions
                self.action_reward[:, 1] = rewards
                self.current_exploration_prob = \
                    max(self.exploration_prob_min, self.current_exploration_prob - self.exploration_prob_decay)
            else:
                self.data[:] = self.replay_memory.latest_slice()

    def validate(self, holdout_size):
        holdout_data = self.replay_memory.sample(holdout_size)[0]
        return numpy.max(self.actor.predict(holdout_data), axis=1).sum()


    def get_observation(self):
        image = self.screen_buffer.max(axis=0)
        return cv2.resize(image, (self.cols, self.rows), interpolation=cv2.INTER_LINEAR)

    def getdata(self, index=None):
        return self.data

    def getlabel(self):
        if self.is_train:
            return self.action_reward
        else:
            return None

__author__ = 'sxjscience'

import mxnet as mx
from mxnet import metric
import numpy
from defaults import *
import logging
from ale_iterator import ALEIterator


class DQNOutputOp(mx.operator.NumpyOp):
    def __init__(self):
        super(DQNOutputOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action_reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_reward_shape = (in_shape[0][0], 2)
        output_shape = in_shape[0]
        return [data_shape, action_reward_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self, out_grad, in_data, out_data, in_grad):
        x = in_data[0]
        action_reward = in_data[1]
        action = action_reward[:, 0].astype(numpy.int)
        reward = action_reward[:, 1]
        dx = in_grad[0]
        dx[:] = 0
        dx[numpy.arange(action.shape[0]), action] = numpy.clip(x[numpy.arange(action.shape[0]), action] - reward, -1, 1)


class DQNInitializer(mx.initializer.Xavier):
    def _init_bias(self, _, arr):
        arr[:] = .1

class DQNMetric(mx.metric.CustomMetric):
    def get(self):
        if self.num_inst != 0:
            return (self.name, self.sum_metric / self.num_inst)
        else:
            return (self.name, self.sum_metric / (self.num_inst + 1))


class DQN(object):
    def __init__(self, iter):
        self.iter = iter
        self.DQNOutput = DQNOutputOp()
        self.dqn_sym = self.dqn_network(action_num=len(self.iter.action_set))
        self.metric = DQNMetric(self.dqn_metric)
        self.online_net = mx.model.FeedForward(symbol=self.dqn_sym, ctx=get_ctx(), initializer=DQNInitializer(),
                                               num_epoch=IteratorDefaults.EPOCHS, numpy_batch_size=self.iter.batch_size,
                                               optimizer='rmsprop', learning_rate=OptimizerDefaults.LEARNING_RATE,
                                               decay_rate=OptimizerDefaults.RMS_DECAY,
                                               eps=OptimizerDefaults.RMS_EPSILON, wd=0)
        self.online_net._init_params(dict(iter.provide_data + iter.provide_label))
        self.shortcut_net = mx.model.FeedForward(symbol=self.dqn_sym, ctx=get_ctx(), initializer=DQNInitializer(),
                                                 numpy_batch_size=self.iter.batch_size,
                                                 arg_params=self.online_net.arg_params)
        self.shortcut_net._init_predictor(dict(iter.provide_data))
        self.update_shortcut()
        self.iter.init_training(actor=self.online_net, critic=self.shortcut_net)
        print self.iter.replay_memory.rewards.sum()

    def dqn_network(self, action_num):
        net = mx.symbol.Variable('data')
        net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
        net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
        net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
        net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
        net = mx.symbol.Flatten(data=net)
        net = mx.symbol.FullyConnected(data=net, name='fc3', num_hidden=256)
        net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
        net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=action_num)
        net = self.DQNOutput(data=net, name='dqn')
        return net

    def dqn_metric(self, action_reward, qvec):
        action_reward_npy = action_reward.asnumpy()
        qvec_npy = qvec.asnumpy()
        action_npy = action_reward_npy[:, 0].astype(numpy.int)
        reward_npy = action_reward_npy[:, 1]
        return abs(qvec_npy[numpy.arange(action_npy.shape[0]), action_npy] - reward_npy).sum()

    def update_shortcut(self):
        # for v in self.online_net.arg_params.values():
        #     v.wait_to_read()
        self.shortcut_net._pred_exec.copy_params_from(self.online_net.arg_params, self.online_net.aux_params)
        # for k in self.online_net.arg_params.keys():
        #     self.shortcut_net._pred_exec.arg_dict[k].wait_to_read()
    def fit(self, iter):
        self.online_net.fit(X=self.iter, eval_metric=self.metric,
                            batch_end_callback=self.dqn_batch_callback, epoch_end_callback=self.dqn_epoch_end_callback)

    def dqn_batch_callback(self, param):
        if self.iter.current_step % DQNDefaults.SHORTCUT_INTERVAL == 0:
            self.update_shortcut()
        return

    def dqn_epoch_end_callback(self, epoch, symbol, arg_params, aux_states):
        # logging.info("Epoch Reward: %f" %self.iter.epoch_reward)
        if (epoch + 1) % DQNDefaults.SAVE_INTERVAL == 0:
            mx.model.save_checkpoint(DQNDefaults.SAVE_DIR + '/' + DQNDefaults.SAVE_PREFIX,
                                     epoch + 1, symbol, arg_params, aux_states)
        return


logging.basicConfig(level=logging.DEBUG)
if not os.path.exists(DQNDefaults.SAVE_DIR):
    os.mkdir(DQNDefaults.SAVE_DIR)
# data_shape = (5, 4, 84, 84)
# action_shape = (5,)
# action_reward_shape = (5, 2)
# X = mx.random.uniform(0, 10, data_shape)
# Y = mx.random.uniform(0, 3, action_shape)
# R = mx.random.uniform(0, 10, action_shape)
# Y_R = mx.ndarray.empty(action_reward_shape)
# Y_R[:] = numpy.vstack((Y.asnumpy(), R.asnumpy())).T
# iter = mx.io.NDArrayIter({'data': X}, {'dqn_action_reward': Y_R}, batch_size=5)
iter = ALEIterator()
dqn = DQN(iter)
dqn.fit(iter)

# W = mx.random.uniform(0, 10, (10, 4, 84, 84))
# print dqn.shortcut_net.predict(W)
# dqn.fit(iter)
# print dqn.online_net.predict(W)
# dqn.update_shortcut()
# print dqn.shortcut_net.predict(W)
# print R.asnumpy()

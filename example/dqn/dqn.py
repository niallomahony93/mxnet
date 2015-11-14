__author__ = 'sxjscience'

import mxnet as mx
from mxnet import metric
import numpy
from defaults import *
import logging


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


def dqn_metric(action_reward, qvec):
    action_reward_npy = action_reward.asnumpy()
    qvec_npy = qvec.asnumpy()
    action_npy = action_reward_npy[:, 0].astype(numpy.int)
    reward_npy = action_reward_npy[:, 1]
    return abs(qvec_npy[numpy.arange(action_npy.shape[0]), action_npy] - reward_npy).sum()


def dqn_network(action_num=4):
    net = mx.symbol.Variable('data')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc3', num_hidden=256)
    net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=action_num)
    DQNOutput = DQNOutputOp()
    net = DQNOutput(data=net, name='dqn')
    return net


class DQN(object):
    def __init__(self):
        data_shape = (5, 4, 84, 84)
        action_reward_shape = (5, 2)
        d = {'data': data_shape, 'dqn_action_reward': action_reward_shape}
        self.dqn_sym = dqn_network()
        self.online_net = mx.model.FeedForward(symbol=self.dqn_sym, ctx=get_ctx(), initializer=DQNInitializer(),
                                               num_epoch=100, numpy_batch_size=5,
                                               learning_rate=0.0001, momentum=0.9, wd=0.00001)
        self.online_net._init_params(d)
        self.shortcut_net = mx.model.FeedForward(symbol=self.dqn_sym, ctx=get_ctx(), initializer=DQNInitializer(),
                                                 numpy_batch_size=5, arg_params=self.online_net.arg_params)
        self.shortcut_net._init_predictor({'data': data_shape})
        self.update_shortcut()

    def update_shortcut(self):
        for v in self.online_net.arg_params.values():
            v.wait_to_read()
        self.shortcut_net._pred_exec.copy_params_from(self.online_net.arg_params, self.online_net.aux_params)
        for k in self.online_net.arg_params.keys():
            self.shortcut_net._pred_exec.arg_dict[k].wait_to_read()
    def fit(self, iter):
        self.online_net.fit(X=iter, eval_metric=mx.metric.CustomMetric(dqn_metric),
                            batch_end_callback=self.dqn_batch_callback)

    def dqn_batch_callback(self, param):
        print param
        return

    def dqn_epoch_end_callback(self, epoch, symbol, arg_params, aux_states):
        return


logging.basicConfig(level=logging.DEBUG)
data_shape = (5, 4, 84, 84)
action_shape = (5,)
action_reward_shape = (5, 2)
X = mx.random.uniform(0, 10, data_shape)
Y = mx.random.uniform(0, 3, action_shape)
R = mx.random.uniform(0, 10, action_shape)
Y_R = mx.ndarray.empty(action_reward_shape)
Y_R[:] = numpy.vstack((Y.asnumpy(), R.asnumpy())).T
iter = mx.io.NDArrayIter({'data': X}, {'dqn_action_reward': Y_R}, batch_size=5)
dqn = DQN()
W = mx.random.uniform(0, 10, (10, 4, 84, 84))
print dqn.shortcut_net.predict(W)
dqn.fit(iter)
print dqn.online_net.predict(W)
dqn.update_shortcut()
print dqn.shortcut_net.predict(W)
print R.asnumpy()

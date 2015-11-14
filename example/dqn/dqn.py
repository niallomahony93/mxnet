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

def dqn_callback(param):
    return

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

data_shape = (5, 4, 84, 84)
action_shape = (5,)
action_reward_shape = (5, 2)
dqn_sym = dqn_network()
X = mx.random.uniform(0, 10, data_shape)
Y = mx.random.uniform(0, 3, action_shape)
R = mx.random.uniform(0, 10, action_shape)
Y_R = mx.ndarray.empty(action_reward_shape)
Y_R[:] = numpy.vstack((Y.asnumpy(), R.asnumpy())).T
d = {'data':data_shape, 'dqn_action_reward': action_reward_shape}
logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(symbol=dqn_sym, ctx=get_ctx(),initializer=DQNInitializer(),
                             num_epoch=100, numpy_batch_size=5,
                             learning_rate=0.0001, momentum=0.9, wd = 0.00001)
arg_shapes, _, aux_shapes = model.symbol.infer_shape(**d)
arg_names, param_names, aux_names = model._init_params(d)
iter = mx.io.NDArrayIter({'data':X}, {'dqn_action_reward':Y_R}, batch_size=5)
print arg_names
print param_names
print aux_names

model.fit(X=iter, eval_metric=mx.metric.CustomMetric(dqn_metric), batch_end_callback=dqn_callback)
Y_new = mx.ndarray.zeros(action_shape)
iter = mx.io.NDArrayIter({'data':X}, batch_size=5)
print model.predict(iter)
print R.asnumpy()


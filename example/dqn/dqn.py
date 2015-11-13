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
        return ['data', 'action', 'reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = in_shape[1]
        reward_shape = (in_shape[0][0],)
        output_shape = (in_shape[0][0],)
        return [data_shape, action_shape, reward_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        action = in_data[1]
        action = action.flatten().astype(numpy.int)
        y[:] = x[numpy.arange(action.shape[0]), action].flatten()

    def backward(self, out_grad, in_data, out_data, in_grad):
        x = in_data[0]
        action = in_data[1]
        action = action.flatten().astype(numpy.int)
        reward = in_data[2]
        dx = in_grad[0]
        dx[:] = 0
        dx[numpy.arange(action.shape[0]), action] = numpy.clip(x[numpy.arange(action.shape[0]), action] - reward, -1, 1)


class DQNInitializer(mx.initializer.Xavier):
    def _init_bias(self, _, arr):
        arr[:] = .1

def dqn_callback(param):
    return

def dqn_metric(label, pred):
    return abs(label.asnumpy() - pred.asnumpy()).sum()

def dqn_network(action_num=4):
    net = mx.symbol.Variable('data')
    action = mx.symbol.Variable('action')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc3', num_hidden=256)
    net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=action_num)
    DQNOutput = DQNOutputOp()
    net = DQNOutput(data=net, action=action, name='dqn')
    return net

data_shape = (5, 4, 84, 84)
action_shape = (5,)
dqn_sym = dqn_network()
X = mx.random.uniform(0, 10, data_shape)
Y = mx.random.uniform(0, 1, action_shape)*3
R = mx.random.uniform(0, 10, action_shape)
d = {'data':data_shape, 'action':action_shape, 'dqn_reward': action_shape}
logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(symbol=dqn_sym, ctx=get_ctx(),initializer=DQNInitializer(),
                             num_epoch=100, numpy_batch_size=5,
                             learning_rate=0.0001, momentum=0.9, wd = 0.00001)
print model.arg_params
iter = mx.io.NDArrayIter({'data':X, 'action':Y}, {'dqn_reward':R}, batch_size=5)
model.fit(X=iter, eval_metric=mx.metric.CustomMetric(dqn_metric), batch_end_callback=dqn_callback)
iter = mx.io.NDArrayIter({'data':X, 'action':None}, batch_size=5)
print model.predict(iter)
print R.asnumpy()


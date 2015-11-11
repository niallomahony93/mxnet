__author__ = 'sxjscience'

import mxnet as mx
import numpy
from defaults import *

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
        y[:] = numpy.argmax(x, axis=1)
    def backward(self, out_grad, in_data, out_data, in_grad):
        x = in_data[0]
        action = in_data[1]
        action = action.flatten().astype(numpy.int)
        reward = in_data[2]
        dx = in_grad[0]
        dx[:] = 0
        dx[numpy.arange(action.shape[0]), action] = numpy.clip(x[numpy.arange(action.shape[0]), action] - reward, -1, 1)

def dqn_network(action_num=4):
    net = mx.symbol.Variable('data')
    action = mx.symbol.Variable('action')
    DQNOutput = DQNOutputOp()
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc3', num_hidden=256)
    net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=action_num)
    net = DQNOutput(data=[net, action], name='dqn')
    return net

data_shape = (4, 100, 100)
dqn_sym = dqn_network()
print dqn_sym.list_arguments()
dqn_sym.simple_bind(ctx=get_ctx(), data=data_shape)



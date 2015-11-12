__author__ = 'sxjscience'

import mxnet as mx
import numpy
from defaults import *

def dqn_loss(qval):
    return numpy.argmax(qval, axis=1)

def dqn_out_grad(qval, action, reward):
    if isinstance(qval, mx.ndarray.NDArray):
        qval = qval.asnumpy()
    action = action.flatten().astype(numpy.int)
    out_grad = numpy.zeros(qval.shape)
    out_grad[numpy.arange(qval.shape[0]), action] = numpy.clip(qval[numpy.arange(action.shape[0]), action] - reward, -1, 1)
    return out_grad

# def dqn_action(qval, exploration_prob=Defaults.EXPLORATION_EPSILON_DECAY):


def dqn_network(action_num=4):
    net = mx.symbol.Variable('data')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc3', num_hidden=256)
    net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=action_num)
    return net

data_shape = (4, 100, 100)
dqn_sym = dqn_network()
print dqn_sym.list_arguments()
dqn_sym.simple_bind(ctx=get_ctx(), data=data_shape)



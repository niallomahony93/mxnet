import mxnet as mx
import mxnet.ndarray as nd
import numpy
import logging
import argparse
from algos import *
from data_loader import *
from utils import *


class CrossEntropySoftmax(mx.operator.NumpyOp):
    def __init__(self):
        super(CrossEntropySoftmax, self).__init__(False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = numpy.exp(x - x.max(axis=1).reshape((x.shape[0], 1))).astype('float32')
        y /= y.sum(axis=1).reshape((x.shape[0], 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        y = out_data[0]
        dx = in_grad[0]
        dx[:] = (y - l)

class LogSoftmax(mx.operator.NumpyOp):
    def __init__(self):
        super(LogSoftmax, self).__init__(False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        print x - x.max(axis=1).reshape((x.shape[0], 1))
        ch = raw_input()
        y[:] = x - x.max(axis=1).reshape((x.shape[0], 1))
        y -= numpy.log(numpy.exp(y).sum(axis=1).reshape((x.shape[0], 1)))
        # y[:] = numpy.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        # y /= y.sum(axis=1).reshape((x.shape[0], 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        y = out_data[0]
        dx = in_grad[0]
        dx[:] = numpy.exp(y) - l

def classification_student_grad(student_outputs, teacher_pred):
    return [student_outputs[0] - teacher_pred]

def regression_student_grad(student_outputs, teacher_pred, teacher_noise_precision):
    student_mean = student_outputs[0]
    student_var = student_outputs[1]
    grad_mean = nd.exp(-student_var) * (student_mean - teacher_pred)

    grad_var = (1 - nd.exp(-student_var) * (nd.square(student_mean - teacher_pred)
                                                  + 1.0 / teacher_noise_precision))/2
    return [grad_mean, grad_var]

def get_mnist_sym(output_op=None):
    net = mx.symbol.Variable('data')
    net = mx.symbol.FullyConnected(data=net, name='mnist_fc1', num_hidden=400)
    net = mx.symbol.Activation(data=net, name='mnist_relu1', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='mnist_fc2', num_hidden=400)
    net = mx.symbol.Activation(data=net, name='mnist_relu2', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='mnist_fc3', num_hidden=10)
    if output_op is None:
        net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    else:
        net = output_op(data=net, name='softmax')
    return net

def get_toy_sym(teacher=True, teacher_noise_precision=None):
    if teacher:
        net = mx.symbol.Variable('data')
        net = mx.symbol.FullyConnected(data=net, name='teacher_fc1', num_hidden=100)
        net = mx.symbol.Activation(data=net, name='teacher_relu1', act_type="relu")
        net = mx.symbol.FullyConnected(data=net, name='teacher_fc2', num_hidden=1)
        net = mx.symbol.LinearRegressionOutput(data=net, name='teacher_output',
                                               grad_scale=teacher_noise_precision)
    else:
        net = mx.symbol.Variable('data')
        net = mx.symbol.FullyConnected(data=net, name='student_fc1', num_hidden=100)
        net = mx.symbol.Activation(data=net, name='student_relu1', act_type="relu")
        student_mean = mx.symbol.FullyConnected(data=net, name='student_mean', num_hidden=1)
        student_var = mx.symbol.FullyConnected(data=net, name='student_var', num_hidden=1)
        net = mx.symbol.Group([student_mean, student_var])
    return net

def dev():
    return mx.cpu()

def run_mnist_SGD():
    X, Y, X_test, Y_test = load_mnist()
    minibatch_size = 100
    net = get_mnist_sym()
    data_shape = (minibatch_size, ) + X.shape[1::]
    data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                   'softmax_label': nd.zeros((minibatch_size,), ctx=dev())}
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    exe, exe_params, _ = SGD(sym=net, dev=dev(), data_inputs=data_inputs, X=X, Y=Y,
                             X_test=X_test, Y_test=Y_test,
                             total_iter_num=1000000,
                             initializer=initializer,
                             lr=5E-6, prior_precision=1.0, minibatch_size=100)
#    sample_test_acc(exe, X=X_test, Y=Y_test, label_num=10, minibatch_size=100)

def run_mnist_SGLD():
    X, Y, X_test, Y_test = load_mnist()
    minibatch_size = 100
    net = get_mnist_sym()
    data_shape = (minibatch_size, ) + X.shape[1::]
    data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                   'softmax_label': nd.zeros((minibatch_size,), ctx=dev())}
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    exe, sample_pool = SGLD(sym=net, dev=dev(), data_inputs=data_inputs, X=X, Y=Y,
                            X_test=X_test, Y_test=Y_test,
                            total_iter_num=1000000,
                            initializer=initializer,
                            learning_rate=4E-6 , prior_precision=1.0, minibatch_size=100,
                            thin_interval=100, burn_in_iter_num=1000)
#    sample_test_acc(exe, sample_pool=sample_pool, X=X_test, Y=Y_test, label_num=10, minibatch_size=100)

def run_mnist_DistilledSGLD():
    X, Y, X_test, Y_test = load_mnist()
    minibatch_size = 100
    teacher_net = get_mnist_sym()
    crossentropy_softmax = CrossEntropySoftmax()
    student_net = get_mnist_sym(crossentropy_softmax)
#    logsoftmax = LogSoftmax()
#    student_net = get_mnist_sym(logsoftmax)
#    student_net = get_mnist_sym(mx.symbol.SoftmaxActivation)
    data_shape = (minibatch_size, ) + X.shape[1::]
    teacher_data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                   'softmax_label': nd.zeros((minibatch_size,), ctx=dev())}
    student_data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                   'softmax_label': nd.zeros((minibatch_size, 10), ctx=dev())}
#    student_data_inputs = {'data': nd.zeros(data_shape, ctx=dev())}
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    student_exe, student_params, _ = \
    DistilledSGLD(teacher_sym=teacher_net, student_sym=student_net,
                  teacher_data_inputs=teacher_data_inputs, student_data_inputs=student_data_inputs,
                  X=X, Y=Y, X_test=X_test, Y_test=Y_test, total_iter_num=1000000,
                  initializer=initializer,
                  teacher_learning_rate=4E-6, student_learning_rate=0.1,
                  student_lr_scheduler=mx.lr_scheduler.FactorScheduler(100000, 0.5),
#                  student_grad_f=classification_student_grad,
                  teacher_prior_precision=2.5, student_prior_precision=0.001,
                  perturb_deviation=0.001, minibatch_size=100, dev=dev())

def run_toy_SGLD():
    X, Y, X_test, Y_test = load_toy()
    minibatch_size = 1
    teacher_noise_precision = 1.0
    net = get_toy_sym(True, teacher_noise_precision)
    data_shape = (minibatch_size, ) + X.shape[1::]
    data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                   'teacher_output_label': nd.zeros((minibatch_size, 1), ctx=dev())}
    initializer = mx.init.Uniform(0.07)
    exe, params, _ = \
    SGLD(sym=net, data_inputs=data_inputs,
         X=X, Y=Y, X_test=X_test, Y_test=Y_test, total_iter_num=50000,
         initializer=initializer,
         learning_rate=1E-4,
#         lr_scheduler=mx.lr_scheduler.FactorScheduler(100000, 0.5),
         prior_precision=0.1,
         burn_in_iter_num=1000,
         thin_interval=10,
         task='regression',
         minibatch_size=minibatch_size, dev=dev())

def run_toy_DistilledSGLD():
    X, Y, X_test, Y_test = load_toy()
    minibatch_size = 1
    teacher_noise_precision = 1.0
    teacher_net = get_toy_sym(True, teacher_noise_precision)
    student_net = get_toy_sym(False)
    data_shape = (minibatch_size, ) + X.shape[1::]
    teacher_data_inputs = {'data': nd.zeros(data_shape, ctx=dev()),
                   'teacher_output_label': nd.zeros((minibatch_size, 1), ctx=dev())}
    student_data_inputs = {'data': nd.zeros(data_shape, ctx=dev())}
#                   'softmax_label': nd.zeros((minibatch_size, 10), ctx=dev())}
    initializer = mx.init.Uniform(0.07)
    student_grad_f = lambda student_outputs, teacher_pred : \
        regression_student_grad(student_outputs, teacher_pred, teacher_noise_precision)
    student_exe, student_params, _ = \
    DistilledSGLD(teacher_sym=teacher_net, student_sym=student_net,
                  teacher_data_inputs=teacher_data_inputs, student_data_inputs=student_data_inputs,
                  X=X, Y=Y, X_test=X_test, Y_test=Y_test, total_iter_num=80000,
                  initializer=initializer,
                  teacher_learning_rate=1E-4, student_learning_rate=0.01,
#                  teacher_lr_scheduler=mx.lr_scheduler.FactorScheduler(100000, 0.5),
                  student_lr_scheduler=mx.lr_scheduler.FactorScheduler(8000, 0.8),
                  student_grad_f=student_grad_f,
                  teacher_prior_precision=0.1, student_prior_precision=0.001,
                  perturb_deviation=0.1, minibatch_size=minibatch_size, task='regression', dev=dev())

def run_toy_HMC():
    X, Y, X_test, Y_test = load_toy()
    minibatch_size = Y.shape[0]
    teacher_noise_precision = 1.0



if __name__ == '__main__':
    numpy.random.seed(100)
    parser = argparse.ArgumentParser(description="MNIST classification example in the paper [NIPS2015]Bayesian Dark Knowledge")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--dataset", type=int, help="Type of algorithm to use. 0 --> TOY, 1 --> MNIST")
    parser.add_argument("-l", "--algorithm", type=int, help="Type of algorithm to use. 0 --> SGD, 1 --> SGLD, other-->DistilledSGLD")
    args = parser.parse_args()
    if args.dataset == 1:
        if 0 == args.algorithm:
            run_mnist_SGD()
        elif 1 == args.algorithm:
            run_mnist_SGLD()
        else:
            run_mnist_DistilledSGLD()
    else:
        print 123
        if 1 == args.algorithm:
            run_toy_SGLD()
        elif 2==args.algorithm:
            run_toy_DistilledSGLD()

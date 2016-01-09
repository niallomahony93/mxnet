import mxnet as mx
import mxnet.ndarray as nd
import numpy


class ExecutorBatchSizePool(object):
    def __init__(self, ctx, sym, data_shapes, params, params_grad, aux_states):
        self.ctx = ctx
        self.sym = sym
        self.params = params
        self.params_grad = params_grad
        self.aux_states = aux_states
        self.data_dims = {}
        self.init_batch_size = data_shapes.values()[0][0]
        for k, v in data_shapes.items():
            self.data_dims[k] = v[1::]
            assert self.init_batch_size == v[0]
        self.exe_pool = {}
        self.hits = {}
        self.get(self.init_batch_size)

    def get(self, batch_size=None):
        assert isinstance(batch_size, (int, long))
        if batch_size is None:
            batch_size = self.init_batch_size
        if batch_size in self.exe_pool:
            return self.exe_pool[batch_size]
        else:
            data_inputs = {k: mx.nd.empty((batch_size,) + s, ctx=self.ctx)
                           for k, s in self.data_dims.items()}
            exe = self.sym.bind(ctx=self.ctx, args=dict(self.params, **data_inputs),
                                args_grad=params_grad,
                                           aux_states=aux_states)
            self.exe_pool[batch_size] = exe
            return exe

def gradOut(q_output, target_reward, action, ctx=None):
    grad = nd.zeros(q_output.shape, ctx=ctx)
    grad[:] = nd.fill_element_0index(grad,
                                  nd.choose_element_0index(q_output, action) - target_reward,
                                  action)
    # print grad.asnumpy()
    return grad

def loss(q_output, target_reward, action):
    return nd.norm(nd.choose_element_0index(q_output, action) - target_reward).asnumpy()[0]/q_output.shape[0]


dev = mx.gpu()

action_num = 4
batch_size = 128
net = mx.symbol.Variable('data')
net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
net = mx.symbol.Flatten(data=net)
net = mx.symbol.FullyConnected(data=net, name='fc3', num_hidden=256)
net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=action_num)

data_shape = (batch_size, 4, 84, 84)
arg_names = net.list_arguments()
aux_names = net.list_auxiliary_states()
param_names = list(set(arg_names) - {'data'})
arg_shapes, output_shapes, aux_shapes = net.infer_shape(data=data_shape)
arg_name_shape = {k: s for k, s in zip(arg_names, arg_shapes)}

params = {n: nd.empty(arg_name_shape[n], ctx=dev) for n in param_names}
params_grad = {n: nd.empty(arg_name_shape[n], ctx=dev) for n in param_names}
aux_states = {k: nd.empty(s, ctx=dev) for k, s in zip(aux_names, aux_shapes)}
data_inputs = {'data': nd.empty(data_shape, ctx=dev)}


params_real = {n: mx.random.uniform(-0.07, 0.07, arg_name_shape[n], ctx=dev) for n in param_names}
executor_pool = ExecutorBatchSizePool(ctx=dev, sym=net, data_shapes={'data': data_shape}, params=params, params_grad=params_grad,
                                  aux_states=aux_states)
executor_pool_real = ExecutorBatchSizePool(ctx=dev, sym=net, data_shapes={'data': data_shape}, params=params_real, params_grad=None,
                                  aux_states=None)

init = mx.initializer.Uniform(0.07)
for k,v in params.items():
    init(k,v)


optimizer = mx.optimizer.create('sgd', learning_rate=0.01, momentum=0.9, lr_scheduler=mx.lr_scheduler.FactorScheduler(1000,0.1))
updater = mx.optimizer.get_updater(optimizer)
# monitor = mx.monitor.Monitor(interval=1000, stat_func=)

dat_dict = {}
action_dict = {}
target_reward_dict = {}
for i in range(100, 120):
    dat_dict[i] = nd.empty((i, 4, 84, 84), ctx=dev)
    action_dict[i] = nd.empty((i, ), ctx=dev)
    target_reward_dict[i] = nd.empty((i, ), ctx=dev)

for _ in range(1000):
    for i in range(100, 120):
        dat_dict[i][:] = mx.random.uniform(0, 1, (i, 4, 84, 84), ctx=dev)
        action_dict[i][:] = nd.array(numpy.random.randint(0, 4, (i, )).astype('float32'), ctx=dev)
        exe = executor_pool.get(i)
        exe_real = executor_pool_real.get(i)
        exe.arg_dict['data'][:] = dat_dict[i]
        exe_real.arg_dict['data'][:] = dat_dict[i]
        exe_real.forward(is_train=False)
        target_reward_dict[i][:] = nd.choose_element_0index(exe_real.outputs[0], action_dict[i])
        exe.forward(is_train=True)
        print loss(exe.outputs[0], target_reward_dict[i], action_dict[i])
        out_grad = gradOut(exe.outputs[0], target_reward_dict[i], action_dict[i], ctx=dev)
        exe.backward([out_grad])
        for k in params:
            updater(k, params_grad[k]/i, params[k])

nd.waitall()

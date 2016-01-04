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
            data_inputs = {k: mx.nd.empty((batch_size,) + s, ctx=self.ctx) for k, s in self.data_dims.items()}
            exe = self.sym.bind(ctx=self.ctx, args=dict(params, **data_inputs), args_grad=params_grad,
                                           aux_states=aux_states)
            self.exe_pool[batch_size] = exe
            return exe

def gradOut(q_output, target_reward, action, ctx=None):
    grad = nd.zeros(q_output.shape, ctx=ctx)
    slice_q = nd.choose_element_0index(q_output, action)


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

executor_pool = ExecutorBatchSizePool(ctx=dev, sym=net, data_shapes={'data': data_shape}, params=params, params_grad=params_grad,
                                  aux_states=aux_states)

init = mx.initializer.Uniform(0.07)
for k,v in params.items():
    init(k,v)

for i in range(10, 20):
    dat = numpy.random.uniform(0, 1, (i, 4, 84, 84))
    action = nd.array(numpy.mod(dat.sum((1, 2, 3)), 4).astype(int).astype('float32'), ctx=dev)
    encoded_action = nd.empty((i, 4), ctx=dev)
    nd.onehot_encode(action, encoded_action)

    print encode_action.asnumpy()
    exe = executor_pool.get(i)
    exe.arg_dict['data'][:] = dat
    exe.forward(is_train=True)
    print exe.outputs[0]
nd.waitall()

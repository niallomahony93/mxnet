import mxnet as mx
import mxnet.ndarray as nd
import numpy

def get_executor(sym, ctx, data_inputs, initializer=None):
    data_shapes = {k: v.shape for k, v in data_inputs.items()}
    arg_names = sym.list_arguments()
    aux_names = sym.list_auxiliary_states()
    param_names = list(set(arg_names) - set(data_inputs.keys()))
    arg_shapes, output_shapes, aux_shapes = sym.infer_shape(**data_shapes)
    arg_name_shape = {k: s for k, s in zip(arg_names, arg_shapes)}
    params = {n: nd.empty(arg_name_shape[n], ctx=ctx) for n in param_names}
    params_grad = {n: nd.empty(arg_name_shape[n], ctx=ctx) for n in param_names}
    aux_states = {k: nd.empty(s, ctx=ctx) for k, s in zip(aux_names, aux_shapes)}
    exe = sym.bind(ctx=ctx, args=dict(params, **data_inputs),
                   args_grad=params_grad,
                   aux_states=aux_states)
    if initializer != None:
        for k, v in params.items():
            initializer(k, v)
    return exe, params, params_grad, aux_states

def copy_param(exe):
    new_param = {k: nd.empty(v.shape, ctx=mx.cpu()) for k,v in exe.arg_dict.items()}
    for k, v in new_param.items():
        exe.arg_dict[k].copyto(v)
    return new_param

def sample_test_acc(exe, X, Y, sample_pool=None, label_num=None, minibatch_size=100):
    if label_num is None:
        pred = numpy.zeros((X.shape[0],)).astype('float32')
    else:
        pred = numpy.zeros((X.shape[0], label_num)).astype('float32')
    iter = mx.io.NDArrayIter(data=X, label=Y, batch_size=minibatch_size, shuffle=False)
    denominator = 0.0
    if sample_pool is None:
        curr_instance = 0
        iter.reset()
        for batch in iter:
            exe.arg_dict['data'][:] = batch.data[0]
            exe.forward(is_train=False)
            pred[curr_instance:curr_instance + minibatch_size - batch.pad, :] \
                += exe.outputs[0].asnumpy()
            curr_instance += minibatch_size - batch.pad
    else:
        for sample in sample_pool:
            if type(sample) is list:
                denominator += sample[0]
            else:
                denominator += 1.0
        for sample in sample_pool:
            if type(sample) is list:
                ratio = sample[0]/denominator
                param = sample[1]
            else:
                ratio = 1.0/denominator
                param = sample
            exe.copy_params_from(param)
            curr_instance = 0
            iter.reset()
            for batch in iter:
                exe.arg_dict['data'][:] = batch.data[0]
                exe.forward(is_train=False)
                pred[curr_instance:curr_instance + minibatch_size - batch.pad, :] \
                    += ratio * exe.outputs[0].asnumpy()
                curr_instance += minibatch_size - batch.pad
    correct = (pred.argmax(axis=1) == Y).sum()
    total = Y.shape[0]
    acc = correct/float(total)
    print 'corect=%d, total=%d, acc=%f' %(correct, total, acc)
    return acc


def sample_test_regression(exe, X, Y, sample_pool=None, minibatch_size=100):
    return

def pred_test(testing_data, exe, param_list=None, save_path=""):
    ret = numpy.zeros((testing_data.shape[0], 2))
    if param_list is None:
        for i in xrange(testing_data.shape[0]):
            exe.arg_dict['data'][:] = testing_data[i, 0]
            exe.forward(is_train=False)
            ret[i, 0] = exe.outputs[0].asnumpy()
            ret[i, 1] = numpy.exp(exe.outputs[1].asnumpy())
        numpy.savetxt(save_path, ret)
    else:
        for i in xrange(testing_data.shape[0]):
            pred = numpy.zeros((len(param_list),))
            for j in xrange(len(param_list)):
                exe.copy_params_from(param_list[j])
                exe.arg_dict['data'][:] = testing_data[i, 0]
                exe.forward(is_train=False)
                pred[j] = exe.outputs[0].asnumpy()
            ret[i, 0] = pred.mean()
            ret[i, 1] = pred.std()**2
        numpy.savetxt(save_path, ret)
    mse = numpy.square(ret[:, 0] - testing_data[:, 0] **3).mean()
    return mse, ret
import mxnet as mx
import mxnet.ndarray as nd
import numpy
import logging


class SGLDScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, begin_rate, end_rate, total_iter_num, factor):
        super(SGLDScheduler, self).__init__()
        if factor >= 1.0:
            raise ValueError("Factor must be less than 1 to make lr reduce")
        self.begin_rate = begin_rate
        self.end_rate = end_rate
        self.total_iter_num = total_iter_num
        self.factor = factor
        self.b = (total_iter_num - 1.0) / ((begin_rate / end_rate) ** (1.0 / factor) - 1.0)
        self.a = begin_rate / (self.b ** (-factor))
        self.count = 0

    def __call__(self, num_update):
        self.base_lr = self.a * ((self.b + num_update - 1) ** (-self.factor))
        self.count += 1
        logging.info("Update[%d]: Change learning rate to %0.5e",
                     num_update, self.base_lr)
        return self.base_lr


def teacher_grad(teacher_mean, target, teacher_noise_precision):
    return (teacher_mean - target) * teacher_noise_precision


def student_grad(student_mean, student_var, teacher_pred, teacher_noise_precision):
    grad_mean = nd.exp(-student_var) * (student_mean - teacher_pred)

    grad_var = (1 - nd.exp(-student_var) * (nd.square(student_mean - teacher_pred)
                                                  + 1 / teacher_noise_precision))/2
    return [grad_mean, grad_var]


def student_loss(student_mean, student_var, teacher_pred, teacher_noise_precision):
    return (0.5 * (student_var + nd.exp(-student_var) * (nd.square(teacher_pred - student_mean)
                                                         + 1 / teacher_noise_precision))).asnumpy()[
        0]


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
    new_param = {k: nd.empty(v.shape, ctx=v.context) for k,v in exe.arg_dict.items()}
    for k, v in new_param.items():
        exe.arg_dict[k].copyto(v)
    return new_param

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

dev = mx.cpu()

teacher = mx.symbol.Variable('data')
teacher = mx.symbol.FullyConnected(data=teacher, name='teacher_fc1', num_hidden=100)
teacher = mx.symbol.Activation(data=teacher, name='teacher_relu1', act_type="relu")
teacher = mx.symbol.FullyConnected(data=teacher, name='teacher_pred', num_hidden=1)

student = mx.symbol.Variable('data')
student = mx.symbol.FullyConnected(data=student, name='student_fc1', num_hidden=100)
student = mx.symbol.Activation(data=student, name='student_relu1', act_type="relu")
student_mean = mx.symbol.FullyConnected(data=student, name='student_mean', num_hidden=1)
student_var = mx.symbol.FullyConnected(data=student, name='student_var', num_hidden=1)
student = mx.symbol.Group([student_mean, student_var])

batch_size = 1
data_shape = (batch_size, 1)
data_inputs = {'data': nd.empty(data_shape, ctx=dev)}

initializer = mx.initializer.Uniform(0.07)
teacher_exe, teacher_params, teacher_params_grad, _ = get_executor(teacher, dev, data_inputs,
                                                                   initializer)
student_exe, student_params, student_params_grad, _ = get_executor(student, dev, data_inputs,
                                                                   initializer)

#X = numpy.random.uniform(-4, 4, (20, 1))
#Y = X * X * X + numpy.random.normal(0, 3, (20, 1))

training_data = numpy.loadtxt('toy_data_train.txt')
testing_data = numpy.loadtxt('toy_data_test.txt')

X = training_data[:, 0].reshape((20, 1))
Y = training_data[:, 1].reshape((20, 1))

total_iter_num = 100000
thinning_interval = 1
sample_num = 2000
burn_in = 0
teacher_prior_precision = 0.1
teacher_noise_precision = 1
student_prior_precision = 0.001

scheduler = SGLDScheduler(begin_rate=1e-4, end_rate=1e-5, total_iter_num=total_iter_num,
                          factor=0.55)
teacher_optimizer = mx.optimizer.create('sgld', learning_rate=1e-4,
                                        rescale_grad=X.shape[0] / float(batch_size),
                                        lr_scheduler=mx.lr_scheduler.FactorScheduler(80000, 0.8),
                                        wd=teacher_prior_precision)
student_optimizer = mx.optimizer.create('sgd', learning_rate=0.01,
                                        rescale_grad=1.0 / batch_size,
                                        lr_scheduler=mx.lr_scheduler.FactorScheduler(5000, 0.9),
                                        wd=student_prior_precision)
teacher_updater = mx.optimizer.get_updater(teacher_optimizer)
student_updater = mx.optimizer.get_updater(student_optimizer)

sgld_sample_list = []

for i in xrange(total_iter_num):
    if i%10000 ==0:
        print 'Iter:', i
    ind = numpy.random.randint(X.shape[0], size=batch_size)
    X_batch = X[ind, :]
    Y_batch = Y[ind, :]
    teacher_exe.arg_dict['data'][:] = X_batch
    teacher_exe.forward(is_train=True)
    teacher_exe.outputs[0].wait_to_read()
    # print 'Teacher Loss:', numpy.linalg.norm(teacher_exe.outputs[0].asnumpy() - Y_batch)
    teacher_exe.backward([teacher_grad(teacher_exe.outputs[0], nd.array(Y_batch, ctx=dev),
                                       teacher_noise_precision)])

    for k in teacher_params:
        teacher_updater(k, teacher_params_grad[k], teacher_params[k])
        # print k, teacher_params_grad[k].asnumpy()
        # ch = raw_input()

    if i >= burn_in:
        if 0 == i%thinning_interval:
            if (i+1) % (total_iter_num/sample_num) == 0:
                sgld_sample_list.append(copy_param(teacher_exe))
            # print student_exe.grad_arrays
            # print student_params
            # print student_params_grad
            # ch = raw_input()
            X_student_batch = X_batch + numpy.random.normal(0, 0.05, X_batch.shape)
            teacher_exe.arg_dict['data'][:] = X_student_batch
            teacher_exe.forward(is_train=False)
            teacher_exe.outputs[0].wait_to_read()
            teacher_pred = teacher_exe.outputs[0]
            student_exe.arg_dict['data'][:] = X_student_batch
            student_exe.forward(is_train=True)
            print numpy.hstack((X_batch*X_batch*X_batch, teacher_exe.outputs[0].asnumpy(), student_exe.outputs[0].asnumpy(), nd.exp(student_exe.outputs[1]).asnumpy()))
            print 'Student Loss:', student_loss(student_exe.outputs[0], student_exe.outputs[1],
                                                teacher_pred, teacher_noise_precision)
            student_exe.backward(student_grad(student_exe.outputs[0], student_exe.outputs[1],
                                              teacher_pred, teacher_noise_precision))
            for k in student_params:
                student_updater(k, student_params_grad[k], student_params[k])


distilled_sgld_mse, distilled_sgld_ret = \
    pred_test(testing_data=testing_data, exe=student_exe, save_path='toy-1d-distilled-sgld.txt')

sgld_mse, sgld_ret = \
    pred_test(testing_data=testing_data, exe=teacher_exe, param_list=sgld_sample_list,
              save_path='toy-1d-sgld.txt')

print 'Distilled SGLD MSE', distilled_sgld_mse
print 'SGLD MSE', distilled_sgld_mse
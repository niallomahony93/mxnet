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
        self.b = (total_iter_num - 1) / ((begin_rate / end_rate) ** (1 / factor) - 1)
        self.a = begin_rate / (self.b ** (-factor))
        self.count = 0

    def __call__(self, num_update):
        self.base_lr = self.a * (self.b + self.count) ** (-self.factor)
        self.count += 1
        logging.info("Update[%d]: Change learning rate to %0.5e",
                     num_update, self.base_lr)
        return self.base_lr


def teacher_grad(teacher_mean, target, teacher_noise_precision):
    return (teacher_mean - target) / teacher_noise_precision


def student_grad(student_mean, student_var, teacher_pred, teacher_noise_precision):
    grad_mean = nd.exp(-student_var) * (student_mean - teacher_pred)
    grad_var = 0.5 * (1 - nd.exp(-student_var) * (nd.square(student_mean - teacher_pred)
                                                  + 1 / teacher_noise_precision))
    return grad_mean, grad_var


def student_loss(student_mean, student_var, teacher_pred, teacher_noise_precision):
    return (0.5 * (student_var + nd.exp(-student_var) * (nd.square(teacher_pred - student_mean)
                                                        + 1 / teacher_noise_precision))).asnumpy()


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
            v.init(k, v)
    return exe, params, params_grad, aux_states


dev = mx.gpu()

teacher = mx.symbol.Variable('data')
teacher = mx.symbol.FullyConnected(data=teacher, name='teacher_fc1', num_hidden=10)
teacher = mx.symbol.Activation(data=teacher, name='teacher_relu1', act_type="relu")
teacher = mx.symbol.FullyConnected(data=teacher, name='teacher_pred', num_hidden=1)

student = mx.symbol.Variable('data')
student = mx.symbol.FullyConnected(data=student, name='student_fc1', num_hidden=10)
student = mx.symbol.Activation(data=student, name='teacher_relu1', act_type="relu")
student_mean = mx.symbol.FullyConnected(data=student, name='student_mean', num_hidden=1)
student_var = mx.symbol.FullyConnected(data=student, name='student_var', num_hidden=1)
student = mx.symbol.Group([student_mean, student_var])

batch_size = 1
data_shape = (batch_size, 1)
data_inputs = {'data': nd.empty(data_shape, ctx=dev)}

initializer = mx.initializer.Uniform(0.07)
teacher_exe, teacher_params, teacher_params_grad, _ = get_executor(teacher, dev, data_inputs)
student_exe, student_params, student_params_grad, _ = get_executor(student, dev, data_inputs)

X = numpy.random.normal(0, 3, (20, 1))
Y = X * X * X + numpy.random.normal(0, 3, (20, 1))

total_iter_num = 1000000
burn_in = 1000
learning_rate = 0.01
teacher_prior_precision = 2.5
teacher_noise_precision = 1.25
student_prior_precision = 0.001

scheduler = SGLDScheduler(begin_rate=1e-4, end_rate=1e-8, total_iter_num=total_iter_num, factor=0.55)
teacher_optimizer = mx.optimizer.create('sgld', learning_rate=1e-5,
                                        rescale_grad=X.shape[1] / float(batch_size),
                                        lr_scheduler=mx.lr_scheduler.FactorScheduler(80000, 0.5),
                                        wd=1.0 / (2 * teacher_prior_precision))
student_optimizer = mx.optimizer.create('sgd', learning_rate=0.01,
                                        rescale_grad=1.0 / batch_size,
                                        lr_scheduler=mx.lr_scheduler.FactorScheduler(5000, 0.8),
                                        wd=1.0 / (2 * student_prior_precision))
teacher_updater = mx.optimizer.get_updater(teacher_optimizer)
student_updater = mx.optimizer.get_updater(student_optimizer)

for i in xrange(total_iter_num):
    ind = numpy.random.randint(X.shape[0], size=batch_size)
    X_batch = X[ind, :]
    Y_batch = Y[ind, :]
    teacher_exe.arg_dict['data'] = X_batch
    teacher_exe.forward(is_train=True)
    teacher_exe.outputs[0].wait_to_read()
    print 'Teacher Loss:', numpy.linalg.norm(teacher_exe.outputs[0].asnumpy() - Y_batch)
    teacher_exe.backward(teacher_grad(teacher_exe.outputs[0], nd.array(Y_batch, ctx=dev),
                                      teacher_noise_precision))
    for k in teacher_params:
        teacher_updater(k, teacher_params_grad[k], teacher_params[k])
    if i >= burn_in:
        X_student_batch = X_batch + numpy.random.normal(0, 0.05, X_batch.shape)
        teacher_exe.arg_dict['data'] = X_student_batch
        teacher_exe.forward(is_train=False)
        teacher_exe.outputs[0].wait_to_read()
        teacher_pred = teacher_exe.outputs[0]
        student_exe.forward(is_train=True)
        student_exe.outputs[0].wait_to_read()
        student_exe.outputs[1].wait_to_read()
        print 'Student Loss:', student_loss(student_exe.outputs[0], student_exe.outputs[1],
                                          teacher_pred, teacher_noise_precision)
        student_exe.backward(student_grad(student_exe.outputs[0], student_exe.outputs[1],
                                          teacher_pred, teacher_noise_precision))

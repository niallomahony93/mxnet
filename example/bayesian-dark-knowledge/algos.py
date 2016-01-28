import mxnet as mx
import mxnet.ndarray as nd
import time
import logging
from utils import *


def calc_potential(exe, params, X, Y, prior_precision):
    exe.copy_params_from(params)
    exe.arg_dict['data'][:] = X
    exe.forward(is_train=False)
    ret =0.0
    ret += (nd.norm(exe.outputs[0] - Y).asscalar() ** 2)/2.0
    for k in exe.arg_dict:
        ret += (nd.norm(exe.arg_dict[k]).asscalar() ** 2)/2.0 * prior_precision
    return ret

def calc_grad(exe, params, X, Y, grad_holder):
    exe.copy_params_from(params)
    exe.arg_dict['data'][:] = X
    exe.forward(is_train=True)
    exe.backward()


def step_HMC(init_states, data, exe, calc_potential, calc_grad, L=10, eps=1E-6):
    assert type(init_states) is dict
    grads = {k: nd.empty(v.shape, v.context) for k, v in init_states.items()}
    end_states = {k: v.copyto(v.context) for k, v in init_states.items()}
    init_momentums = {k: mx.random.normal(0, 1, v.shape) for k,v in init_states.items()}
    end_momentums = {k: v.copyto(v.context) for k, v in init_momentums}

    #1. Make a half step for momentum at the beginning
    for k, momentum in end_momentums.items():
        calc_grad(exe, end_states, data, grads)
        momentum[:] = momentum - (eps/2) * grads[k]
    #2. Alternate full steps for position and momentum
    for i in range(L):
        #2.1 Full step for position
        for k, state in end_states.items():
            state[:] = state + eps * end_momentums[k]
        #2.2 Full step for the momentum, except at the end of trajectory we perform a half step
        if i != L-1:
            calc_grad(end_states, data, grads)
            for k, momentum in end_momentums.items():
                momentum[:] = momentum - eps * grads[k]
        else:
            calc_grad(end_states, data, grads)
            for k, momentum in end_momentums.items():
                # We should reverse the sign of the momentum at the end
                momentum[:] = -(momentum - eps / 2.0 * grads[k])
    #3. Calculate acceptance ratio and accept/reject the move
    init_potential = calc_potential(init_states, data)
    init_kinetic = sum([nd.sum(nd.square(momentum)) / 2.0
                        for momentum in init_momentums.values()]).asscalar()
    end_potential = calc_potential(end_states, data)
    end_kinetic = sum([nd.sum(nd.square(momentum)) / 2.0
                       for momentum in end_momentums.values()]).asscalar()
    r = numpy.random.rand(1)
    if r < numpy.exp(-(end_potential + end_kinetic) + (init_potential + init_kinetic)):
        return end_states, 1
    else:
        return init_states, 0


def HMC(sym, data_inputs, X, Y, X_test, Y_test, sample_num,
        initializer=None,
        learning_rate=1E-6, L=10, dev=mx.gpu()):
    exe, params, params_grad, _ = get_executor(sym, dev, data_inputs, initializer)
    #for i in xrange(sample_num):
        #sample, is_accept = step_HMC(params, [X,Y],  )

    #return sample_pool

def SGD(sym, data_inputs, X, Y, X_test, Y_test, total_iter_num,
        lr=None,
        lr_scheduler=None, prior_precision=1,
        out_grad_f=None,
        initializer = None,
        minibatch_size=100, dev=mx.gpu()):
    if out_grad_f is None:
        label_key = list(set(data_inputs.keys()) - set('data'))[0]
    exe, params, params_grad, _ = get_executor(sym, dev, data_inputs, initializer)
#    print params
    optimizer = mx.optimizer.create('sgd', learning_rate=lr,
                                    rescale_grad=X.shape[0]/minibatch_size,
                                    lr_scheduler=lr_scheduler,
                                    wd=prior_precision,
                                    arg_names=params.keys())
    updater = mx.optimizer.get_updater(optimizer)
    for i in xrange(total_iter_num):
        indices = numpy.random.randint(X.shape[0], size=minibatch_size)
        X_batch = X[indices]
        Y_batch = Y[indices]
        exe.arg_dict['data'][:] = X_batch
        if out_grad_f is None:
            exe.arg_dict[label_key][:] = Y_batch
            exe.forward(is_train=True)
            exe.backward()
        else:
            exe.forward(is_train=True)
            exe.backward(out_grad_f(exe.outputs, nd.array(Y_batch, ctx=dev)))
        for k in params:
            updater(k, params_grad[k], params[k])
        if (i + 1) % 1000 == 0:
            print "Current Iter Num: %d" %(i+1)
            print -numpy.log(exe.outputs[0].asnumpy()[numpy.arange(minibatch_size), Y_batch]).sum()
            sample_test_acc(exe, X=X_test, Y=Y_test, label_num=10, minibatch_size=100)
    return exe, params, params_grad

def SGLD(sym, X, Y, X_test, Y_test, total_iter_num,
         data_inputs=None,
         learning_rate=None,
         lr_scheduler=None, prior_precision=1,
         out_grad_f=None,
         initializer=None,
         minibatch_size=100, thin_interval=100, burn_in_iter_num=1000, dev=mx.gpu()):
    if out_grad_f is None:
        label_key = list(set(data_inputs.keys()) - set('data'))[0]
    exe, params, params_grad, _ = get_executor(sym, dev, data_inputs, initializer)
    optimizer = mx.optimizer.create('sgld', learning_rate=learning_rate,
                                    rescale_grad= X.shape[0]/minibatch_size,
                                    lr_scheduler=lr_scheduler,
                                    wd=prior_precision)
    updater = mx.optimizer.get_updater(optimizer)
    sample_pool = []
    for i in xrange(total_iter_num):
        indices = numpy.random.randint(X.shape[0], size=minibatch_size)
        X_batch = X[indices]
        Y_batch = Y[indices]
        exe.arg_dict['data'][:] = X_batch
        if out_grad_f is None:
            exe.arg_dict[label_key][:] = Y_batch
            exe.forward(is_train=True)
            exe.backward()
        else:
            exe.forward(is_train=True)
            exe.backward(out_grad_f(exe.outputs, nd.array(Y_batch, ctx=dev)))
        for k in params:
            updater(k, params_grad[k], params[k])
        if i < burn_in_iter_num:
            continue
        else:
            if 0 == (i - burn_in_iter_num) % thin_interval:
                if optimizer.lr_scheduler is not None:
                    lr = optimizer.lr_scheduler(optimizer.num_update)
                else:
                    lr = learning_rate
                sample_pool.append([lr, copy_param(exe)])
        if (i + 1) % 100000 == 0:
            print "Current Iter Num: %d" %(i+1)
            print -numpy.log(exe.outputs[0].asnumpy()[numpy.arange(minibatch_size), Y_batch]).sum()
            sample_test_acc(exe, sample_pool=sample_pool, X=X_test, Y=Y_test, label_num=10, minibatch_size=100)
    return exe, sample_pool

def DistilledSGLD(teacher_sym, student_sym,
                  teacher_data_inputs, student_data_inputs,
                  X, Y, X_test, Y_test, total_iter_num,
                  teacher_learning_rate, student_learning_rate,
                  teacher_lr_scheduler=None, student_lr_scheduler=None,
                  teacher_grad_f=None, student_grad_f=None,
                  teacher_prior_precision=1, student_prior_precision=0.001,
                  perturb_deviation=0.001,
                  initializer=None,
                  minibatch_size=100, dev=mx.gpu()):
    teacher_exe, teacher_params, teacher_params_grad, _ = \
        get_executor(teacher_sym, dev, teacher_data_inputs, initializer)
    student_exe, student_params, student_params_grad, _ = \
        get_executor(student_sym, dev, student_data_inputs, initializer)
    if teacher_grad_f is None:
        teacher_label_key = list(set(teacher_data_inputs.keys()) - set('data'))[0]
    if student_grad_f is None:
        student_label_key = list(set(student_data_inputs.keys()) - set('data'))[0]
    teacher_optimizer = mx.optimizer.create('sgld',
                                    learning_rate=teacher_learning_rate,
                                    rescale_grad=X.shape[0] / float(minibatch_size),
                                    lr_scheduler=teacher_lr_scheduler,
                                    wd=teacher_prior_precision)
    student_optimizer = mx.optimizer.create('sgd',
                                    learning_rate=student_learning_rate,
                                    rescale_grad=1.0/float(minibatch_size),
                                    lr_scheduler=student_lr_scheduler,
                                    wd=student_prior_precision)
    teacher_updater = mx.optimizer.get_updater(teacher_optimizer)
    student_updater = mx.optimizer.get_updater(student_optimizer)

    for epoch in xrange(total_iter_num/(X.shape[0]/minibatch_size)):
        iterator = mx.io.NDArrayIter(data=X, label=Y, batch_size=minibatch_size, shuffle=True)
#        print "Epoch %d" %epoch ,
        start = time.time()
        for batch in iterator:
            #1.1 Draw random minibatch
            X_batch = batch.data[0]
            Y_batch = batch.label[0]

            #1.2 Update teacher
            teacher_exe.arg_dict['data'][:] = X_batch
            if teacher_grad_f is None:
                teacher_exe.arg_dict[teacher_label_key][:] = Y_batch
                teacher_exe.forward(is_train=True)
                teacher_exe.backward()
            else:
                teacher_exe.forward(is_train=True)
                teacher_exe.backward(teacher_grad_f(teacher_exe.outputs, nd.array(Y_batch, ctx=dev)))

            for k in teacher_params:
                teacher_updater(k, teacher_params_grad[k], teacher_params[k])

            #2.1 Draw random minibatch and do random perturbation
            X_student_batch = X_batch + mx.random.normal(0, perturb_deviation, X_batch.shape, mx.cpu())
            #nd.waitall()
            #2.2 Get teacher predictions
            teacher_exe.arg_dict['data'][:] = X_student_batch
            teacher_exe.forward(is_train=False)
            teacher_pred = teacher_exe.outputs[0]
            teacher_pred.wait_to_read()
            #2.3 Update student
            student_exe.arg_dict['data'][:] = X_student_batch
            if student_grad_f is None:
                student_exe.arg_dict[student_label_key][:] = teacher_pred
                student_exe.forward(is_train=True)
                student_exe.backward()
            else:
                student_exe.forward(is_train=True)
                student_exe.backward(student_grad_f(student_exe.outputs, teacher_pred))
            for k in student_params:
                student_updater(k, student_params_grad[k], student_params[k])
        end = time.time()
#        print "Time Spent: %f" %(end-start)
        if (epoch + 1) % 2 == 0:
            print "Current Epoch Num: %d" %(epoch+1)
            sample_test_acc(student_exe, X=X_test, Y=Y_test, label_num=10, minibatch_size=100)

    return student_exe, student_params, student_params_grad
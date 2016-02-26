import mxnet as mx
import numpy
from defaults import *
from utils import *
import logging

logger = logging.getLogger(__name__)

#TODO Support RNN for sym
class Critic(object):
    """Critic, Differentiable Approximator for Q(s, a) or V(s)

    Parameters
    ----------
    state_dim : tuple
        Indicates the size of the `s` input
        E.g, (4, 84, 84) means the state has 4 channels, 84 rows/cols
    action_dim: tuple/int
        1. typ='discrete':
            `action_dim` means the number of possible actions.
        2. typ='continuous':
            `action_dim` indicates the size of the `a` input.
    typ: str, optional
        Indicates whether the action we can perform is 'discrete' or 'continuous'
        Default is 'discrete'
    """
    def __init__(self, data_shapes, sym, params=None, params_grad=None, aux_states=None,
                 initializer=mx.init.Uniform(0.07), ctx=mx.gpu(),
                 optimizer_param_dict=None, name='CriticNet'):
        self.sym = sym
        self.ctx = ctx
        self.data_shapes = data_shapes.copy()
        self.name = name
        self.optimizer_param_dict = optimizer_param_dict.copy()
        if optimizer_param_dict is not None:
            #TODO We may need to change here for distributed setting
            self.optimizer = mx.optimizer.create(**optimizer_param_dict)
            self.updater = mx.optimizer.get_updater(self.optimizer)
        else:
            self.optimizer = None
            self.updater = None
        self.initializer=initializer
        if (params is None and params_grad is None and aux_states is None):
            assert initializer is not None, 'You must set the initializer if you donnot give the ' \
                                            'initial params!'
            arg_names = sym.list_arguments()
            aux_names = sym.list_auxiliary_states()
            param_names = list(set(arg_names) - set(data_shapes.keys()))
            arg_shapes, output_shapes, aux_shapes = sym.infer_shape(data=data_shapes)
            self.arg_name_shape = {k: s for k, s in zip(arg_names, arg_shapes)}
            self.params = {n: nd.empty(self.arg_name_shape[n], ctx=ctx) for n in param_names}
            self.params_grad = {n: nd.empty(self.arg_name_shape[n], ctx=ctx) for n in param_names}
            self.aux_states = {k: nd.empty(s, ctx=ctx) for k, s in zip(aux_names, aux_shapes)}
            for k, v in self.params.items():
                initializer(k, v)
        else:
            self.arg_name_shape = dict(data_shapes.items() + [(k, v.shape) for k, v in params.items()])
            self.params = params
            self.params_grad = params_grad
            self.aux_states = aux_states
        self.executor_pool = ExecutorBatchSizePool(ctx=self.ctx, sym=self.sym,
                                                   data_shapes=self.data_shapes,
                                                   params=self.params, params_grad=self.params_grad,
                                                   aux_states=self.aux_states)
    """
    Compute the Q(s,a) or V(s) score
    """
    def score(self, batch_size, **input_dict):
        exe = self.executor_pool.get(batch_size)
        for k,v in input_dict:
            exe.arg_dict[k][:] = v
        exe.forward(is_train=False)
        return exe.outputs

    def fit_target(self, batch_size, **input_dict):
        assert self.updater is not None, "Updater not set!"
        exe = self.executor_pool.get(batch_size)
        for k,v in input_dict:
            exe.arg_dict[k][:] = v
        exe.forward(is_train=True)
        exe.backward()
        for k in self.params:
            self.updater(k, self.params[k], self.params_grad[k])

    def copyto(self, name=None, ctx=None):
        if ctx is None:
            ctx = self.ctx
        data_shapes = self.data_shapes.copy()
        sym = self.sym
        params = {k: v.copyto(ctx) for k, v in self.params.items()}
        params_grad = {k: v.copyto(ctx) for k, v in self.params_grad.items()}
        aux_states = None if self.aux_states is None else \
            {k: v.copyto(ctx) for k, v in self.aux_states.items()}
        optimizer_param_dict = self.optimizer_param_dict.copy()
        if name is None:
            name = self.name + '-copy-' + str(ctx)
        critic = Critic(data_shapes=data_shapes, sym=sym, params=params, params_grad=params_grad,
                        aux_states=aux_states, ctx=ctx, optimizer_param_dict=optimizer_param_dict,
                                                                            name=name)
        return critic

    def print_stat(self):
        logging.info("Name: %s" %self.name)
        assert self.params is not None, "You must set the "
        logging.info("Params: " + ' '.join(["%s:%s" %(str(k), str(v.shape)) for k, v in self.params.items()]))
        logging.info("Aux States: ")
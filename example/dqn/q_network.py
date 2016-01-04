import mxnet as mx
import numpy
from defaults import *
import logging

logger = logging.getLogger(__name__)


class QNetwork(object):
    """QNetwork, Differentiable Approximator for Q(s, a)

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
    def __init__(self, state_dim, action_dim, sym, typ='discrete'):
        self.sym = sym
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.typ = typ
        self.exe_batch_size_dict = {}

    def score(self, state, action=None):
        if action is None:
            assert 'discrete' == self.typ
        batch_size = state.shape[0]

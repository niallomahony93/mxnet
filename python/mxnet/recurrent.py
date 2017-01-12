from __future__ import absolute_import, division, print_function
import numbers
import numpy as np
from collections import namedtuple
from . import symbol

LSTMState = namedtuple("LSTMState", ["c", "h"])
RNNState = namedtuple("RNNState", ["h"])

def _get_sym_list(syms, default_names=None, default_shapes=None):
    if syms is None and default_names is not None:
        if default_shapes is not None:
            return [symbol.Variable(name=name, shape=shape) for (name, shape)
                    in zip(default_names, default_shapes)]
        else:
            return [symbol.Variable(name=name) for name in default_names]
    assert isinstance(syms, (list, tuple, symbol.Symbol))
    if isinstance(syms, (list, tuple)):
        if default_names is not None and len(syms) != len(default_names):
            raise ValueError("Size of symbols do not match expectation. Received %d, Expected %d. "
                             "syms=%s, names=%s" %(len(syms), len(default_names),
                                                   str(list(sym.name for sym in syms)),
                                                   str(default_names)))
        return list(syms)
    else:
        if default_names is not None and len(default_names) != 1:
            raise ValueError("Size of symbols do not match expectation. Received 1, Expected %d. "
                             "syms=%s, names=%s"
                             % (len(default_names), str([syms.name]), str(default_names)))
        return [syms]


def _get_numeric_list(values, typ, expected_len=None):
    if isinstance(values, numbers.Number):
        if expected_len is not None:
            return [typ(values)] * expected_len
        else:
            return [typ(values)]
    elif isinstance(values, (list, tuple)):
        if expected_len is not None:
            assert len(values) == expected_len
        try:
            ret = [typ(value) for value in values]
            return ret
        except(ValueError):
            print("Need iterable with numeric elements, received: %s" %str(values))
            import sys
            sys.exit(1)
    else:
        raise ValueError("Unaccepted value type, values=%s" %str(values))


def _get_int_list(values, expected_len=None):
    return _get_numeric_list(values, np.int32, expected_len)


def _get_float_list(values, expected_len=None):
    return _get_numeric_list(values, np.float32, expected_len)


def step_vanilla_rnn(num_hidden, data, prev_h, act_f,
                     i2h_weight, i2h_bias, h2h_weight, h2h_bias, dropout, seq_len):

    data = symbol.Reshape(data, shape=(-1, 0), reverse=True)
    if dropout > 0.:
        data = symbol.Dropout(data=data, p=dropout)
    i2h = symbol.FullyConnected(data=data,
                                weight=i2h_weight,
                                bias=i2h_bias,
                                num_hidden=num_hidden)
    i2h = symbol.SliceChannel(i2h, num_outputs=seq_len, axis=0)
    all_h = []
    for i in range(seq_len):
        h2h = symbol.FullyConnected(data=prev_h,
                                    weight=h2h_weight,
                                    bias=h2h_bias,
                                    num_hidden=num_hidden)
        new_h = act_f(i2h[i] + h2h)
        all_h.append(new_h)
        prev_h = new_h
    return symbol.Reshape(symbol.Concat(*all_h, num_args=len(all_h), dim=0),
                          shape=(seq_len, -1, 0), reverse=True), all_h[-1]


def step_relu_rnn(num_hidden, data, prev_h, i2h_weight, i2h_bias, h2h_weight, h2h_bias,
                  dropout=0., seq_len=1, name="relu_rnn"):
    return step_vanilla_rnn(num_hidden=num_hidden, data=data, prev_h=prev_h,
                            act_f=lambda x: symbol.Activation(x, act_type="relu"),
                            i2h_weight=i2h_weight, i2h_bias=i2h_bias,
                            h2h_weight=h2h_weight, h2h_bias=h2h_bias,
                            dropout=dropout, seq_len=seq_len, name=name)


def step_tanh_rnn(num_hidden, data, prev_h, i2h_weight, i2h_bias, h2h_weight, h2h_bias,
                  dropout=0., seq_len=1, name="tanh_rnn"):
    return step_vanilla_rnn(num_hidden=num_hidden, data=data, prev_h=prev_h,
                            act_f=lambda x: symbol.Activation(x, act_type="tanh"),
                            i2h_weight=i2h_weight, i2h_bias=i2h_bias,
                            h2h_weight=h2h_weight, h2h_bias=h2h_bias,
                            dropout=dropout, seq_len=seq_len, name=name)


def step_lstm(num_hidden, data, prev_h, prev_c, i2h_weight, i2h_bias, h2h_weight, h2h_bias,
              dropout=0., seq_len=1, name="lstm"):
    data = symbol.Reshape(data, shape=(-1, 0), reverse=True)
    if dropout > 0.:
        data = symbol.Dropout(data=data, p=dropout, name=name + ":dropout")
    i2h = symbol.FullyConnected(data=data,
                                weight=i2h_weight,
                                bias=i2h_bias,
                                num_hidden=num_hidden * 4)
    i2h = symbol.SliceChannel(i2h, num_outputs=seq_len, axis=0, name=name + ":i2h")
    all_c = []
    all_h = []
    for i in range(seq_len):
        h2h = symbol.FullyConnected(data=prev_h,
                                    weight=h2h_weight,
                                    bias=h2h_bias,
                                    num_hidden=num_hidden * 4,
                                    name=name + ":h2h")
        gates = i2h[i] + h2h
        slice_gates = symbol.SliceChannel(gates, num_outputs=4, axis=1)
        input_gate = symbol.Activation(slice_gates[0], act_type="sigmoid",
                                       name=name + ":gi")
        forget_gate = symbol.Activation(slice_gates[1], act_type="sigmoid",
                                        name=name + ":gf")
        new_mem = symbol.Activation(slice_gates[2], act_type="tanh",
                                    name=name + ":new_mem")
        out_gate = symbol.Activation(slice_gates[3], act_type="sigmoid",
                                     name=name + ":go")
        new_c = forget_gate * prev_c + input_gate * new_mem
        new_h = out_gate * symbol.Activation(new_c, act_type="tanh")
        all_h.append(new_h)
        all_c.append(new_c)
        prev_h = new_h
        prev_c = new_c
    return symbol.Reshape(symbol.Concat(*all_h, num_args=len(all_h), dim=0),
                          shape=(seq_len, -1, 0), reverse=True),\
           symbol.Reshape(symbol.Concat(*all_c, num_args=len(all_c), dim=0),
                   shape=(seq_len, -1, 0), reverse=True),\
           all_h[-1], all_c[-1]


def step_gru(num_hidden, data, prev_h, i2h_weight, i2h_bias, h2h_weight, h2h_bias,
             dropout=0., seq_len=1, name="gru"):
    data = symbol.Reshape(data, shape=(-1, 0), reverse=True)
    if dropout > 0.:
        data = symbol.Dropout(data=data, p=dropout, name=name + ":dropout")
    i2h = symbol.FullyConnected(data=data,
                                weight=i2h_weight,
                                bias=i2h_bias,
                                num_hidden=num_hidden * 3,
                                name=name + ":i2h")
    i2h = symbol.SliceChannel(i2h, num_outputs=seq_len, axis=0, name=name + ":i2h")
    all_h = []
    for i in range(seq_len):
        h2h = symbol.FullyConnected(data=prev_h,
                                    weight=h2h_weight,
                                    bias=h2h_bias,
                                    num_hidden=num_hidden * 3,
                                    name=name + ":h2h")
        i2h_slice = symbol.SliceChannel(i2h[i], num_outputs=3, axis=1)
        h2h_slice = symbol.SliceChannel(h2h, num_outputs=3, axis=1)
        reset_gate = symbol.Activation(i2h_slice[0] + h2h_slice[0], act_type="sigmoid",
                                       name=name + ":gr")
        update_gate = symbol.Activation(i2h_slice[1] + h2h_slice[1], act_type="sigmoid",
                                        name=name + ":gu")
        new_mem = symbol.Activation(i2h_slice[2] + reset_gate * h2h_slice[2], act_type="tanh",
                                    name=name + ":new_mem")
        new_h = update_gate * prev_h + (1 - update_gate) * new_mem
        all_h.append(new_h)
        prev_h = new_h
    return symbol.Reshape(symbol.Concat(*all_h, num_args=len(all_h), dim=0),
                          shape=(seq_len, -1, 0), reverse=True), all_h[-1]


def get_rnn_param_shapes(num_hidden, data_dim, typ):
    """

    Parameters
    ----------
    num_hidden
    data_dim
    typ

    Returns
    -------

    """
    ret = dict()
    mult = 1
    if typ == "lstm":
        mult = 4
    elif typ == "gru":
        mult = 3
    ret['i2h_weight'] = (mult * num_hidden, data_dim)
    ret['h2h_weight'] = (mult * num_hidden, num_hidden)
    ret['i2h_bias'] = (mult * num_hidden,)
    ret['h2h_bias'] = (mult * num_hidden,)
    return ret


def get_cudnn_parameters(i2h_weight, i2h_bias, h2h_weight, h2h_bias):
    """Get a single param symbol for a CuDNN RNN layer based on the given parameters

    Parameters
    ----------
    i2h_weight
    i2h_bias
    h2h_weight
    h2h_bias

    Returns
    -------
    """
    return symbol.Concat(symbol.Reshape(data=i2h_weight, shape=(-1,)),
                         symbol.Reshape(data=h2h_weight, shape=(-1,)),
                         i2h_bias,
                         h2h_bias, num_args=4, dim=0)


class BaseRNN(object):
    """Abstract base class for RNN layer

    To use a recurrent neural network, we can first create an RNN object and use the step function
    during the symbol construction.
    """
    def __init__(self, num_hidden,
                 dropout=0., recurrent_dropout=0.,
                 i2h_weight=None, i2h_bias=None,
                 h2h_weight=None, h2h_bias=None):

class RNNFactory(object):
    """High level API for constructing a single RNN layer.

    To use a recurrent neural network, we can first create an RNN object and use the step function
    during the symbol construction.

    Currently four types of RNNs are supported and all parameters per layer are grouped into 4 matrices.
    The data layout and transition rules are similar to the RNN API in CuDNN (https://developer.nvidia.com/cudnn)
    1) ReLU RNN:
        h_t = ReLU(W_i x_t + R_i h_{t-1} + b_{W_i} + b_{R_i})

        Parameters:
            W_{i2h} = W_i
            b_{i2h} = b_{W_i}
            W_{h2h} = R_i
            b_{h2h} = b_{R_i}
    2) Tanh RNN:
        h_t = tanh(W_i x_t + R_i h_{t-1} + b_{W_i} + b_{R_i})

        Parameters:
            W_{i2h} = W_i
            b_{i2h} = b_{W_i}
            W_{h2h} = R_i
            b_{h2h} = b_{R_i}
    3) LSTM:
        i_t = \sigma(W_i x_t + R_i h_{t-1} + b_{W_i} + b_{R_i})
        f_t = \sigma(W_f x_t + R_f h_{t-1} + b_{W_f} + b_{R_f})
        o_t = \sigma(W_o x_t + R_o h_{t-1} + b_{W_o} + b_{R_o})
        c^\prime_t = tanh(W_c x_t + R_c h_{t-1} + b_{W_c} + b_{R_c})
        c_t = f_t \circ c_{t-1} + i_t \circ c^\prime_t
        h_t = o_t \circ tanh(c_t)

        Parameters: (input_gate, forget_gate, new_mem, output_gate)
            W_{i2h} = [W_i, W_f, W_c, W_o]
            b_{i2h} = [b_{W_i}, b_{W_f}, b_{W_c}, b_{W_o}]
            W_{h2h} = [R_i, R_f, R_c, R_o]
            b_{h2h} = [b_{R_i}, b_{R_f}, b_{R_c}, b_{R_o}]
    4) GRU:
        i_t = \sigma(W_i x_t + R_i h_{t-1} + b_{W_i} + b_{R_i})
        r_t = \sigma(W_r x_t + R_r h_{t-1} + b_{W_r} + b_{R_r})
        h^\prime_t = tanh(W_h x_t + r_t \circ (R_h h_{t-1} + b_{R_h}) + b_{W_h})
        h_t = (1 - i_t) \circ h^\prime_t + i_t \circ h_{t-1}

        Parameters: (reset_gate, update_gate, new_mem)
            W_{i2h} = [W_r, W_i, W_h]
            b_{i2h} = [b_{W_r}, b_{W_i}, b_{W_h}]
            W_{h2h} = [R_r, R_i, R_h]
            b_{h2h} = [b_{R_r}, b_{R_i}, b_{R_h}]
    """

    def __init__(self, num_hidden, typ='lstm',
                 dropout=0., recurrent_dropout=0.,
                 i2h_weight=None, i2h_bias=None,
                 h2h_weight=None, h2h_bias=None,
                 init_h=None, init_c=None,
                 data_dim=None,
                 cudnn_opt=False,
                 name='LSTM'):
        """Initialization of the RNNFactory

        Parameters
        ----------
        num_hidden : int
            Size of the hidden state for all the layers
        typ: str
            Type of the Recurrent Neural Network, can be 'gru', 'lstm', 'rnn_relu', 'rnn_tanh'
        dropout : float, optional
            Dropout ratios applied to the input of each RNN layer. Use 0 to indicate no-dropout.
        recurrent_dropout : float, optional
            Recurrent dropout ratios for the hidden layer.
            Use 0 to indicate no-recurrent-dropout.
            * The dropout mask is kept fixed for all time-stamps. We follow this paper
            [NIPS2016] A Theoretically Grounded Application of Dropout in Recurrent Neural Networks,
                       Yarin Gal and Zoubin Ghahramani,
            Link: https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
        i2h_weight : mx.sym.symbol, optional
            Weight of the connections between the input and the hidden state.
        i2h_bias : mx.sym.symbol, optional
            Bias of the connections between the input and the hidden state.
        h2h_weight : mx.sym.symbol, optional
            Weight of the connections (including gates) between the hidden states of consecutive timestamps.
        h2h_bias : mx.sym.symbol, optional
            Bias of the connections (including gates) between the hidden states of consecutive timestamps.
        init_h : mx.sym.symbol, optional
            Initial hidden state of the layer
            If set to None, it will be initialized as zero
        init_c : list or tuple, optional
            Initial cell states of all the layers. Only applicable when `typ` is "LSTM"
            If set to None, it will be initialized as zero
        data_dim : int or None, optional
            Dimension of the input data to the symbol.
            data_dim is only required if cudnn_opt is on
        cudnn_opt : bool, optional
            If True, the CuDNN version of RNN will be used. Also, the generated symbol could only be
            used with GPU and recurrent_dropout cannot be used.
        name : str
            Name of the object
        """
        self.name = name
        self.num_hidden = num_hidden
        self.data_dim = data_dim
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.typ = typ.lower()
        assert self.typ in ('gru', 'lstm', 'rnn_relu', 'rnn_tanh'),\
            "RNNFactory: typ=%s is currently not supported. We only support" \
            " 'gru', 'lstm', 'rnn_relu', 'rnn_tanh'." %typ
        if cudnn_opt:
            default_shapes = get_rnn_param_shapes(num_hidden=self.num_hidden, data_dim=data_dim,
                                                  typ=self.typ)
        self.i2h_weight = _get_sym_list(i2h_weight,
                                        default_names=[self.name + "_l%d_i2h_weight" % i
                                                       for i in range(self.layer_num)],
                                        default_shapes=default_shapes["i2h_weight"])
        self.i2h_bias = _get_sym_list(i2h_bias,
                                      default_names=[self.name + "_l%d_i2h_bias" % i
                                                     for i in range(self.layer_num)],
                                      default_shapes=default_shapes["i2h_bias"])
        self.h2h_weight = _get_sym_list(h2h_weight,
                                        default_names=[self.name + "_l%d_h2h_weight" % i
                                                       for i in range(self.layer_num)],
                                        default_shapes=default_shapes["h2h_weight"])
        self.h2h_bias = _get_sym_list(h2h_bias,
                                      default_names=[self.name + "_l%d_h2h_bias" % i
                                                     for i in range(self.layer_num)],
                                      default_shapes=default_shapes["h2h_bias"])
        self.init_h = _get_sym_list(init_h,
                                    default_names=[self.name + "_l%d_init_h" % i
                                                   for i in range(self.layer_num)])
        if typ == 'lstm':
            self.init_c = _get_sym_list(init_c,
                                        default_names=[self.name + "_l%d_init_c" % i
                                                       for i in range(self.layer_num)])
        else:
            assert init_c is None, "init_c should only be used when `typ=lstm`"
            self.init_c = None
        self.cudnn_opt = cudnn_opt
        if self.cudnn_opt:
            assert self.recurrent_dropout == [0.] * self.layer_num,\
                "recurrent dropout is not available when cudnn is used"

    @property
    def params(self):
        return self.i2h_weight + self.i2h_bias + self.h2h_weight + self.h2h_bias

    def step(self, data, prev_h=None, prev_c=None, seq_len=1, ret_typ="all"):
        """Feed the data sequence into the RNN and get the state symbols.

        Parameters
        ----------
        data : list or tuple or Symbol
            The input data. Shape: (seq_len, batch_size, data_dim)
        prev_h : list or tuple or Symbol or None, optional
            The initial hidden states. If None, the symbol constructed during initialization
            will be used.
            Also, number of the initial states must be the same as the layer number,
            e.g, [h0_init, h1_init, h2_init] for a 3-layer RNN
        prev_c : list or tuple or Symbol or None, optional
            The initial cell states. Only applicable when `typ` is 'lstm'. If None,
            the symbol constructed during initialization will be used.
            Also, number of the initial states must be the same as the layer number,
            e.g, [c0_init, c1_init, c2_init] for a 3-layer LSTM
        seq_len : int, optional
            Length of the data sequence
        ret_typ : str, optional
            Determine the parts of the states to return, which can be 'all', 'out', 'state'
            IMPORTANT!! When `cudnn_opt` is on, only the 'out' flag is supported.
            If 'all', symbols that represent states of all the timestamps as well as
             the state of the last timestamp will be returned,
                e.g, For a 3-layer GRU and a length-10 data sequence, the return value will be
                     ([h0, h1, h2], [h0_9, h1_9, h2_9])
                      Here all hi are of shape(seq_len, batch_size, num_hidden[i]) and
                      all hi_j are of shape(batch_size, num_hidden[i])
                     For a 3-layer LSTM and length-10 data sequence, the return value contains both states and cells
                     ([h0, h1, h2], [c0, c1, c2], [h0_9, h1_9, h2_9], [c0_9, c1_9, c2_9])
            If 'out', state outputs of the layers will be returned,
                e.g, For a 3-layer GRU/LSTM and length-10 data sequence, the return value will be
                     [h0, h1, h2]
            If 'state', last state/cell will be returned,
                e.g, For a 3-layer GRU and length-10 data sequence, the return value will be
                     [h0_9, h1_9, h2_9]
                     For a 3-layer LSTM and length-10 data sequence, the return value will be
                     ([h0_9, h1_9, h2_9], [c0_9, c1_9, c2_9])

        Returns
        -------
        tuple
            States generated by feeding the data sequence to the network.
            Refer to the explanation under `ret_typ` argument
        """
        prev_h = self.init_h if prev_h is None else _get_sym_list(prev_h)
        all_h = []
        all_c = []
        last_h = []
        last_c = []
        if self.typ == 'lstm':
            prev_c = self.init_c if prev_c is None else get_sym_list(prev_c)
        else:
            assert prev_c is None,\
                'Cell states is only applicable for LSTM, type of the RNN is %s' %self.typ
        assert seq_len > 0
        if isinstance(data, (list, tuple)):
            assert len(data) == seq_len, \
                "Data length error, expected:%d, received:%d" % (len(data), seq_len)
            data = symbol.Reshape(symbol.Concat(*data, num_args=len(data), dim=0),
                                  shape=(seq_len, -1, 0), reverse=True)
        if self.cudnn_opt:
            # Use the CuDNN version for each layer.
            assert ret_typ in ("out", ), "Only `ret_type=out` is supported " \
                                               "when CuDNN is used."
            for i in range(self.layer_num):
                if self.typ == "lstm":
                    rnn = symbol.RNN(data=data,
                                     state_size=self.num_hidden[i],
                                     num_layers=1,
                                     parameters=get_cudnn_parameters(i2h_weight=self.i2h_weight[i],
                                                                     h2h_weight=self.h2h_weight[i],
                                                                     i2h_bias=self.i2h_bias[i],
                                                                     h2h_bias=self.h2h_bias[i]),
                                     mode=self.typ,
                                     p=self.dropout[i],
                                     state=symbol.expand_dims(prev_h[i], axis=0),
                                     state_cell=symbol.expand_dims(prev_c[i], axis=0),
                                     name=self.name + "->layer%d" %i,
                                     state_outputs=False)
                    data = rnn
                    all_h.append(rnn)
                else:
                    rnn = symbol.RNN(data=data,
                                     state_size=self.num_hidden[i],
                                     num_layers=1,
                                     parameters=get_cudnn_parameters(i2h_weight=self.i2h_weight[i],
                                                                     h2h_weight=self.h2h_weight[i],
                                                                     i2h_bias=self.i2h_bias[i],
                                                                     h2h_bias=self.h2h_bias[i]),
                                     mode=self.typ,
                                     p=self.dropout[i],
                                     state=symbol.expand_dims(prev_h[i], axis=0),
                                     name=self.name + "->layer%d" %i,
                                     state_outputs=False)
                    data = rnn
                    all_h.append(rnn)
            if ret_typ == 'out':
                return all_h
            else:
                raise NotImplementedError
        else:
            #TODO Optimize this part by computing matrix multiplication first
            for i in range(self.layer_num):
                if self.typ == "lstm":
                    layer_all_h, layer_all_c, layer_last_h, layer_last_c =\
                        step_lstm(num_hidden=self.num_hidden[i], data=data,
                                  prev_h=prev_h[i], prev_c=prev_c[i],
                                  i2h_weight=self.i2h_weight[i], i2h_bias=self.i2h_bias[i],
                                  h2h_weight=self.h2h_weight[i], h2h_bias=self.h2h_bias[i],
                                  seq_len=seq_len,
                                  dropout=self.dropout[i],
                                  name=self.name + "->layer%d"%i)
                    all_h.append(layer_all_h)
                    all_c.append(layer_all_c)
                    last_h.append(layer_last_h)
                    last_c.append(layer_last_c)
                else:
                    step_func = None
                    if self.typ == 'rnn_tanh':
                        step_func = step_tanh_rnn
                    elif self.typ == 'rnn_relu':
                        step_func = step_relu_rnn
                    elif self.typ == 'gru':
                        step_func = step_gru
                    layer_all_h, layer_last_h = \
                        step_func(num_hidden=self.num_hidden[i], data=data,
                                  prev_h=prev_h[i],
                                  i2h_weight=self.i2h_weight[i], i2h_bias=self.i2h_bias[i],
                                  h2h_weight=self.h2h_weight[i], h2h_bias=self.h2h_bias[i],
                                  seq_len=seq_len,
                                  dropout=self.dropout[i],
                                  name=self.name + "->layer%d" % i)
                    all_h.append(layer_all_h)
                    last_h.append(layer_last_h)
                data = all_h[-1]
        if ret_typ == 'all':
            if self.typ == 'lstm':
                return all_h, all_c, last_h, last_c
            else:
                return all_h, last_h
        elif ret_typ == 'out':
            return all_h
        elif ret_typ == 'state':
            if self.typ == 'lstm':
                return last_h, last_c
            else:
                return last_h

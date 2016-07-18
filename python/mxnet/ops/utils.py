import numbers
import numpy as np
import sys
from ..symbol import Symbol, Variable

def get_sym_list(syms, default_names=None):
    if syms is None:
        if default_names is not None:
            return [Variable(name=name) for name in default_names]
    assert isinstance(syms, (list, tuple, Symbol))
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


def get_int_list(values):
    if isinstance(values, numbers.Number):
        return [np.int32(values)]
    elif isinstance(values, (list, tuple)):
        try:
            ret = [np.int32(value) for value in values]
            return ret
        except(ValueError):
            print("Need iterable with numeric elements, received: %s" %str(values))
            sys.exit(1)
    else:
        raise ValueError("Unaccepted value type, values=%s" %str(values))


def get_float_list(values):
    if isinstance(values, numbers.Number):
        return [np.float32(values)]
    elif isinstance(values, (list, tuple)):
        try:
            ret = [np.float32(value) for value in values]
            return ret
        except(ValueError):
            print("Need iterable with numeric elements, received: %s" %str(values))
            sys.exit(1)
    else:
        raise ValueError("Unaccepted value type, values=%s" % str(values))

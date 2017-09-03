# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Contrib NDArray API of MXNet."""
__all__ = []

# pylint: disable=undefined-variable
def global_norm(t_list, name=None):
    """Computes the global norm of multiple tensors.

    Given a tuple or list of tensors t_list, this operation returns the global norm of the elements
     in all tensors in t_list. The global norm is computed as:

    ``global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))``

    Any entries in t_list that are of type None are ignored.

    Parameters
    ----------
    t_list: list or tuple
        The symbol list
    name: str, optional
        Name of the global norm symbol

    Returns
    -------
    ret: Symbol
        The global norm. The shape of the resulting symbols will be (1,)

    Examples
    --------
    >>> x = mx.sym.ones((2, 3))
    >>> y = mx.sym.ones((5, 6))
    >>> z = mx.sym.ones((4, 2, 3))
    >>> ret = mx.sym.global_norm([x, y, z])
    >>> ret.eval()[0].asscalar()
    7.74597
    """
    import mxnet as mx
    ret = None
    for t in t_list:
        if t is not None:
            if ret is None:
                ret = mx.sym.square(mx.sym.norm(t))
            else:
                ret = ret + mx.sym.square(mx.sym.norm(t))
    ret = mx.sym.sqrt(ret, name=name)
    return ret
# pylint: enable=undefined-variable
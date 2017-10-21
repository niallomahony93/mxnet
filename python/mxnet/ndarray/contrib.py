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

# coding: utf-8
# pylint: disable=wildcard-import, unused-wildcard-import
"""Contrib NDArray API of MXNet."""
try:
    from .gen_contrib import *
except ImportError:
    pass

__all__ = []

def global_norm(t_list):
    """Computes the global norm of multiple tensors.

    Given a tuple or list of tensors t_list, this operation returns the global norm of the elements
     in all tensors in t_list. The global norm is computed as:

    ``global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))``

    Any entries in t_list that are of type None are ignored.

    Parameters
    ----------
    t_list: list or tuple
        The NDArray list

    Returns
    -------
    ret: NDArray
        The global norm. The shape of the NDArray will be (1,)

    Examples
    --------
    >>> x = mx.nd.ones((2, 3))
    >>> y = mx.nd.ones((5, 6))
    >>> z = mx.nd.ones((4, 2, 3))
    >>> print(mx.nd.global_norm([x, y, z]).asscalar())
    7.74597
    >>> xnone = None
    >>> ret = mx.nd.global_norm([x, y, z, xnone])
    >>> print(ret.asscalar())
    7.74597
    """
    import mxnet.ndarray as nd
    ret = None
    for arr in t_list:
        if arr is not None:
            if ret is None:
                ret = nd.square(nd.norm(arr))
            else:
                ret += nd.square(nd.norm(arr))
    ret = nd.sqrt(ret)
    return ret
# pylint: disable=too-many-locals, invalid-name
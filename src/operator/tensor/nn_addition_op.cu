/*!
*  Copyright (c) 2017 by Contributors
* \file nn_addition_op.cu
* \brief GPU Implementation of nn additional operations
*/
// this will be invoked by gcc and compile GPU version
#include "./nn_addition_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(LocalCorrelation)
.set_attr<FCompute>("FCompute<gpu>", LocalCorrelationForward_<gpu>);

NNVM_REGISTER_OP(_backward_LocalCorrelation)
.set_attr<FCompute>("FCompute<gpu>", LocalCorrelationBackward_<gpu>);

NNVM_REGISTER_OP(LocalFilter)
.set_attr<FCompute>("FCompute<gpu>", LocalFilterForward_<gpu>);

NNVM_REGISTER_OP(_backward_LocalFilter)
.set_attr<FCompute>("FCompute<gpu>", LocalFilterBackward_<gpu>);

}  // namespace op
}  // namespace mxnet

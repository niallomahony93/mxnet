/*!
*  Copyright (c) 2017 by Contributors
* \file nn_addition_op.cu
* \brief GPU Implementation of nn additional operations
*/
// this will be invoked by gcc and compile GPU version
#include "./nn_addition_op-inl.h"
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

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

NNVM_REGISTER_OP(BSN)
.set_attr<FCompute>("FCompute<gpu>", BinaryStochasticNeuronCompute<gpu>);

NNVM_REGISTER_OP(_backward_BSN)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::sigmoid_grad>>);
}  // namespace op
}  // namespace mxnet

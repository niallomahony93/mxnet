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

NNVM_REGISTER_OP(LocalSparseFilter)
.set_attr<FCompute>("FCompute<gpu>", LocalSparseFilterForward_<gpu>);

NNVM_REGISTER_OP(_backward_LocalSparseFilter)
.set_attr<FCompute>("FCompute<gpu>", LocalSparseFilterBackward_<gpu>);

NNVM_REGISTER_OP(BSN)
.set_attr<FCompute>("FCompute<gpu>", BinaryStochasticNeuronCompute<gpu>);

NNVM_REGISTER_OP(_backward_BSN)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<gpu, unary_bwd<mshadow_op::sigmoid_grad>>);

NNVM_REGISTER_OP(argsort_last)
.set_attr<FCompute>("FCompute<gpu>", ArgSortLast<gpu>);
}  // namespace op
}  // namespace mxnet

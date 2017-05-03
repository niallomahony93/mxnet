/*!
*  Copyright (c) 2017 by Contributors
* \file local_sparse_filter.h
* \brief Function defintion of nn related operators
*/
#ifndef MXNET_OPERATOR_TENSOR_LOCAL_SPARSE_FILTER_H_
#define MXNET_OPERATOR_TENSOR_LOCAL_SPARSE_FILTER_H_
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include "../mxnet_op.h"

namespace mxnet {
namespace op {
template<typename DType>
void LocalSparseFilterForwardImpl(const mshadow::Tensor<cpu, 4, DType> &data,
                                  const mshadow::Tensor<cpu, 3, DType> &weight,
                                  const mshadow::Tensor<cpu, 1, DType> &bias,
                                  const mshadow::Tensor<cpu, 5, DType> &values,
                                  const mshadow::Tensor<cpu, 5, DType> &indices,
                                  const mshadow::Tensor<cpu, 4, DType> &out) {
  LOG(FATAL) << "Not Implemented";
}
}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./local_sparse_filter.cuh"
#endif
#endif  // MXNET_OPERATOR_TENSOR_LOCAL_SPARSE_FILTER_H_

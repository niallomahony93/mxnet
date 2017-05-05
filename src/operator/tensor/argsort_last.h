/*!
*  Copyright (c) 2017 by Contributors
* \file argsort_last.h
* \brief
*/
#ifndef MXNET_OPERATOR_ARGSORT_LAST_H_
#define MXNET_OPERATOR_ARGSORT_LAST_H_
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include "../mxnet_op.h"

namespace mxnet {
namespace op {
template<typename DType>
void ArgSortLastImpl(const mshadow::Tensor<cpu, 1, DType> &data,
                     const mshadow::Tensor<cpu, 1, DType> &out,
                     const mshadow::Tensor<cpu, 1, int> &d_offsets,
                     bool is_ascend,
                     int batch_num,
                     const Resource &resource) {
  LOG(FATAL) << "Not Implemented";
}
}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./argsort_last.cuh"
#endif
#endif  // MXNET_OPERATOR_ARGSORT_LAST_H_

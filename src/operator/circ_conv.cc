/*!
 *  Copyright (c) 2015 by Contributors
 * \file circ_conv.cc
 * \brief CPU Implementation of matrix operations
 * \author Xingjian Shi
 */
// this will be invoked by gcc and compile CPU version
#include "./circ_conv-inl.h"


namespace mshadow {
template<typename DType>
inline DType modn(DType a, DType N) {
  return (a < 0) ? a + N : ((a >= N) ? a - N : a);
}

template<typename DType>
inline void CircularConvolution1DForwardImpl_(Tensor<cpu, 2, DType> out,
                                              const Tensor<cpu, 2, DType> &data,
                                              const Tensor<cpu, 2, DType> &weight) {
  const int batch_size = data.size(0);
  const int content_size = data.size(1);
  const int kernel_size = weight.size(1);
  for (int n = 0; n < batch_size; ++n) {
    for (int i = 0; i < content_size; ++i) {
      for (int j = 0; j < kernel_size; ++j) {
        int indx = modn(i - j, content_size);
        out[n][i] += data[n][indx] * weight[n][j];
      }
    }
  }
}

template<typename DType>
inline void CircularConvolution1DBackwardImpl_(const Tensor<cpu, 2, DType> &out_grad,
                                               Tensor<cpu, 2, DType> data_grad,
                                               Tensor<cpu, 2, DType> weight_grad,
                                               const Tensor<cpu, 2, DType> &data,
                                               const Tensor<cpu, 2, DType> &weight) {
  const int batch_size = data_grad.size(0);
  const int content_size = data_grad.size(1);
  const int kernel_size = weight_grad.size(1);
  // Calculate data_grad
  for (int n = 0; n < batch_size; ++n) {
    for (int i = 0; i < content_size; ++i) {
      for (int j = 0; j < kernel_size; ++j) {
        int indx = modn(i + j, content_size);
        data_grad[n][i] += out_grad[n][indx] * weight[n][j];
      }
    }
  }
  // Calculate weight_grad
  for (int n = 0; n < batch_size; ++n) {
    for (int i = 0; i < kernel_size; ++i) {
      for (int j = 0; j < content_size; ++j) {
        int indx = modn(j - i, content_size);
        weight_grad[n][i] += out_grad[n][j] * data[n][indx];
      }
    }
  }
  return;
}
}  // namespace mshadow
/*!
 *  Copyright (c) 2015 by Contributors
 * \file circ_conv.cu
 * \brief GPU Implementation of circular convolution
 */
// this will be invoked by gcc and compile GPU version
#include "./circ_conv-inl.h"
#include "mshadow/cuda/tensor_gpu-inl.cuh"
#define MX_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)


namespace mshadow {
namespace cuda {

template<typename DType>
DType uppow2(DType num) {
  return 1 << static_cast<DType>(ceil(log2(num)));
}

template<typename DType>
__device__ DType modn(DType a, DType N) {
  return (a < 0) ? a + N : ((a >= N) ? a - N : a);
}

template<int x_bits, typename DType>
__global__ void CircularConvolution1DForwardKernel(const int batch_size,
                                                   const int content_size,
                                                   const int kernel_size,
                                                   const DType *data,
                                                   const DType *weight,
                                                   DType *out) {
  const unsigned x_size = 1 << x_bits;
  const index_t y = blockIdx.x;
  __shared__ DType s_data_[x_size];
  if (threadIdx.x < content_size) {
    s_data_[threadIdx.x] = data[threadIdx.x];
  } else if (threadIdx.x < content_size + kernel_size) {
    s_data_[threadIdx.x] = weight[threadIdx.x - content_size];
  } else {
    s_data_[threadIdx.x] = 0;
  }
  __syncthreads();
  int thread_id = static_cast<int> (threadIdx.x);
  if (threadIdx.x < content_size) {
    for (int i = 0; i < kernel_size; ++i) {
      int indx = modn(thread_id + kernel_size - 1 - i, content_size);
      out[y * blockDim.x + threadIdx.x] += s_data_[indx] * s_data_[i + content_size];
    }
  }
}

template<int x_bits, typename DType>
__global__ void CircularConvolution1DBackwardKernel(const int batch_size,
                                                          const int content_size,
                                                          const int kernel_size,
                                                          const DType *out_grad,
                                                          const DType *data,
                                                          const DType *weight,
                                                          DType *data_grad,
                                                          DType *weight_grad) {
  const unsigned x_size = 1 << x_bits;
  const index_t y = blockIdx.x;
  __shared__ DType s_data_[x_size];
  if (threadIdx.x < content_size) {
    s_data_[threadIdx.x] = data[threadIdx.x];
  } else if (threadIdx.x < content_size + kernel_size) {
    s_data_[threadIdx.x] = weight[threadIdx.x - content_size];
  } else if (threadIdx.x < content_size * 2 + kernel_size) {
    s_data_[threadIdx.x] = out_grad[threadIdx.x - content_size - kernel_size];
  } else {
    s_data_[threadIdx.x] = 0;
  }
  __syncthreads();
  int thread_id = static_cast<int>(threadIdx.x);
  if (threadIdx.x < content_size) {
    for (int i = 0; i < kernel_size; ++i) {
      int indx = modn(thread_id - kernel_size + 1 - i, content_size);
      data_grad[y * blockDim.x + threadIdx.x] +=
        s_data_[indx + content_size + kernel_size] * s_data_[i + content_size];
    }
  } else if (threadIdx.x < content_size * 2) {
    for (int i = 0; i < kernel_size; ++i) {
      int indx = modn(thread_id + kernel_size - 1 - i, content_size);
      weight_grad[y * blockDim.x + i] += s_data_[thread_id + content_size + kernel_size] * s_data_[indx];
    }
  }
}
}  // namespace cuda

template<typename DType>
inline void CircularConvolution1DForwardImpl_(const Tensor<gpu, 2, DType> &out,
                                              const Tensor<gpu, 2, DType> &data,
                                              const Tensor<gpu, 2, DType> &weight) {
  using namespace cuda;
  DType *out_ = out.dptr_;
  const DType *data_ = data.dptr_;
  const DType *weight_ = weight.dptr_;
  const int batch_size = data.size(0);
  const int content_size = data.size(1);
  const int kernel_size = weight.size(1);
  const int thread_num = uppow2(content_size + kernel_size);
  CHECK(kBaseThreadNum > (content_size + kernel_size));
  dim3 dimGrid(kBaseThreadNum);
  dim3 dimBlock(batch_size);
  CheckLaunchParam(dimGrid, dimBlock, "Circular Convolution Forward");
  //TODO Optimize the kernel!
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  CircularConvolution1DForwardKernel<kBaseThreadBits, DType> << <dimGrid, dimBlock, 0, stream >> > (
    batch_size, content_size, kernel_size, data_, weight_, out_);
}

template<typename DType>
inline void CircularConvolution1DBackwardImpl_(const Tensor<gpu, 2, DType> &out_grad,
                                               const Tensor<gpu, 2, DType> &data_grad,
                                               const Tensor<gpu, 2, DType> &weight_grad,
                                               const Tensor<gpu, 2, DType> &data,
                                               const Tensor<gpu, 2, DType> &weight) {
  using namespace cuda;
  const DType *out_grad_ = out_grad.dptr_;
  DType *data_grad_ = data_grad.dptr_;
  DType *weight_grad_ = weight_grad.dptr_;
  const DType *data_ = data.dptr_;
  const DType *weight_ = weight.dptr_;
  const int batch_size = data_grad.size(0);
  const int content_size = data_grad.size(1);
  const int kernel_size = weight_grad.size(1);
  CHECK(kBaseThreadNum > (2 * content_size + kernel_size));
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(batch_size);
  CheckLaunchParam(dimGrid, dimBlock, "Circular Convolution Backward");
  //TODO Optimize the kernel!
  cudaStream_t stream = Stream<gpu>::GetStream(out_grad.stream_);
  CircularConvolution1DBackwardKernel<kBaseThreadBits, DType> << <dimGrid, dimBlock, 0, stream >> > (
    batch_size, content_size, kernel_size, out_grad_, data_, weight_, data_grad_, weight_grad_);
  return;
}
}  // namespace mshadow
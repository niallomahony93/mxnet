/*!
*  Copyright (c) 2017 by Contributors
* \file local_sparse_filter.cuh
* \brief Function defintion of nn related operators
*/
#ifndef MXNET_OPERATOR_TENSOR_LOCAL_SPARSE_FILTER_CUH_
#define MXNET_OPERATOR_TENSOR_LOCAL_SPARSE_FILTER_CUH_
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <vector>
#include "../mxnet_op.h"

namespace mxnet {
namespace op {
#define ADDRESS_4D_BCHW(b, c, h, w, C, H, W) ((((b) * (C) + (c)) * (H) + (h)) * (W) + (w))
#define ADDRESS_3D_BCHW(b, c, hw, C, HW) (((b) * (C) + (c)) * (HW) + (hw))

#define ADDRESS_4D_BHWC(b, h, w, c, H, W, C) ((((b) * (H) + (h)) * (W) + (w)) * (C) + (c))
#define ADDRESS_3D_BHWC(b, hw, c, HW, C) (((b) * (HW) + (hw)) * (C) + (c))


const int TILE_SIZE = 32;
const int kWarpSize = 32;

__device__ void sumReduceShMem(volatile float s[])
{
  /* obviously only works for 32 elements */
  /* sums up a shared memory array of 32 elements, stores it in s[0] */
  /* whole warp can then read first element (broadcasting) */
  if (threadIdx.x < 16) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 16]; }
  if (threadIdx.x < 8) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 8]; }
  if (threadIdx.x < 4) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 4]; }
  if (threadIdx.x < 2) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 2]; }
  if (threadIdx.x < 1) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 1]; }
}

template<typename DType>
__inline__ __device__
int warpReduceSum(DType val) {
  for (DType offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
}

template<typename DType>
__global__ void LocalSparseFilterForwardKernelBHWC(const int B, const int inC, const int H, const int W,
                                                   const int outC, const int L, const int K, const DType pad_val,
                                                   DType* out,
                                                   const DType* __restrict data,
                                                   const DType* __restrict weight,
                                                   const DType* __restrict bias,
                                                   const DType* __restrict values,
                                                   const DType* __restrict indices) {
  //TODO Use extern instead
  __shared__ float local_connection_val[128];
  __shared__ int local_connection_ind[128];
  __shared__ float data_shared[TILE_SIZE][TILE_SIZE + 1];
  __shared__ float data_acc_mem[TILE_SIZE][TILE_SIZE + 1];
  __shared__ float out_shared[TILE_SIZE];
  __shared__ volatile float weight_shared[TILE_SIZE][TILE_SIZE + 1]; // Add 1 to avoid bank conflict
  int tx = threadIdx.x, ty = threadIdx.y;
  int tid = tx + ty * blockDim.x;
  if (tx == 0) {
    weight_shared[ty][TILE_SIZE] = 0.0f;
  }
  __syncthreads();
  for (int index = blockIdx.x; index < B * H * W; index += gridDim.x) {
    int w = index % W;
    int h = (index / W) % H;
    int b = index / W / H;
    // Load the local connection values and indices
    if (tid < L * K) {
      int address = ADDRESS_4D_BCHW(b, tid, h, w, L * K, H, W);
      local_connection_val[tid] = values[address];
      local_connection_ind[tid] = __float2int_rn(indices[address]);
    }
    __syncthreads();
#pragma unroll
    for (int oc_base = 0; oc_base < outC; oc_base += TILE_SIZE) {
      if (ty == 0 && oc_base + tx < outC) {
        out_shared[tx] = bias[oc_base + tx];
      }
      __syncthreads();
      int oc = oc_base + ty;
#pragma unroll
      for (int l = 0; l < L; ++l) {
#pragma unroll
        for (int ic_base = 0; ic_base < inC; ic_base += TILE_SIZE) {
          int ic = ic_base + tx;
          // Load the weight into shared memory
          if (ic < inC && oc < outC) {
            weight_shared[ty][tx] = weight[(l * outC + oc) * inC + ic];
          } else {
            weight_shared[ty][tx] = 0.0f;
          }
          // Load the local connection data into shared memory.
          if (ic < inC && ty < K) {
            if (local_connection_ind[l * K + ty] >= 0 && local_connection_ind[l * K + ty] < H * W) {
              data_shared[ty][tx] = local_connection_val[l * K + ty] * data[ADDRESS_3D_BHWC(b, local_connection_ind[l * K + ty], ic, H * W, inC)];
            } else {
              data_shared[ty][tx] = local_connection_val[l * K + ty] * pad_val;
            }
          } else {
            data_shared[ty][tx] = 0;
          }
          __syncthreads();
          data_acc_mem[tx][ty] = data_shared[ty][tx];
          __syncthreads();
          sumReduceShMem(data_acc_mem[ty]);
          __syncthreads();
          // Calculate the result inplace in the weight matrix
          weight_shared[ty][tx] *= data_acc_mem[tx][0];
          __syncthreads();
          // mshadow::cuda::Reduce1D<mshadow::red::sum, 5>(weight_shared[ty]);
          // __syncthreads();
          sumReduceShMem(weight_shared[ty]);
          __syncthreads();
          // Write the result back to the shared output vector
          if (tx == 0 && oc < outC) {
            out_shared[ty] += weight_shared[ty][0];
          }
        }
      }
      __syncthreads();
      if (tx == 0 && oc < outC) {
        out[ADDRESS_4D_BCHW(b, oc, h, w, outC, H, W)] = out_shared[ty];
      }
      __syncthreads();
    }
  }
}


template<typename DType>
void LocalSparseFilterForwardImpl(const mshadow::Tensor<gpu, 4, DType> &data,
                                  const mshadow::Tensor<gpu, 3, DType> &weight,
                                  const mshadow::Tensor<gpu, 1, DType> &bias,
                                  const mshadow::Tensor<gpu, 5, DType> &values,
                                  const mshadow::Tensor<gpu, 5, DType> &indices,
                                  const mshadow::Tensor<gpu, 4, DType> &out,
                                  const DType pad_val) {
  using namespace mshadow;
  using namespace mshadow::cuda;
  int B = data.shape_[0];
  int L = values.shape_[1];
  int K = values.shape_[2];
  int H = data.shape_[1];
  int W = data.shape_[2];
  int inC = data.shape_[3];
  int outC = out.shape_[1];
  const int grid_dim_x = B * H * W;
  // const int grid_dim_y = (outC + TILE_SIZE - 1) / TILE_SIZE;
  CHECK_LT(L * K, 128);
  dim3 dimGrid(grid_dim_x);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE);
  CheckLaunchParam(dimGrid, dimBlock, "LocalSparseFilterForward");
  cudaStream_t stream = Stream<gpu>::GetStream(data.stream_);
  LocalSparseFilterForwardKernelBHWC << <dimGrid, dimBlock, 0, stream >> >
    (B, inC, H, W, outC, L, K, pad_val, out.dptr_, data.dptr_, weight.dptr_, bias.dptr_, values.dptr_, indices.dptr_);
  cudaError err = cudaPeekAtLastError();
  CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

template<typename DType>
__global__ void LocalSparseFilterBackwardAccKernelBHWC(const int B, const int inC, const int H, const int W,
                                                       const int outC, const int L, const int K, const DType pad_val,
                                                       const bool need_data_grad, const bool need_weight_grad, const bool need_values_grad,
                                                       DType* data_grad, DType* weight_grad, DType* values_grad,
                                                       const DType* __restrict out_grad,
                                                       const DType* __restrict data,
                                                       const DType* __restrict weight,
                                                       const DType* __restrict values,
                                                       const DType* __restrict indices) {
  //TODO Use extern instead
  __shared__ bool need_data_grad_dev;
  __shared__ bool need_weight_grad_dev;
  __shared__ bool need_values_grad_dev;
  __shared__ float local_connection_val[128];
  __shared__ int local_connection_ind[128];
  __shared__ float data_shared[TILE_SIZE][TILE_SIZE + 1];
  __shared__ volatile float weight_shared[TILE_SIZE][TILE_SIZE + 1]; // Add 1 to avoid bank conflict
  __shared__ float out_grad_shared[TILE_SIZE];
  __shared__ volatile float val_grad_mem_shared[TILE_SIZE][TILE_SIZE + 1]; // Used to calculate the gradient for values
  __shared__ volatile float data_grad_mem_shared[TILE_SIZE][TILE_SIZE + 1]; // Used to calculate the gradient for data
  __shared__ volatile float weight_grad_mem_shared[TILE_SIZE][TILE_SIZE + 1]; // Used to calculate the gradient for weight
  int tx = threadIdx.x, ty = threadIdx.y;
  int tid = tx + ty * blockDim.x;
  if (tx == 0) {
    weight_shared[ty][TILE_SIZE] = 0.0f;
  }
  if (tx == 0 && ty == 0) {
    // Copy the flags to device memory
    need_data_grad_dev = need_data_grad;
    need_weight_grad_dev = need_weight_grad;
    need_values_grad_dev = need_values_grad;
  }
  __syncthreads();
  for (int index = blockIdx.x; index < B * H * W; index += gridDim.x) {
    int w = index % W;
    int h = (index / W) % H;
    int b = index / W / H;
    // Load the initial data + the local connection values and indices
    if (tid < L * K) {
      int address = ADDRESS_4D_BCHW(b, tid, h, w, L * K, H, W);
      local_connection_val[tid] = values[address];
      local_connection_ind[tid] = __float2int_rn(indices[address]);
    }
    __syncthreads();
#pragma unroll
    for (int oc_base = 0; oc_base < outC; oc_base += TILE_SIZE) {
      int oc = oc_base + ty;
#pragma unroll
      for (int l = 0; l < L; ++l) {
        for (int ic_base = 0; ic_base < inC; ic_base += TILE_SIZE) {
          int ic = ic_base + tx;
          // Load the weight into shared memory
          if (ic < inC && oc < outC) {
            weight_shared[ty][tx] = weight[(l * outC + oc) * inC + ic];
            data_grad_mem_shared[tx][ty] = weight_shared[ty][tx];
          } else {
            weight_shared[ty][tx] = 0.0f;
            data_grad_mem_shared[tx][ty] = 0.0f;
          }
          if (tx + oc_base < outC && ty == TILE_SIZE) {
            out_grad_shared[tx] = out_grad[ADDRESS_4D_BHWC(b, h, w, tx + oc_base, H, W, outC)];
          }
          // Load the local connected data into shared memory.
          if (ic < inC && ty < K) {
            if (local_connection_ind[l * K + ty] >= 0 && local_connection_ind[l * K + ty] < H * W) {
              int address = ADDRESS_3D_BHWC(b, local_connection_ind[l * K + ty], ic, H * W, inC);
              data_shared[ty][tx] = data[address];
            } else {
              data_shared[ty][tx] = pad_val;
            }
          } else {
            data_shared[ty][tx] = 0;
          }
          // Calculate the gradient for value
          val_grad_mem_shared[ty][tx] = data_shared[ty][tx];
        }
      }
    }
  }
}

template<typename DType>
void LocalSparseFilterBackwardAccImpl(const mshadow::Tensor<gpu, 4, DType> &out_grad,
                                      const mshadow::Tensor<gpu, 4, DType> &data,
                                      const mshadow::Tensor<gpu, 3, DType> &weight,
                                      const mshadow::Tensor<gpu, 5, DType> &values,
                                      const mshadow::Tensor<gpu, 5, DType> &indices,
                                      const mshadow::Tensor<gpu, 4, DType> &data_grad,
                                      const mshadow::Tensor<gpu, 3, DType> &weight_grad,
                                      const mshadow::Tensor<gpu, 5, DType> &values_grad,
                                      const bool need_data_grad,
                                      const bool need_weight_grad,
                                      const bool need_values_grad,
                                      const DType pad_val) {
  using namespace mshadow;
  using namespace mshadow::cuda;
  int B = data.shape_[0];
  int L = values.shape_[1];
  int K = values.shape_[2];
  int H = data.shape_[1];
  int W = data.shape_[2];
  int inC = data.shape_[3];
  int outC = out_grad.shape_[1];
  const int grid_dim_x = B * H * W;
  // const int grid_dim_y = (outC + TILE_SIZE - 1) / TILE_SIZE;
  CHECK_LT(L * K, 128);
  CHECK_LT(K, 32);
  dim3 dimGrid(grid_dim_x);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE);
  CheckLaunchParam(dimGrid, dimBlock, "LocalSparseFilterForward");
  cudaStream_t stream = Stream<gpu>::GetStream(data.stream_);
  LocalSparseFilterBackwardAccKernelBHWC << <dimGrid, dimBlock, 0, stream >> >
    (B, inC, H, W, outC, L, K, pad_val, need_data_grad, need_weight_grad, need_values_grad,
     data_grad.dptr_, weight_grad.dptr_, values_grad.dptr_,
     out_grad.dptr_, data.dptr_, weight.dptr_, values.dptr_, indices.dptr_);
  cudaError err = cudaPeekAtLastError();
  CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_LOCAL_SPARSE_FILTER_CUH_

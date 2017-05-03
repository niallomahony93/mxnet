/*!
*  Copyright (c) 2017 by Contributors
* \file nn_addition_op.cu
* \brief GPU Implementation of nn additional operations
*/
// this will be invoked by gcc and compile GPU version
#include "./nn_addition_op-inl.h"
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

#define ADDRESS_4D(c, h, w, B, C, H, W) ((((b) * (C) + (c)) * (H) + (h)) * (W) + (w))
#define ADDRESS_3D(c, hw, B, C, HW) (((b) * (C) + (c)) * (HW) + (hw))

namespace mxnet {
namespace op {
const int TILE_SIZE = 32;

template<typename DType>
__global__ void LocalSparseFilterForwardKernel(const int B, const int inC, const int H, const int W,
                                               const int outC, const int L, const int K,
                                               DType* out,
                                               const DType* __restrict data,
                                               const DType* __restrict weight,
                                               const DType* __restrict bias,
                                               const DType* __restrict values,
                                               const DType* __restrict indices) {
  //TODO Use extern instead
  __shared__ float local_connection_val;
  __shared__ int local_connection_ind;
  __shared__ float data_shared[TILE_SIZE];
  __shared__ float out_shared[TILE_SIZE];
  __shared__ float weight_shared[TILE_SIZE][TILE_SIZE];
  int tx = threadIdx.x, ty = threadIdx.y;
  for (int index = blockIdx.x + blockIdx.y * gridDim.x;
    index < B * H * W;
    index += gridDim.x * gridDim.y) {
    int w = index % W;
    index /= W;
    int h = index % H;
    index /= H;
    int b = index;
    int oc = blockIdx.y * TILE_SIZE + ty; // Enumerate the outC
    if (tx == 0) {
      if (oc < outC) {
        out_shared[ty] = bias[ty];
      } else {
        out_shared[ty] = 0.0f;
      }
    }
    __syncthreads();
#pragma unroll
    for (int ic = 0; ic < inC; ic += TILE_SIZE) {
      for (int l = 0; l < L; ++l) {
        if (oc + ty < outC && ic + tx < inC) {
          // Load Weight
          weight_shared[ty][tx] = weight[(l * outC + oc + ty) * inC + ic + tx];
        } else {
          weight_shared[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < K; ++k) {
          if (ty == 0 && tx == 0) {
            int address = ADDRESS_4D(l * K + k, h, w, B, L * K, H, W);
            local_connection_val = values[address];
            local_connection_ind = __float2int_rn(indices[address]);
          }
          __syncthreads();
          if (ty == 0) {
            if (ic + tx < inC) {
              data_shared[tx] = data[ADDRESS_3D(ic + tx, local_connection_ind, B, C, H * W)];
            } else {
              data_shared[tx] = 0;
            }
          }
          __syncthreads();
          weight_shared[ty][tx] *= (local_connection_val * data_shared[tx]);
          __syncthreads();
          Reduce1D<mshadow::red::sum, 5>(weight_shared[ty]);
          __syncthreads();
          if (tx == 0) {
            out_shared[ty] += weight_shared[ty];
          }
          __syncthreads();
        }
      }
    }
    if (tx == 0) {
      if (oc < outC) {
        out[ADDRESS_4D(oc, h, w, B, outC, H, W)] = out_shared[ty];
      }
    }
    __syncthreads();
  }
}

template<typename DType>
void LocalSparseFilterForwardImpl(const mshadow::Tensor<gpu, 4, DType> &data,
                                  const mshadow::Tensor<gpu, 3, DType> &weight,
                                  const mshadow::Tensor<gpu, 1, DType> &bias,
                                  const mshadow::Tensor<gpu, 5, DType> &values,
                                  const mshadow::Tensor<gpu, 5, DType> &indices,
                                  const mshadow::Tensor<gpu, 5, DType> &out) {
  using namespace mshadow::cuda;
  int B = data.shape_[0];
  int inC = data.shape_[1];
  int H = data.shape_[2];
  int W = data.shape_[3];
  int L = values.shape_[1];
  int K = values.shape_[2];
  int outC = weight.shape_[1];
  const int grid_dim_x = B * H * W;
  const int grid_dim_y = outC / TILE_SIZE;
  dim3 dimGrid(grid_dim_x, grid_dim_y);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE);
  CheckLaunchParam(dimGrid, dimBlock, "LocalSparseFilterForward");
  cudaStream_t stream = Stream<gpu>::GetStream(data.stream_);
  LocalSparseFilterForwardKernel << <dimGrid, dimBlock, 0, stream >> >
    (B, inC, H, W, outC, L, K, out, data, weight, bias, values, indices);
  cudaError err = cudaPeekAtLastError();
  CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}
}
}

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
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::sigmoid_grad>>);
}  // namespace op
}  // namespace mxnet

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file layer_norm.cu
 * \brief Implements Ba et. al, Layer Normalization (https://arxiv.org/abs/1607.06450).
*/
#include "./layer_norm-inl.h"

using namespace mshadow::cuda;

namespace mxnet {
namespace op {

template <typename DType>
__device__ __forceinline__ DType WARP_SHFL(DType value, int src_lane,
                                           int width = 32, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
  return __shfl_sync(mask, value, src_lane, width);
#else
  return __shfl(value, src_lane, width);
#endif
}

/* A single updating step of the Welford's online algorithm to calculate the mean and variance.
 * The value 'curr' will be accumulated to the (mean, sigma2, count) triplet.
 *
 */
template<typename DType>
__device__ void welford_online_sum_step(const DType curr,
                                   DType& mean,
                                   DType& sigma2,
                                   DType& count) {
  count += DType(1);
  DType delta = curr - mean;
  mean += delta / count;
  sigma2 += delta * (curr - mean);
}

/* Merge the mean/variance of two partitions. It's the key step of the Chan's parallel algorithm.
 * The (lhs_mean, lhs_sigma2, lhs_count) will be merged into (rhs_mean, rhs_sigma2, rhs_count)
 *
 * See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance for more details.
 *
 *  TODO(sxjscience) Explore the possibility of int lhs_count and rhs_count
 */
template<typename DType>
__device__ void chan_merge_partition(const DType lhs_mean,
                                     const DType lhs_sigma2,
                                     const DType lhs_count,
                                     DType& rhs_mean,
                                     DType& rhs_sigma2,
                                     DType& rhs_count) {
  DType delta = lhs_mean - rhs_mean;
  DType nA = lhs_count;
  DType nB = rhs_count;
  rhs_count = nA + nB;
  if (rhs_count > DType(0)) {
    nA = nA / rhs_count;
    nB = nB / rhs_count;
    rhs_mean = nA * lhs_mean + nB * rhs_mean;
    rhs_sigma2 = rhs_sigma2 + lhs_sigma2 + delta * delta * nA * nB * rhs_count;
  } else {
    rhs_mean = DType(0);
    rhs_sigma2 = DType(0);
  }
}

/* Use the Chan's Parallel Algorithm to merge all (mean, sigma2, counts) within a warp of threads.
 * After calling the function, threadIdx.x == 0 will store the result of the
 * aggregated (mean, sigma2, counts).
 *
 *
 */
template<typename DType>
__device__ void warp_merge_mean_sigma2(DType mean, DType sigma2, DType count) {
  for (int l = 0; l <= 4; ++l) {
    int src_lane = (threadIdx.x + (1<<l)) & 31;
    DType meanB = WARP_SHFL(mean, src_lane);
    DType sigma2B = WARP_SHFL(sigma2, src_lane);
    DType countB = WARP_SHFL(count, src_lane);
    chan_merge_partition(meanB, sigma2B, countB, mean, sigma2, count);
  }
}


/* Fused CUDA kernel for layer normalization. It computes the LayerNorm when axis=-1.
 * Shape of the input tensors:
 *      in_data = (nbatch, nchannel)
 *      gamma = (nchannel,)
 *      beta = (nchannel,)
 *      out_data = (nchannel,)
 *      mean_data = (nbatch,)
 *      var_data = (nbatch,)
 *  It's always launched with (blockDim.x, blockDim.y) = (WARP_SIZE, blockDim.y)
 *  Also, when blockDim.y > 1, it requires shared memory that has size:
 *      sizeof(DType) * blockDim.y + sizeof(DType) * blockDim.y / 2
 */
template<typename DType>
__global__ void LayerNormFusedForwardKernelContig(const int nbatch,
                                                  const int nchannel,
                                                  const float eps,
                                                  const DType* in_data,
                                                  const DType* gamma,
                                                  const DType* beta,
                                                  DType* out_data,
                                                  DType* mean_data,
                                                  DType* std_data) {
  int bid = blockIdx.x + blockIdx.y * gridDim.x;
  const int nthread = blockDim.x * blockDim.y;
  DType count = 0;
  DType mean = 0;
  DType sigma2 = 0;
  const int N_ACCUM = 4;  // TODO(sxjscience) Profile
  const int TY = blockDim.y;
  extern __shared__ char buf[];  // Shared memory size

  if (bid < nbatch) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const DType* col_vals = in_data + bid * nchannel;
    // Each thread takes charge of 4 consecutive numbers
    // To minimize branch divergence, we split the for-loop into two parts.
    int l = N_ACCUM * tid;
    for (; l + N_ACCUM - 1 < nchannel; l += N_ACCUM * nthread) {
#pragma unroll
      for (int i = 0; i < N_ACCUM; ++i) {
        welford_online_sum_step(col_vals[l + i], mean, sigma2, count);
      }
    }
    for(; l < nchannel; ++l) {
      welford_online_sum_step(col_vals[l], mean, sigma2, count);
    }
    // Merge the mean/sigma2 within a warp
    // threadIdx.x == 0 will store the reduction result.
    warp_merge_mean_sigma2(mean, sigma2, count);
    if (blockDim.y == 1) {
      mean = WARP_SHFL(mean, 0);
      sigma2 = WARP_SHFL(sigma2 / nchannel, 0); // Calculate the variance
    } else {
      // Inter-warp reduction. Copy the upper-half of the warps to shared memory
      // and merge with the lower-half warp
      DType* mean_buf = reinterpret_cast<DType*>(buf);
      DType* sigma2_buf = reinterpret_cast<DType*>(buf + sizeof(DType) * blockDim.y / 2);
      DType* count_buf = reinterpret_cast<DType*>(buf + sizeof(DType) * blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int idx = threadIdx.y - offset;
          mean_buf[idx] = mean;
          sigma2_buf[idx] = sigma2;
          count_buf[idx] = count;
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          chan_merge_partition(mean_buf[threadIdx.y],
                               sigma2_buf[threadIdx.y],
                               count_buf[threadIdx.y], mean, sigma2, count);
        }
        __syncthreads();
      }
      // Broadcast the result to all threads
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        mean_buf[0] = mean;
        sigma2_buf[0] = sigma2;
        count_buf[0] = count;
      }
      __syncthreads();
      mean = mean_buf[0];
      sigma2 = sigma2_buf[0] / nchannel;
    }
    // Calculate the out_data: gamma * (x - mean) / sqrt(var + eps) + beta
    DType std_eps = sqrt(sigma2 + eps);
    DType invstd_eps = static_cast<DType>(1) / std_eps;
    if (gamma != NULL && beta != NULL) {
      for (int i = tid; i < nchannel; i += nthread) {
        out_data[bid * nchannel + i] =
          gamma[i] * invstd_eps * (in_data[bid * nchannel + i] - mean) + beta[i];
      }
    } else if (gamma == NULL && beta != NULL) {
      for (int i = tid; i < nchannel; i += nthread) {
        out_data[bid * nchannel + i] = invstd_eps * (in_data[bid * nchannel + i] - mean) + beta[i];
      }
    } else if (gamma != NULL && beta == NULL) {
      for (int i = tid; i < nchannel; i += nthread) {
        out_data[bid * nchannel + i] =
          gamma[i] * invstd_eps * (in_data[bid * nchannel + i] - mean);
      }
    } else {
      for (int i = tid; i < nchannel; i += nthread) {
        out_data[bid * nchannel + i] = invstd_eps * (in_data[bid * nchannel + i] - mean);
      }
    }
    // Write the out_data and var_data
    if(threadIdx.x == 0 && threadIdx.y == 0) {
      out_data[bid] = mean;
      std_data[bid] = std_eps;
    }
  }
}

void LayerNormGPUContig(const LayerNormParam param,
                        const OpContext& ctx, const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 3U);
  mxnet::TShape data_shape(2);
  mxnet::TShape mean_shape(1);
  size_t in_ndim = inputs[layernorm::kData].ndim();
  data_shape[0] = mean_shape[0] = inputs[layernorm::kData].shape_.ProdShape(0, in_ndim - 1);
  data_shape[1] = inputs[layernorm::kData].shape_[in_ndim - 1];
  const TBlob in_data = inputs[layernorm::kData].reshape(data_shape);
  const TBlob gamma = inputs[layernorm::kGamma];
  const TBlob beta = inputs[layernorm::kBeta];
  const TBlob out_data = outputs[layernorm::kOut].reshape(data_shape);
  const TBlob mean_data = outputs[layernorm::kMean].reshape(mean_shape);
  const TBlob std_data = outputs[layernorm::kStd].reshape(mean_shape);
  // Make sure the inputs are contiguous
  CHECK_EQ(in_data.CheckContiguous(), true);
  CHECK_EQ(gamma.CheckContiguous(), true);
  CHECK_EQ(beta.CheckContiguous(), true);
  CHECK_EQ(out_data.CheckContiguous(), true);
  CHECK_EQ(mean_data.CheckContiguous(), true);
  CHECK_EQ(std_data.CheckContiguous(), true);

  // Lauch the kernel. The dynamic shared memory size is sizeof(DType) * threadDim.y + sizeof(DType) *
  int nbatch = data_shape[0];
  int nchannel = data_shape[1];
  float eps = param.eps;
  int ngrid_x = (nbatch > kMaxGridDim) ? (nbatch + kBaseGridNum - 1) / kBaseGridNum : nbatch;
  int ngrid_y = (nbatch > kMaxGridDim) ? kBaseGridNum : 1;
  int nthread_y = 0;
  const dim3 dimGrid(ngrid_x, ngrid_y, 1);
  if(nchannel <= 32) {
    nthread_y = 1;
  } else if(nchannel <= 64) {
    nthread_y = 2;
  } else {
    nthread_y = 4;
  }
  const dim3 dimBlock(32, nthread_y, 1);
  MSHADOW_REAL_TYPE_SWITCH(in_data.type_flag_, DType, {
    int nshared = nthread_y > 1 ? nthread_y * sizeof(DType) + (nthread_y / 2) * sizeof(DType) : 0;
    cudaStream_t stream = Stream<gpu>::GetStream(ctx.get_stream<gpu>());
    CheckLaunchParam(dimGrid, dimBlock);
    LayerNormFusedForwardKernelContig<<<dimBlock, dimGrid, nshared, stream>>>
     (nbatch, nchannel, eps,
      in_data.dptr<DType>(), gamma.dptr<DType>(), beta.dptr<DType>(),
      out_data.dptr<DType>(), mean_data.dptr<DType>(), std_data.dptr<DType>());
    MSHADOW_CUDA_POST_KERNEL_CHECK(LayerNormFusedForwardKernelContig);
  });
}

template<>
void LayerNormCompute<gpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx, const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  if (req[0] == kNullOp) return;
  CHECK_NE(req[0], kAddTo);
  int axis = param.axis;
  if (axis < 0) {
    axis += static_cast<int>(inputs[0].ndim());
  }
  CHECK(axis >= 0 && axis < inputs[0].ndim()) << "Channel axis out of range: " << param.axis;
  if(axis == inputs[0].ndim() - 1) {
    // Try to use the accelerated CUDA kernels
    return LayerNormGPUContig(param, ctx, inputs, req, outputs);
  }
  return LayerNormComputeGeneral<gpu>(attrs, ctx, inputs, req, outputs);
}

NNVM_REGISTER_OP(LayerNorm)
.set_attr<FCompute>("FCompute<gpu>", LayerNormCompute<gpu>);

NNVM_REGISTER_OP(_backward_LayerNorm)
.set_attr<FCompute>("FCompute<gpu>", LayerNormGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

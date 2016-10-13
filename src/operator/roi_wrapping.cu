/*!
 * Copyright (c) 2015 by Contributors
 * \file roi_pooling.cu
 * \brief roi pooling operator
 * \author Ross Girshick, Kye-Hyeon Kim, Jian Guo
*/
#include "./roi_wrapping-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>


namespace mshadow {
namespace cuda {

__device__ bool between(int value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}

__device__ int get_address_BCHW(int b, int c, int h, int w,
                                int channel_num, int height, int width) {
  return ((b * channel_num + c) * height + h) * width + w;
}

template<bool explicit_batch, typename DType>
__device__ void parse_roi_coords(int n, const DType * rois, DType spatial_scale,
                                 int &batch_ind, DType &x1, DType &y1, DType &x2, DType &y2) {
  if (explicit_batch) {
    int shift = 5 * n;
    batch_ind = rois[shift];
    x1 = rois[shift + 1] * spatial_scale;
    y1 = rois[shift + 2] * spatial_scale;
    x2 = rois[shift + 3] * spatial_scale;
    y2 = rois[shift + 4] * spatial_scale;
  }
  else {
    int shift = 4 * n;
    x1 = rois[shift] * spatial_scale;
    y1 = rois[shift + 1] * spatial_scale;
    x2 = rois[shift + 2] * spatial_scale;
    y2 = rois[shift + 3] * spatial_scale;
  }
}
template<bool explicit_batch, typename DType>
__global__ void BilinearPoolForwardKernel(const int count, const DType* bottom_data,
                                          const DType spatial_scale,
                                          const int batch_num, const int channels,
                                          const int height, const int width,
                                          const int pooled_height, const int pooled_width,
                                          const DType* bottom_rois, DType* top_data) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, py, px) is an element in the pooled output
    int px = index % pooled_width;
    int py = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    // parse rois
    int batch_ind;
    DType roi_x1, roi_y1, roi_x2, roi_y2;
    parse_roi_coords<explicit_batch>(n, bottom_rois, spatial_scale,
                                     batch_ind, roi_x1, roi_y1, roi_x2, roi_y2);
    // output zero for negative batch inds
    if (batch_ind < 0) {
      top_data[index] = 0;
      continue;
    }
    // Force malformed ROIs to be 1x1
    DType roi_width = max(roi_x2 - roi_x1 + 1.0, 1.0);
    DType roi_height = max(roi_y2 - roi_y1 + 1.0, 1.0);
    DType bottom_x = static_cast<DType>(px) * roi_width / static_cast<DType>(pooled_width) + roi_x1;
    DType bottom_y = static_cast<DType>(py)* roi_height / static_cast<DType>(pooled_height) + roi_y1;
    // Get the topleft coordinates and the corresponding deltas, x - floor(x) and y - floor(y)
    int topleft_x, topleft_y;
    float topleft_dx, topleft_dy;
    topleft_x = floor(bottom_x);
    topleft_y = floor(bottom_y);
    topleft_dx = bottom_x - static_cast<DType>(topleft_x);
    topleft_dy = bottom_y - static_cast<DType>(topleft_y);
    int topleft_ind = get_address_BCHW(batch_ind, c, topleft_y, topleft_x,
                                       channels, height, width);
    int topright_ind = get_address_BCHW(batch_ind, c, topleft_y, topleft_x + 1,
                                        channels, height, width);
    int bottomleft_ind = get_address_BCHW(batch_ind, c, topleft_y + 1, topleft_x,
                                          channels, height, width);
    int bottomright_ind = get_address_BCHW(batch_ind, c, topleft_y + 1, topleft_x + 1,
                                           channels, height, width);
    DType topleft_v = topleft_ind < bottom_count ? bottom_data[topleft_ind] : 0;
    DType topright_v = topright_ind < bottom_count ? bottom_data[topright_ind] : 0;
    DType bottomleft_v = bottomleft_ind < bottom_count ? bottom_data[bottomleft] : 0;
    DType bottomright_v = bottomright_ind < bottom_count ? bottom_data[bottomright_ind] : 0;
    top_data[index] = (1 - topleft_dx) * (1 - topleft_dy) * topleft_v
                      + (1 - topleft_dx) * topleft_dy * topright_v
                      + topleft_dx * (1 - topleft_dy) * bottomleft_v
                      + topleft_dx * topleft_dy * bottomright_v;
  }
}

template<typename DType>
inline void BilinearPoolForward(const Tensor<gpu, 4, DType> &out,
                                const Tensor<gpu, 4, DType> &data,
                                const Tensor<gpu, 2, DType> &bbox,
                                DType spatial_scale,
                                bool explicit_batch) {
  const DType *bottom_data = data.dptr_;
  const DType *bottom_rois = bbox.dptr_;
  DType *top_data = out.dptr_;
  const int count = out.shape_.Size();
  const int batch_num = data.size(0);
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  const int grid_dim_x = (gridSize > kMaxGridNum) ? kMaxGridNum : gridSize;
  const int grid_dim_y = (gridSize > kMaxGridNum) ? (gridSize + kMaxGridNum - 1) / kMaxGridNum : 1;
  dim3 dimGrid(grid_dim_x, grid_dim_y);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "BilinearPooling Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  if (explicit_batch) {
    BilinearPoolForwardKernel<true, DType> <<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, batch_num, channels, height, width,
      pooled_height, pooled_width, bottom_rois, top_data);
  }
  else {
    BilinearPoolForwardKernel<false, Dtype> <<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, batch_num, channels, height, width,
      pooled_height, pooled_width, bottom_rois, top_data);
  }
}

template<bool explicit_batch, typename DType>
__global__ void BilinearPoolBackwardAccDataKernel(const int count, const DType* top_diff,
                                                  const int num_rois, const DType spatial_scale,
                                                  const int channels,
                                                  const int height, const int width,
                                                  const int pooled_height, const int pooled_width,
                                                  DType* bottom_data_diff, const DType* bottom_rois) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, py, px) is an element in the pooled output
    int px = index % pooled_width;
    int py = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    // parse rois
    int batch_ind;
    DType roi_x1, roi_y1, roi_x2, roi_y2;
    parse_roi_coords<explicit_batch>(n, bottom_rois, spatial_scale, explicit_batch,
                                     batch_ind, roi_x1, roi_y1, roi_x2, roi_y2);
    // do not accumulate gradient for negative batch inds
    if (batch_ind < 0) {
      continue;
    }
    // Force malformed ROIs to be 1x1
    DType roi_width = max(roi_x2 - roi_x1 + 1.0, 1.0);
    DType roi_height = max(roi_y2 - roi_y1 + 1.0, 1.0);
    DType bottom_x = static_cast<DType>(px)* roi_width / static_cast<DType>(pooled_width) + roi_x1;
    DType bottom_y = static_cast<DType>(py)* roi_height / static_cast<DType>(pooled_height) + roi_y1;
    // Get the topleft coordinates and the corresponding deltas, x - floor(x) and y - floor(y)
    int topleft_x, topleft_y;
    DType topleft_dx, topleft_dy;
    topleft_x = floor(bottom_x);
    topleft_y = floor(bottom_y);
    topleft_dx = bottom_x - static_cast<DType>(topleft_x);
    topleft_dy = bottom_y - static_cast<DType>(topleft_y);
    int topleft_ind = get_address_BCHW(batch_ind, c, topleft_y, topleft_x,
                                       channels, height, width);
    int topright_ind = get_address_BCHW(batch_ind, c, topleft_y, topleft_x + 1,
                                        channels, height, width);
    int bottomleft_ind = get_address_BCHW(batch_ind, c, topleft_y + 1, topleft_x,
                                          channels, height, width);
    int bottomright_ind = get_address_BCHW(batch_ind, c, topleft_y + 1, topleft_x + 1,
                                           channels, height, width);
    // Calculate data_grad, use atomic_add to backpropagate the out_grad
    DType ograd = top_diff[index];
    if (topleft_ind < count) {
      atomicAdd(bottom_data_diff + topleft_ind, (1 - topleft_dx) * (1 - topleft_dy) * ograd);
    }
    if (topright_ind < count) {
      atomicAdd(bottom_data_diff + topright_ind, (1 - topleft_dx) * topleft_dy * ograd);
    }
    if (bottomleft_ind < count) {
      atomicAdd(bottom_data_diff + bottomleft_ind, topleft_dx * (1 - topleft_dy) * ograd);
    }
    if (bottomright_ind < count) {
      atomicAdd(bottom_data_diff + bottomright_ind, topleft_dx * topleft_dy * ograd);
    }
  }
}

template<int warp_bits, bool explicit_batch, typename DType>
__global__ void BilinearPoolBackwardAccROIKernel(const int count, const DType* top_diff,
                                                 const int num_rois, const DType spatial_scale,
                                                 const int channels,
                                                 const int height, const int width,
                                                 const int pooled_height, const int pooled_width,
                                                 DType* bottom_roi_diff, const DType* bottom_rois,
                                                 const DType* bottom_data) {
  const int warp_size = 1 << warp_bits;
  // We use the y-threads to traverse the grids and use x-threads to traverse the channels.
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.y + threadIdx.y;
       index < count;
       index += blockDim.y * gridDim.x * gridDim.y) {
    // (n, py, px) is an element in the pooled output
    int px = index % pooled_width;
    int py = (index / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height;
    // parse rois
    int batch_ind;
    DType roi_x1, roi_y1, roi_x2, roi_y2;
    parse_roi_coords<explicit_batch>(n, bottom_rois, spatial_scale,
                                     batch_ind, roi_x1, roi_y1, roi_x2, roi_y2);
    // Do not accumulate gradient for negative batch inds
    if (batch_ind < 0) {
      continue;
    }
    // Do not backpropage gradient for malformed ROIs
    if ((roi_x1 < roi_x2) || (roi_y1 < roi_y2)) {
      continue;
    }
    DType roi_width = roi_x2 - roi_x1 + 1.0;
    DType roi_height = roi_y2 - roi_y1 + 1.0;
    DType x_ratio = static_cast<DType>(px) / static_cast<DType>(pooled_width);
    DType y_ratio = static_cast<DType>(py) / static_cast<DType>(pooled_height);
    DType bottom_x = x_ratio * roi_width + roi_x1;
    DType bottom_y = y_ratio * roi_height + roi_y1;
    // Get the topleft coordinates and the corresponding deltas, x - floor(x) and y - floor(y)
    int topleft_x, topleft_y;
    DType topleft_dx, topleft_dy;
    topleft_x = floor(bottom_x);
    topleft_y = floor(bottom_y);
    topleft_dx = bottom_x - static_cast<DType>topleft_x;
    topleft_dy = bottom_y - static_cast<DType>topleft_y;
    DType grad_dx = 0;
    DType grad_dy = 0;
    // Use the x-threads to traverse the channels
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
      int topleft_ind = get_address_BCHW(batch_ind, c, topleft_y, topleft_x,
                                         channels, height, width);
      int topright_ind = get_address_BCHW(batch_ind, c, topleft_y, topleft_x + 1,
                                          channels, height, width);
      int bottomleft_ind = get_address_BCHW(batch_ind, c, topleft_y + 1, topleft_x,
                                            channels, height, width);
      int bottomright_ind = get_address_BCHW(batch_ind, c, topleft_y + 1, topleft_x + 1,
                                             channels, height, width);
      int outgrad_index = get_address_BCHW(n, c, py, px,
                                           channels, pooled_height, pooled_width);
      DType ograd = top_diff[outgrad_index];
      DType topleft_v = 0;
      DType topright_v = 0;
      DType bottomleft_v = 0;
      DType bottomright_v = 0;
      if (topleft_ind < count) {
        topleft_v = bottom_data[topleft_ind];
      }
      if (topright_ind < count) {
        topright_v = bottom_data[topright_ind];
      }
      if (bottomleft_ind < count) {
        bottomleft_v = bottom_data[bottomleft_ind];
      }
      if (bottomright_ind < count) {
        bottomright_v = bottom_data[bottomright_ind];
      }
      grad_dx += ((1 - topleft_dy) * (bottomleft_v - topleft_v) +
                 topleft_dy * (bottomright_v - topright_v));
      grad_dy += ((1 - topleft_dx) * (topright_v - topleft_v) +
                 topleft_dx * (bottomright_v - topright_v));
      grad_dx *= ograd;
      grad_dy *= ograd;
    }
    __shared__ volatile DType __shmem[warp_size][warp_size];
    __shmem[threadIdx.y][threadIdx.x] = grad_dx;
    __syncthreads();
    Reduce1D<mshadow::red::sum, warp_bits>(__shmem[threadIdx.y]);
    __syncthreads();
    grad_dx = __shmem[threadIdx.y][0];
    __shmem[threadIdx.y][threadIdx.x] = grad_dy;
    __syncthreads();
    Reduce1D<mshadow::red::sum, warp_bits>(__shmem[threadIdx.y]);
    __syncthreads();
    grad_dy = __shmem[threadIdx.y][0];
    DType grad_x1 = (1 - x_ratio) * grad_dx;
    DType grad_y1 = (1 - y_ratio) * grad_dy;
    DType grad_x2 = x_ratio * grad_dx;
    DType grad_y2 = y_ratio * grad_dy;
    if (threadIdx.x == 0) {
      if (explicit_batch) {
        bottom_roi_diff[5 * n + 1] += grad_x1;
        bottom_roi_diff[5 * n + 2] += grad_y1;
        bottom_roi_diff[5 * n + 3] += grad_x2;
        bottom_roi_diff[5 * n + 4] += grad_y2;
      } else {
        bottom_roi_diff[4 * n + 0] += grad_x1;
        bottom_roi_diff[4 * n + 1] += grad_y1;
        bottom_roi_diff[4 * n + 2] += grad_x2;
        bottom_roi_diff[4 * n + 3] += grad_y2;
      }
    }
  }
}

inline void BilinearPoolBackwardAccData(const Tensor<gpu, 4, float> &in_data_grad,
                                        const Tensor<gpu, 4, float> &out_grad,
                                        const Tensor<gpu, 2, float> &bbox,
                                        float spatial_scale,
                                        bool explicit_batch) {
  const float *top_diff = out_grad.dptr_;
  const float *bottom_rois = bbox.dptr_;
  float *bottom_data_diff = in_data_grad.dptr_;
  const int count = out_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int channels = in_data_grad.size(1);
  const int height = in_data_grad.size(2);
  const int width = in_data_grad.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  const int grid_dim_x = (gridSize > kMaxGridNum) ? kMaxGridNum : gridSize;
  const int grid_dim_y = (gridSize > kMaxGridNum) ? (gridSize + kMaxGridNum - 1) / kMaxGridNum : 1;
  dim3 dimGrid(grid_dim_x, grid_dim_y);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "BilinearPooling Backward Data");
  cudaStream_t stream = Stream<gpu>::GetStream(in_data_grad.stream_);
  if (explicit_batch) {
    BilinearPoolBackwardAccDataKernel<true, float> <<<dimGrid, dimBlock, 0, stream >> >(
      count, top_diff, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_data_diff, bottom_rois);
  } else {
    BilinearPoolBackwardAccDataKernel<false, float> <<<dimGrid, dimBlock, 0, stream >> >(
      count, top_diff, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_data_diff, bottom_rois);
  }
  
}

inline void BilinearPoolBackwardAccROI(const Tensor<gpu, 2, float> &in_roi_grad,
                                       const Tensor<gpu, 4, float> &out_grad,
                                       const Tensor<gpu, 2, float> &bbox,
                                       const Tensor<gpu, 4, float> &data,
                                       float spatial_scale,
                                       bool explicit_batch) {
  const float *top_diff = out_grad.dptr_;
  const float *bottom_rois = bbox.dptr_;
  const float *bottom_data = data.dptr_;
  float *bottom_rois_diff = in_roi_grad.dptr_;
  const int count = out_grad.size(0) * out_grad.size(2) * out_grad.size(3);
  const int num_rois = bbox.size(0);
  const int channels = out_grad.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMemUnit;
  const int grid_dim_x = (gridSize > kMaxGridNum) ? kMaxGridNum : gridSize;
  const int grid_dim_y = (gridSize > kMaxGridNum) ? (gridSize + kMaxGridNum - 1) / kMaxGridNum : 1;
  dim3 dimGrid(grid_dim_x, grid_dim_y);
  dim3 dimBlock(kMemUnit, kMemUnit);
  CheckLaunchParam(dimGrid, dimBlock, "BilinearPooling Data Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_roi_grad.stream_);
  if (explicit_batch) {
    BilinearPoolBackwardAccROIKernel<true, float> <<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois_diff, bottom_rois, bottom_data);
  } else {
    BilinearPoolBackwardAccROIKernel<false, float> <<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois_diff, bottom_rois, bottom_data);
  }
}

}  // namespace cuda

template<typename DType>
inline void BilinearPoolForward(const Tensor<gpu, 4, DType> &out,
                                const Tensor<gpu, 4, DType> &data,
                                const Tensor<gpu, 2, DType> &bbox,
                                DType spatial_scale,
                                bool explicit_batch) {
  cuda::BilinearPoolForward(out, data, bbox, spatial_scale, explicit_batch);
}

inline void BilinearPoolBackwardAccData(const Tensor<gpu, 4, float> &in_data_grad,
                                        const Tensor<gpu, 4, float> &out_grad,
                                        const Tensor<gpu, 2, float> &bbox,
                                        float spatial_scale,
                                        bool explicit_batch) {
  cuda::BilinearPoolBackwardAccData(in_data_grad, out_grad, bbox, spatial_scale);
}

inline void BilinearPoolBackwardAccROI(const Tensor<gpu, 2, float> &in_roi_grad,
                                       const Tensor<gpu, 4, float> &out_grad,
                                       const Tensor<gpu, 2, float> &bbox,
                                       const Tensor<gpu, 4, float> &data,
                                       float spatial_scale,
                                       bool explicit_batch) {
  cuda::BilinearPoolBackwardAccROI(in_roi_grad, out_grad, bbox, data,
                                   spatial_scale, explicit_batch);
}
}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ROIWrappingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIWrappingOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

/*!
 * Copyright (c) 2016 by Contributors
 * \file roi_wrapping.cu
 * \brief roi wrapping operator
 * \author Xingjian Shi
*/
#include "./roi_wrapping-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>


namespace mshadow {
namespace cuda {

template<typename DType>
__device__ bool between(DType value, DType lowerBound, DType upperBound) {
  return (value >= lowerBound && value < upperBound);
}

__device__ int get_address_BCHW(int b, int c, int h, int w,
                                int channel_num, int height, int width) {
  return ((b * channel_num + c) * height + h) * width + w;
}

template<bool anti_aliasing, typename DType>
__device__ void get_kernel_begin_end(DType x, int kernel_size, DType scale_x, int width,
                                     int &begin_x, int &end_x) {
  if (anti_aliasing && scale_x > 1) {
    begin_x = static_cast<int>(floor(x - kernel_size * scale_x)) + 1;
    end_x = static_cast<int>(ceil(x + kernel_size * scale_x)) - 1;
  } else {
    begin_x = static_cast<int>(floor(x)) - kernel_size + 1;
    end_x = static_cast<int>(ceil(x)) + kernel_size - 1;
  }
  begin_x = max(begin_x, 0);
  end_x = min(end_x, width - 1);
}

template<typename DType>
__device__ void parse_roi_coords(int n, const DType * rois, float spatial_scale, bool explicit_batch,
                                 int &batch_ind, DType &x1, DType &y1, DType &x2, DType &y2) {
  if (explicit_batch) {
    int shift = 5 * n;
    batch_ind = static_cast<int>(rois[shift]);
    x1 = rois[shift + 1] * spatial_scale;
    y1 = rois[shift + 2] * spatial_scale;
    x2 = rois[shift + 3] * spatial_scale;
    y2 = rois[shift + 4] * spatial_scale;
  }
  else {
    int shift = 4 * n;
    batch_ind = n;
    x1 = rois[shift] * spatial_scale;
    y1 = rois[shift + 1] * spatial_scale;
    x2 = rois[shift + 2] * spatial_scale;
    y2 = rois[shift + 3] * spatial_scale;
  }
}
template<typename InterpOp, bool anti_aliasing, typename DType>
__global__ void ROIWrapForwardKernel(const int count, DType* top_data,
                                     const DType* bottom_data, const DType* bottom_rois,
                                     const int batch_num, const int channels,
                                     const int height, const int width,
                                     const int pooled_height, const int pooled_width,
                                     const float spatial_scale,
                                     const bool explicit_batch,
                                     const int interp_kernel_size) {
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
    parse_roi_coords(n, bottom_rois, spatial_scale, explicit_batch,
                     batch_ind, roi_x1, roi_y1, roi_x2, roi_y2);
    // output zero for negative batch inds
    if (batch_ind < 0) {
      top_data[index] = 0;
      continue;
    }
    // Force malformed ROIs to be 1x1
    DType roi_width = max(roi_x2 - roi_x1 + DType(1.0), DType(1.0));
    DType roi_height = max(roi_y2 - roi_y1 + DType(1.0), DType(1.0));
    DType bottom_x = static_cast<DType>(2 * px - pooled_width + 1) * roi_width / static_cast<DType>(2 * pooled_width)
                     + (roi_x1 + roi_x2) / DType(2.0);
    DType bottom_y = static_cast<DType>(2 * py - pooled_height + 1) * roi_height / static_cast<DType>(2 * pooled_height)
                     + (roi_y1 + roi_y2) / DType(2.0);
    DType scale_x = roi_width / static_cast<DType>(pooled_width);
    DType scale_y = roi_height / static_cast<DType>(pooled_height);
    // Get the begin and end indices of the interpolation kernel
    int begin_x, begin_y, end_x, end_y;
    DType acc_weight = 0;
    get_kernel_begin_end<anti_aliasing>(bottom_x, interp_kernel_size, scale_x, width, begin_x, end_x);
    get_kernel_begin_end<anti_aliasing>(bottom_y, interp_kernel_size, scale_y, height, begin_y, end_y);

    for (int kx = begin_x; kx <= end_x; ++kx) {
      DType dx = bottom_x - static_cast<DType>(kx);
      for (int ky = begin_y; ky <= end_y; ++ky) {
        int ind = get_address_BCHW(batch_ind, c, ky, kx, channels, height, width);
        DType dy = bottom_y - static_cast<DType>(ky);
        DType weight = InterpOp::Val(dx, scale_x) * InterpOp::Val(dy, scale_y);
        acc_weight += weight;
        top_data[index] += weight * bottom_data[ind];
      }
    }
    if (0 != acc_weight) {
      top_data[index] /= acc_weight;
    } else {
      top_data[index] = 0;
    }
  }
}

template<typename InterpOp, bool anti_aliasing, typename DType>
inline void ROIWrapForward(const Tensor<gpu, 4, DType> &out,
                                const Tensor<gpu, 4, DType> &data,
                                const Tensor<gpu, 2, DType> &bbox,
                                float spatial_scale,
                                bool explicit_batch,
                                int interp_kernel_size) {
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
  CheckLaunchParam(dimGrid, dimBlock, "ROIWrapping Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  ROIWrapForwardKernel<InterpOp, anti_aliasing, DType> << <dimGrid, dimBlock, 0, stream >> >(
    count, top_data, bottom_data, bottom_rois, batch_num, channels, height, width,
    pooled_height, pooled_width, spatial_scale, explicit_batch, interp_kernel_size);
  cudaError err = cudaPeekAtLastError();
  CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

template<typename InterpOp, bool anti_aliasing, typename DType>
__global__ void ROIWrapBackwardAccDataKernel(const int count, DType* bottom_data_diff,
                                             const DType* bottom_rois, const DType* top_diff,
                                             const int batch_num, const int channels,
                                             const int height, const int width,
                                             const int pooled_height, const int pooled_width,
                                             const float spatial_scale,
                                             const bool explicit_batch,
                                             const int interp_kernel_size) {
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
    parse_roi_coords(n, bottom_rois, spatial_scale, explicit_batch,
                     batch_ind, roi_x1, roi_y1, roi_x2, roi_y2);
    // do not accumulate gradient for negative batch inds
    if (batch_ind < 0) {
      continue;
    }
    // Force malformed ROIs to be 1x1
    DType roi_width = max(roi_x2 - roi_x1 + DType(1.0), DType(1.0));
    DType roi_height = max(roi_y2 - roi_y1 + DType(1.0), DType(1.0));
    DType bottom_x = static_cast<DType>(2 * px - pooled_width + 1) * roi_width / static_cast<DType>(2 * pooled_width)
                     + (roi_x1 + roi_x2) / DType(2.0);
    DType bottom_y = static_cast<DType>(2 * py - pooled_height + 1) * roi_height / static_cast<DType>(2 * pooled_height)
                     + (roi_y1 + roi_y2) / DType(2.0);
    DType scale_x = roi_width / static_cast<DType>(pooled_width);
    DType scale_y = roi_height / static_cast<DType>(pooled_height);
    // Get the begin and end indices of the interpolation kernel
    int begin_x, begin_y, end_x, end_y;
    DType acc_weight_x = 0;
    DType acc_weight_y = 0;
    DType acc_weight = 0;
    get_kernel_begin_end<anti_aliasing>(bottom_x, interp_kernel_size, scale_x, width, begin_x, end_x);
    get_kernel_begin_end<anti_aliasing>(bottom_y, interp_kernel_size, scale_y, height, begin_y, end_y);
    // first calculate the accumulate weight
    for (int kx = begin_x; kx <= end_x; ++kx) {
      DType dx = bottom_x - static_cast<DType>(kx);
      acc_weight_x += InterpOp::Val(dx, scale_x);
    }
    for (int ky = begin_y; ky <= end_y; ++ky) {
      DType dy = bottom_y - static_cast<DType>(ky);
      acc_weight_y += InterpOp::Val(dy, scale_y);
    }
    if (0 == acc_weight_x || 0 == acc_weight_y) {
      continue;
    }
    acc_weight = acc_weight_x * acc_weight_y;
    // use atomicadd to backpropage gradient
    for (int kx = begin_x; kx <= end_x; ++kx) {
      DType dx = bottom_x - static_cast<DType>(kx);
      for (int ky = begin_y; ky <= end_y; ++ky) {
        int ind = get_address_BCHW(batch_ind, c, ky, kx, channels, height, width);
        DType dy = bottom_y - static_cast<DType>(ky);
        atomicAdd(&bottom_data_diff[ind],
          InterpOp::Val(dx, scale_x) * InterpOp::Val(dy, scale_y)
          / acc_weight * top_diff[index]);
      }
    }
  }
}

template<int warp_bits, typename InterpOp, bool anti_aliasing, typename DType>
__global__ void ROIWrapBackwardAccROIKernel(const int count, DType* bottom_roi_diff,
                                            const DType* bottom_data, const DType* bottom_rois, 
                                            const DType* top_diff,
                                            const int batch_num, const int channels,
                                            const int height, const int width,
                                            const int pooled_height, const int pooled_width,
                                            const float spatial_scale,
                                            const bool explicit_batch,
                                            const int interp_kernel_size) {
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
    parse_roi_coords(n, bottom_rois, spatial_scale, explicit_batch,
                     batch_ind, roi_x1, roi_y1, roi_x2, roi_y2);
    // Do not accumulate gradient for negative batch inds
    if (batch_ind < 0) {
      continue;
    }
    // Do not backpropage gradient for malformed ROIs
    if ((roi_x1 > roi_x2) || (roi_y1 > roi_y2)) {
      continue;
    }
    DType roi_width = roi_x2 - roi_x1 + 1.0;
    DType roi_height = roi_y2 - roi_y1 + 1.0;
    DType x_relative = static_cast<DType>(2 * px - pooled_width + 1) / static_cast<DType>(2 * pooled_width);
    DType y_relative = static_cast<DType>(2 * py - pooled_height + 1) / static_cast<DType>(2 * pooled_height);
    DType bottom_x = x_relative * roi_width + DType(0.5) * (roi_x1 + roi_x2);
    DType bottom_y = y_relative * roi_height + DType(0.5) * (roi_y1 + roi_y2);
    DType scale_x = roi_width / static_cast<DType>(pooled_width);
    DType scale_y = roi_height / static_cast<DType>(pooled_height);
    // Get the begin and end indices of the interpolation kernel
    int begin_x, begin_y, end_x, end_y;
    get_kernel_begin_end<anti_aliasing>(bottom_x, interp_kernel_size, scale_x, width, begin_x, end_x);
    get_kernel_begin_end<anti_aliasing>(bottom_y, interp_kernel_size, scale_y, height, begin_y, end_y);
    // Initialize the aggregation weights and gradients.
    // We will compute the gradient w.r.t the corresponding bottom coordinates and w.r.t the scale separately.
    // Since we are using 
    DType acc_weight_x = 0;
    DType acc_weight_y = 0;
    DType acc_grad_bottom_x = 0;
    DType acc_grad_bottom_y = 0;
    DType acc_grad_scale_x = 0;
    DType acc_grad_scale_y = 0;
    DType grad_bottom_x = 0;
    DType grad_bottom_y = 0;
    DType grad_scale_x = 0;
    DType grad_scale_y = 0;
    for (int kx = begin_x; kx <= end_x; ++kx) {
      DType dx = bottom_x - static_cast<DType>(kx);
      acc_weight_x += InterpOp::Val(dx, scale_x);
      DType dx_grad = InterpOp::Grad(dx, scale_x);
      acc_grad_bottom_x += dx_grad;
      acc_grad_scale_x += InterpOp::GradS(dx, scale_x, dx_grad);
    }
    for (int ky = begin_y; ky <= end_y; ++ky) {
      DType dy = bottom_y - static_cast<DType>(ky);
      acc_weight_y += InterpOp::Val(dy, scale_y);
      DType dy_grad = InterpOp::Grad(dy, scale_y);
      acc_grad_bottom_y += dy_grad;
      acc_grad_scale_y += InterpOp::GradS(dy, scale_y, dy_grad);
    }
    if (0 == acc_weight_x || 0 == acc_weight_y) {
      continue;
    }
    // Use the x-threads to traverse the channels
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
      DType ograd = top_diff[get_address_BCHW(n, c, py, px,
                                       channels, pooled_height, pooled_width)];
      DType window_grad_bottom_x = 0;
      DType window_grad_bottom_y = 0;
      DType window_grad_scale_x = 0;
      DType window_grad_scale_y = 0;
      // Get the window gradient, which is the gradient w.r.t data * kx/acc_x * ky/acc_y
      for (int kx = begin_x; kx <= end_x; ++kx) {
        DType dx = bottom_x - static_cast<DType>(kx);
        DType kx_val = InterpOp::Val(dx, scale_x);
        DType kx_grad = InterpOp::Grad(dx, scale_x);
        DType kx_scale_grad = InterpOp::GradS(dx, scale_x, kx_grad);
        for (int ky = begin_y; ky <= end_y; ++ky) {
          int ind = get_address_BCHW(batch_ind, c, ky, kx, channels, height, width);
          DType dy = bottom_y - static_cast<DType>(ky);
          DType data_val = bottom_data[ind];
          DType ky_val = InterpOp::Val(dy, scale_y);
          DType ky_grad = InterpOp::Grad(dy, scale_y);
          DType ky_scale_grad = InterpOp::GradS(dy, scale_y, ky_grad);
          window_grad_bottom_x += data_val * ky_val * (kx_grad * acc_weight_x - kx_val * acc_grad_bottom_x);
          window_grad_bottom_y += data_val * kx_val * (ky_grad * acc_weight_y - ky_val * acc_grad_bottom_y);
          window_grad_scale_x += data_val * ky_val * (kx_scale_grad * acc_weight_x - kx_val * acc_grad_scale_x);
          window_grad_scale_y += data_val * kx_val * (ky_scale_grad * acc_weight_y - ky_val * acc_grad_scale_y);
        }
      }
      DType mult_x = ograd / (acc_weight_y * acc_weight_x * acc_weight_x);
      DType mult_y = ograd / (acc_weight_x * acc_weight_y * acc_weight_y);
      grad_bottom_x += mult_x * window_grad_bottom_x;
      grad_bottom_y += mult_y * window_grad_bottom_y;
      grad_scale_x += mult_x * window_grad_scale_x;
      grad_scale_y += mult_y * window_grad_scale_y;
    }
    __shared__ volatile DType __shmem[warp_size/2][warp_size];
    __shmem[threadIdx.y][threadIdx.x] = grad_bottom_x;
    __syncthreads();
    Reduce1D<mshadow::red::sum, warp_bits>(__shmem[threadIdx.y]);
    __syncthreads();
    grad_bottom_x = __shmem[threadIdx.y][0];
    __shmem[threadIdx.y][threadIdx.x] = grad_bottom_y;
    __syncthreads();
    Reduce1D<mshadow::red::sum, warp_bits>(__shmem[threadIdx.y]);
    __syncthreads();
    grad_bottom_y = __shmem[threadIdx.y][0];
    if (anti_aliasing && scale_x > 1) {
      __shmem[threadIdx.y][threadIdx.x] = grad_scale_x;
      __syncthreads();
      Reduce1D<mshadow::red::sum, warp_bits>(__shmem[threadIdx.y]);
      __syncthreads();
      grad_scale_x = __shmem[threadIdx.y][0];
    }
    if (anti_aliasing && scale_y > 1) {
      __shmem[threadIdx.y][threadIdx.x] = grad_scale_y;
      __syncthreads();
      Reduce1D<mshadow::red::sum, warp_bits>(__shmem[threadIdx.y]);
      __syncthreads();
      grad_scale_y = __shmem[threadIdx.y][0];
    }
    DType grad_x1 = ((0.5 - x_relative) * grad_bottom_x - grad_scale_x / pooled_width) * spatial_scale;
    DType grad_y1 = ((0.5 - y_relative) * grad_bottom_y - grad_scale_y / pooled_height) * spatial_scale;
    DType grad_x2 = ((0.5 + x_relative) * grad_bottom_x + grad_scale_x / pooled_width)  * spatial_scale;
    DType grad_y2 = ((0.5 + y_relative) * grad_bottom_y + grad_scale_y / pooled_height) * spatial_scale;
    if (threadIdx.x == 0) {
      if (explicit_batch) {
        atomicAdd(&bottom_roi_diff[5 * n + 1], grad_x1);
        atomicAdd(&bottom_roi_diff[5 * n + 2], grad_y1);
        atomicAdd(&bottom_roi_diff[5 * n + 3], grad_x2);
        atomicAdd(&bottom_roi_diff[5 * n + 4], grad_y2);
      } else {
        atomicAdd(&bottom_roi_diff[4 * n], grad_x1);
        atomicAdd(&bottom_roi_diff[4 * n + 1], grad_y1);
        atomicAdd(&bottom_roi_diff[4 * n + 2], grad_x2);
        atomicAdd(&bottom_roi_diff[4 * n + 3], grad_y2);
      }
    }
  }
}

template<typename InterpOp, bool anti_aliasing>
inline void ROIWrapBackwardAccData(const Tensor<gpu, 4, float> &in_data_grad,
                                   const Tensor<gpu, 4, float> &out_grad,
                                   const Tensor<gpu, 2, float> &bbox,
                                   float spatial_scale,
                                   bool explicit_batch,
                                   int interp_kernel_size) {
  const float *top_diff = out_grad.dptr_;
  const float *bottom_rois = bbox.dptr_;
  float *bottom_data_diff = in_data_grad.dptr_;
  const int count = out_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int batch_num = in_data_grad.size(0);
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
  CheckLaunchParam(dimGrid, dimBlock, "ROIWrapping Backward Data");
  cudaStream_t stream = Stream<gpu>::GetStream(in_data_grad.stream_);
  ROIWrapBackwardAccDataKernel<InterpOp, anti_aliasing, float> <<<dimGrid, dimBlock, 0, stream >> >(
    count, bottom_data_diff, bottom_rois, top_diff, batch_num, channels, height, width,
    pooled_height, pooled_width, spatial_scale, explicit_batch, interp_kernel_size);
  cudaError err = cudaPeekAtLastError();
  CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

template<typename InterpOp, bool anti_aliasing>
inline void ROIWrapBackwardAccROI(const Tensor<gpu, 2, float> &in_roi_grad,
                                  const Tensor<gpu, 4, float> &out_grad,
                                  const Tensor<gpu, 4, float> &data,
                                  const Tensor<gpu, 2, float> &bbox,
                                  float spatial_scale,
                                  bool explicit_batch,
                                  int interp_kernel_size) {
  const float *top_diff = out_grad.dptr_;
  const float *bottom_rois = bbox.dptr_;
  const float *bottom_data = data.dptr_;
  float *bottom_rois_diff = in_roi_grad.dptr_;
  const int count = out_grad.size(0) * out_grad.size(2) * out_grad.size(3);
  const int num_rois = bbox.size(0);
  const int batch_num = data.size(0);
  const int channels = out_grad.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  // For kernel launching, we use the y_threads to traverse all the n*h*w grids and use x_threads to traverse the channels.
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / (kMemUnit / 2);
  const int grid_dim_x = (gridSize > kMaxGridNum) ? kMaxGridNum : gridSize;
  const int grid_dim_y = (gridSize > kMaxGridNum) ? (gridSize + kMaxGridNum - 1) / kMaxGridNum : 1;
  dim3 dimGrid(grid_dim_x, grid_dim_y);
  dim3 dimBlock(kMemUnit, kMemUnit / 2);
  CheckLaunchParam(dimGrid, dimBlock, "ROIWrapping Data Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_roi_grad.stream_);
  ROIWrapBackwardAccROIKernel<kMemUnitBits, InterpOp, anti_aliasing>
    <<<dimGrid, dimBlock, 0, stream>>>(
    count, bottom_rois_diff, bottom_data, bottom_rois, top_diff,
    batch_num, channels, height, width,
    pooled_height, pooled_width, spatial_scale, explicit_batch, interp_kernel_size);
  cudaError err = cudaPeekAtLastError();
  CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

}  // namespace cuda

template<typename InterpOp, bool anti_aliasing, typename DType>
void ROIWrapForward(const Tensor<gpu, 4, DType> &out,
                           const Tensor<gpu, 4, DType> &data,
                           const Tensor<gpu, 2, DType> &bbox,
                           float spatial_scale,
                           bool explicit_batch,
                           int interp_kernel_size) {
  cuda::ROIWrapForward<InterpOp, anti_aliasing>(out, data, bbox, spatial_scale,
    explicit_batch, interp_kernel_size);
}

template<typename InterpOp, bool anti_aliasing>
void ROIWrapBackwardAccData(const Tensor<gpu, 4, float> &in_data_grad,
                                   const Tensor<gpu, 4, float> &out_grad,
                                   const Tensor<gpu, 2, float> &bbox,
                                   float spatial_scale,
                                   bool explicit_batch,
                                   int interp_kernel_size) {
  cuda::ROIWrapBackwardAccData<InterpOp, anti_aliasing>(in_data_grad, out_grad, bbox,
    spatial_scale, explicit_batch, interp_kernel_size);
}

template<typename InterpOp, bool anti_aliasing>
void ROIWrapBackwardAccROI(const Tensor<gpu, 2, float> &in_roi_grad,
                                  const Tensor<gpu, 4, float> &out_grad,
                                  const Tensor<gpu, 4, float> &data,
                                  const Tensor<gpu, 2, float> &bbox,
                                  float spatial_scale,
                                  bool explicit_batch,
                                  int interp_kernel_size) {
  cuda::ROIWrapBackwardAccROI<InterpOp, anti_aliasing>
    (in_roi_grad, out_grad, data, bbox, spatial_scale, explicit_batch, interp_kernel_size);
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

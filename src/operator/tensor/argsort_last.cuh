/*!
*  Copyright (c) 2017 by Contributors
* \file argsort_last.cuh
* \brief Function defintion of nn related operators
*/
#ifndef MXNET_OPERATOR_ARGSORT_LAST_CUH_
#define MXNET_OPERATOR_ARGSORT_LAST_CUH_
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <vector>
#include "../mxnet_op.h"
#include <cub/device/device_segmented_radix_sort.cuh>

namespace mxnet {
namespace op {

template<typename DType>
__global__ void fill_in_initial_indices(DType* out, int col_size, int total_ele_num) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < total_ele_num) {
    int row_id = tid % col_size;
    out[tid] = roundf(DType(row_id));
  }
}

template<typename DType>
void ArgSortLastImpl(const mshadow::Tensor<gpu, 1, DType> &data,
                     const mshadow::Tensor<gpu, 1, DType> &out,
                     const mshadow::Tensor<gpu, 1, int> &d_offsets,
                     bool is_ascend,
                     int batch_num,
                     const Resource &resource) {
  using namespace mshadow::expr;
  using namespace mshadow;
  using namespace mshadow::cuda;
  cudaStream_t stream = Stream<gpu>::GetStream(data.stream_);
  int num_items = data.shape_.Size();
  int col_size = num_items / batch_num;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairs<DType, DType>(NULL, temp_storage_bytes,
    NULL, NULL, NULL, NULL,
    num_items, batch_num, d_offsets.dptr_, d_offsets.dptr_ + 1, 0, sizeof(DType) * 8, stream);
  size_t keys_bytes = num_items * sizeof(DType);
  size_t values_bytes = num_items * sizeof(DType);
  mshadow::Tensor<gpu, 1, void*> workspace = resource.get_space_typed<gpu, 1, void*>(mshadow::Shape1(keys_bytes + values_bytes + temp_storage_bytes), data.stream_);
  mshadow::Tensor<gpu, 1, DType> temp_keys_out = mshadow::Tensor<gpu, 1, DType>(reinterpret_cast<DType*>(workspace.dptr_), Shape1(num_items), data.stream_);
  mshadow::Tensor<gpu, 1, DType> temp_values_in = mshadow::Tensor<gpu, 1, DType>(reinterpret_cast<DType*>(workspace.dptr_ + keys_bytes),
    Shape1(num_items), data.stream_);
  void* temp_store_ptr = workspace.dptr_ + keys_bytes + values_bytes;
  const int grid_dim_x = (data.shape_.Size() + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(grid_dim_x);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "fill_in_initial_indices");
  fill_in_initial_indices << <dimGrid, dimBlock, 0, stream >> > (temp_values_in.dptr_, col_size, num_items);
  cudaError err = cudaPeekAtLastError();
  CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  if (is_ascend) {
    cub::DeviceSegmentedRadixSort::SortPairs<DType, DType>(temp_store_ptr, temp_storage_bytes,
      data.dptr_, temp_keys_out.dptr_, temp_values_in.dptr_, out.dptr_,
      num_items, batch_num, d_offsets.dptr_, d_offsets.dptr_ + 1, 0, sizeof(DType) * 8, stream);
  } else {
    cub::DeviceSegmentedRadixSort::SortPairsDescending<DType, DType>(temp_store_ptr, temp_storage_bytes,
      data.dptr_, temp_keys_out.dptr_, temp_values_in.dptr_, out.dptr_,
      num_items, batch_num, d_offsets.dptr_, d_offsets.dptr_ + 1, 0, sizeof(DType) * 8, stream);
  }
}
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ARGSORT_LAST_CUH_

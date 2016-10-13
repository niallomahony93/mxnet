/*!
 * Copyright (c) 2016 by Contributors
 * \file roi_wrapping.cc
 * \brief roi wrapping operator
 * \author Xingjian Shi
*/
#include "./roi_wrapping-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;


namespace mshadow {
template<typename DType>
void parse_roi_coords(const DType * rois, DType spatial_scale, bool explicit_batch,
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
    x1 = rois[shift + 0] * spatial_scale;
    y1 = rois[shift + 1] * spatial_scale;
    x2 = rois[shift + 2] * spatial_scale;
    y2 = rois[shift + 3] * spatial_scale;
  }
}
template<typename DType>
inline void BilinearPoolForward(const Tensor<cpu, 4, DType> &out,
                                const Tensor<cpu, 4, DType> &data,
                                const Tensor<cpu, 2, DType> &bbox,
                                DType spatial_scale,
                                bool explicit_batch) {
  const DType *bottom_data = data.dptr_;
  const DType *bottom_rois = bbox.dptr_;
  //DType *top_data = out.dptr_;
  const int count = out.shape_.Size();
  const int batch_num = data.size(0);
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int roi_num = bbox.size(0);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  for (index_t n = 0; n < roi_num; ++n) {
    for (index_t c = 0; c < channels; ++c) {
      for (index_t h = 0; h < pooled_height; ++h) {
        for (index_t w = 0; w < pooled_width; ++w) {
          int batch_ind;
          DType roi_x1, roi_y1, roi_x2, roi_y2;
          parse_roi_coords(bottom_rois, spatial_scale, explicit_batch,
                           batch_ind, roi_x1, roi_y1, roi_x2, roi_y2);
          // Force malformed ROIs to be 1x1
          DType roi_width = max(roi_x2 - roi_x1 + 1, 1);
          DType roi_height = max(roi_y2 - roi_y1 + 1, 1);
          DType bottom_x = static_cast<DType>(px)* roi_width / static_cast<DType>(pooled_width) + roi_x1;
          DType bottom_y = static_cast<DType>(py)* roi_height / static_cast<DType>(pooled_height) + roi_y1;
          // Get the topleft coordinates and the corresponding deltas, x - floor(x) and y - floor(y)
          int topleft_x, topleft_y;
          float topleft_dx, topleft_dy;
          topleft_x = floor(bottom_x);
          topleft_y = floor(bottom_y);
          topleft_dx = bottom_x - static_cast<DType>topleft_x;
          topleft_dy = bottom_y - static_cast<DType>topleft_y;
          DType topleft_v = topleft_x >= 0 && topleft_x < width &&
                            topleft_y >=0  && topleft_y < height
                            ? bottom_data[batch_ind][c][topleft_y][topleft_x] : 0;
          DType topright_v = topleft_x + 1 >= 0 && topleft_x + 1 < width &&
                             topleft_y >=0  && topleft_y < height
                             ? bottom_data[batch_ind][c][topleft_y][topleft_x + 1] : 0;
          DType bottomleft_v = topleft_x >= 0 && topleft_x < width &&
                               topleft_y + 1 >=0  && topleft_y + 1 < height
                               ? bottom_data[batch_ind][c][topleft_y + 1][topleft_x] : 0;
          DType bottomright_v = topleft_x + 1 >= 0 && topleft_x + 1 < width &&
                                topleft_y + 1 >=0  && topleft_y + 1 < height
                                ? bottom_data[batch_ind][c][topleft_y + 1][topleft_x + 1] : 0;
          out[n][c][h][w] = (1 - topleft_dx) * (1 - topleft_dy) * topleft_v
                      + (1 - topleft_dx) * topleft_dy * topright_v
                      + topleft_dx * (1 - topleft_dy) * bottomleft_v
                      + topleft_dx * topleft_dy * bottomright_v;
        }
      }
    }
  }
}
inline void BilinearPoolBackwardAccData(const Tensor<cpu, 4, float> &in_data_grad,
                                        const Tensor<cpu, 4, float> &out_grad,
                                        const Tensor<cpu, 2, float> &bbox,
                                        float spatial_scale,
                                        bool explicit_batch) {
  LOG(FATAL) << "Not Implemented!";
}

inline void BilinearPoolBackwardAccROI(const Tensor<cpu, 2, float> &in_roi_grad,
                                       const Tensor<cpu, 4, float> &out_grad,
                                       const Tensor<cpu, 2, float> &bbox,
                                       const Tensor<cpu, 4, float> &data,
                                       float spatial_scale,
                                       bool explicit_batch) {
  LOG(FATAL) << "Not Implemented!";
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(ROIWrappingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIWrappingOp<cpu, DType>(param);
  });
  return op;
}

Operator *ROIWrappingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ROIWrappingParam);

MXNET_REGISTER_OP_PROPERTY(ROIWrapping, ROIWrappingProp)
.describe("Performs region-of-interest pooling on inputs. Resize bounding box coordinates by "
"spatial_scale and crop input feature maps accordingly. The cropped feature maps are pooled "
"by image interpolation to a fixed size output indicated by pooled_size. Shape of the output will "
"be (roi_num, channels, pooled_height, pooled_width).")
.add_argument("data", "Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("rois", "Symbol", "Bounding box coordinates, a 2D array of "
"[[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners "
"of designated region of interest. batch_index indicates the index of corresponding image "
"in the input data")
.add_arguments(ROIWrappingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

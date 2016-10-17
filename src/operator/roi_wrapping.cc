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




namespace mshadow {
template<typename DType>
void parse_roi_coords(const int n, const DType * rois, float spatial_scale, bool explicit_batch,
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
    x1 = rois[shift + 0] * spatial_scale;
    y1 = rois[shift + 1] * spatial_scale;
    x2 = rois[shift + 2] * spatial_scale;
    y2 = rois[shift + 3] * spatial_scale;
  }
}

template<bool anti_aliasing, typename DType>
void get_kernel_begin_end(DType x, int kernel_size, DType scale_x, int width,
                          int &begin_x, int &end_x) {
  if (anti_aliasing && scale_x > 1) {
    begin_x = static_cast<int>(floor(x - DType(kernel_size) * scale_x)) + 1;
    end_x = static_cast<int>(ceil(x + DType(kernel_size) * scale_x)) - 1;
  }
  else {
    begin_x = static_cast<int>(floor(x)) - kernel_size + 1;
    end_x = static_cast<int>(ceil(x)) + kernel_size - 1;
  }
  begin_x = max(begin_x, 0);
  end_x = min(end_x, width - 1);
}

int get_address_BCHW(int b, int c, int h, int w,
  int channel_num, int height, int width) {
  return ((b * channel_num + c) * height + h) * width + w;
}

template<typename InterpOp, bool anti_aliasing, typename DType>
void ROIWrapForward(const Tensor<cpu, 4, DType> &out,
                           const Tensor<cpu, 4, DType> &data,
                           const Tensor<cpu, 2, DType> &bbox,
                           float spatial_scale,
                           bool explicit_batch,
                           int interp_kernel_size) {
  const DType *bottom_rois = bbox.dptr_;
  const int batch_num = data.size(0);
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int roi_num = out.size(0);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  for (int n = 0; n < roi_num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int py = 0; py < pooled_height; ++py) {
        for (int px = 0; px < pooled_width; ++px) {
          int batch_ind;
          DType roi_x1, roi_y1, roi_x2, roi_y2;
          parse_roi_coords(n, bottom_rois, spatial_scale, explicit_batch,
                           batch_ind, roi_x1, roi_y1, roi_x2, roi_y2);
          // output zero for negative batch inds
          if (batch_ind < 0) {
            out[n][c][py][px] = 0;
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
          // Get the topleft coordinates and the corresponding deltas, x - floor(x) and y - floor(y)
          int begin_x, begin_y, end_x, end_y;
          DType acc_weight = 0;
          get_kernel_begin_end<anti_aliasing>(bottom_x, interp_kernel_size, scale_x, width, begin_x, end_x);
          get_kernel_begin_end<anti_aliasing>(bottom_y, interp_kernel_size, scale_y, height, begin_y, end_y);

          for (int kx = begin_x; kx <= end_x; kx++) {
            DType dx = bottom_x - static_cast<DType>(kx);
            for (int ky = begin_y; ky <= end_y; ky++) {
              DType dy = bottom_y - static_cast<DType>(ky);
              DType weight = InterpOp::Val(dx, scale_x) * InterpOp::Val(dy, scale_y);
              acc_weight += weight;
              out[n][c][py][px] += weight * data[batch_ind][c][ky][kx];
            }
          }
          if (acc_weight != 0) {
            out[n][c][py][px] /= acc_weight;
          } else {
            out[n][c][py][px] = 0;
          }
        }
      }
    }
  }
}
template<typename InterpOp, bool anti_aliasing, typename DType>
void ROIWrapBackwardAccData(const Tensor<cpu, 4, DType> &in_data_grad,
                                   const Tensor<cpu, 4, DType> &out_grad,
                                   const Tensor<cpu, 2, DType> &bbox,
                                   float spatial_scale,
                                   bool explicit_batch,
                                   int interp_kernel_size) {
  const DType *bottom_rois = bbox.dptr_;
  const int batch_num = in_data_grad.size(0);
  const int channels = in_data_grad.size(1);
  const int height = in_data_grad.size(2);
  const int width = in_data_grad.size(3);
  const int roi_num = out_grad.size(0);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  for (int n = 0; n < roi_num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int py = 0; py < pooled_height; ++py) {
        for (int px = 0; px < pooled_width; ++px) {
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
          if (acc_weight_x == 0 || acc_weight_y == 0) {
            continue;
          }
          acc_weight = acc_weight_x * acc_weight_y;
          // use atomicadd to backpropage gradient
          for (int kx = begin_x; kx <= end_x; ++kx) {
            DType dx = bottom_x - static_cast<DType>(kx);
            for (int ky = begin_y; ky <= end_y; ++ky) {
              DType dy = bottom_y - static_cast<DType>(ky);
              in_data_grad[batch_ind][c][ky][kx] += InterpOp::Val(dx, scale_x) * InterpOp::Val(dy, scale_y)
                / acc_weight * out_grad[n][c][py][px];
            }
          }
        }
      }
    }
  }
}
template<typename InterpOp, bool anti_aliasing, typename DType>
void ROIWrapBackwardAccROI(const Tensor<cpu, 2, DType> &in_roi_grad,
                           const Tensor<cpu, 4, DType> &out_grad,
                           const Tensor<cpu, 4, DType> &data,
                           const Tensor<cpu, 2, DType> &bbox,
                           float spatial_scale,
                           bool explicit_batch,
                           int interp_kernel_size) {
  const DType *top_diff = out_grad.dptr_;
  const DType *bottom_rois = bbox.dptr_;
  const DType *bottom_data = data.dptr_;
  float *bottom_roi_diff = in_roi_grad.dptr_;
  const int count = out_grad.size(0) * out_grad.size(2) * out_grad.size(3);
  const int num_rois = bbox.size(0);
  const int batch_num = data.size(0);
  const int channels = out_grad.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  for (int n = 0; n < num_rois; n++) {
    for (int py = 0; py < pooled_height; py++) {
      for (int px = 0; px < pooled_width; px++) {
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
        if (acc_weight_x == 0 || acc_weight_y == 0) {
          continue;
        }
        // Use the x-threads to traverse the channels
        for (int c = 0; c < channels; c++) {
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
        DType grad_x1 = ((0.5 - x_relative) * grad_bottom_x - grad_scale_x / pooled_width) * spatial_scale;
        DType grad_y1 = ((0.5 - y_relative) * grad_bottom_y - grad_scale_y / pooled_height) * spatial_scale;
        DType grad_x2 = ((0.5 + x_relative) * grad_bottom_x + grad_scale_x / pooled_width)  * spatial_scale;
        DType grad_y2 = ((0.5 + y_relative) * grad_bottom_y + grad_scale_y / pooled_height) * spatial_scale;
        if (explicit_batch) {
          bottom_roi_diff[5 * n + 1]+= grad_x1;
          bottom_roi_diff[5 * n + 2]+= grad_y1;
          bottom_roi_diff[5 * n + 3]+= grad_x2;
          bottom_roi_diff[5 * n + 4]+= grad_y2;
        } else {
          bottom_roi_diff[4 * n] += grad_x1;
          bottom_roi_diff[4 * n + 1]+= grad_y1;
          bottom_roi_diff[4 * n + 2]+= grad_x2;
          bottom_roi_diff[4 * n + 3]+= grad_y2;
        }
      }
    }
  }
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

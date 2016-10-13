/*!
 * Copyright (c) 2016 by Contributors
 * \file roi_wrapping-inl.h
 * \brief roi pooling operator and symbol
 * \author Xingjian Shi
*/
#ifndef MXNET_OPERATOR_ROI_WRAPPING_INL_H_
#define MXNET_OPERATOR_ROI_WRAPPING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator_util.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace roiwrap_enum {
enum ROIWrappingOpInputs {kData, kBox};
enum ROIWrappingOpOutputs {kOut};
enum ROIWrappingInterpType {kBilinear};
}  // roiwrap_enum

struct ROIWrappingParam : public dmlc::Parameter<ROIWrappingParam> {
  TShape pooled_size;
  float spatial_scale;
  bool explicit_batch;
  int interp_type;
  DMLC_DECLARE_PARAMETER(ROIWrappingParam) {
    DMLC_DECLARE_FIELD(pooled_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("fix pooled size: (h, w)");
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0).set_default(1.0)
    .describe("Ratio of input feature map height (or w) to raw image height (or w). "
    "Equals the reciprocal of total stride in convolutional layers");
    DMLC_DECLARE_FIELD(interp_type)
    .add_enum("bilinear", roiwrap_enum::kBilinear)
    .describe("The interpolation kernel to use. \"bilinear\" means the bilinear kernel");
    DMLC_DECLARE_FIELD(explicit_batch).set_default(false)
    .describe("If set to True, the rois should be of shape (n, 5) and each roi should be "
              "[batch_ind, x1, y1, x2, y2]. If set to False, the rois should be of shape (n, 4) "
              "and each roi should be [x1, y1, x2, y2]");
  }
};

template<typename xpu, typename DType>
class ROIWrappingOp : public Operator {
public:
  explicit ROIWrappingOp(ROIWrappingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
    const std::vector<TBlob> &in_data,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &out_data,
    const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(out_data[roiwrap_enum::kOut].shape_[0], in_data[roiwrap_enum::kBox].shape_[0]);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[roiwrap_enum::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[roiwrap_enum::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[roiwrap_enum::kOut].get<xpu, 4, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    if (req[roiwrap_enum::kOut] != kNullOp) {
      if (param_.interp_type == roiwrap_enum::kBilinear) {
        BilinearPoolForward(out, data, bbox, param_.spatial_scale, param_.explicit_batch);
      } else {
        LOG(FATAL) << "ROIWrapping: interp_type=" << param_.interp_type << " is not supported!";
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(out_grad[roiwrap_enum::kOut].shape_[0], in_data[roiwrap_enum::kBox].shape_[0]);
    CHECK_NE(req[roiwrap_enum::kData], kWriteInplace) <<
      "ROIWrapping: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[roiwrap_enum::kBox], kWriteInplace) <<
      "ROIWrapping: Backward doesn't support kWriteInplace.";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, real_t> grad_out = out_grad[roiwrap_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2, real_t> bbox = in_data[roiwrap_enum::kBox].get<xpu, 2, real_t>(s);
    Tensor<xpu, 4, real_t> grad_in = in_grad[roiwrap_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2, real_t> grad_roi = in_grad[roiwrap_enum::kBox].get<xpu, 2, real_t>(s);
    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(grad_in.CheckContiguous(), true);
    CHECK_EQ(grad_roi.CheckContiguous(), true);
    if (kAddTo == req[roiwrap_enum::kData] || kWriteTo == req[roiwrap_enum::kData]) {
      if (kWriteTo == req[roiwrap_enum::kData]) {
        grad_in = 0.0f;
      }
      if (param_.interp_type == roiwrap_enum::kBilinear) {
        BilinearPoolBackwardAccData(grad_in, grad_out, bbox, param_.spatial_scale, param_.explicit_batch);
      } else {
        LOG(FATAL) << "ROIWrapping: interp_type=" << param_.interp_type << " is not supported!";
      }
    }
    if (kAddTo == req[roiwrap_enum::kBox] || kWriteTo == req[roiwrap_enum::kBox]) {
      Tensor<xpu, 4, real_t> data = in_data[roiwrap_enum::kData].get<xpu, 4, real_t>(s);
      if (kWriteTo == req[roiwrap_enum::kBox]) {
        grad_roi = 0.0f;
      }
      if (param_.interp_type == roiwrap_enum::kBilinear) {
        BilinearPoolBackwardAccROI(grad_roi, grad_out, bbox, data, param_.spatial_scale, param_.explicit_batch);
      } else {
        LOG(FATAL) << "ROIWrapping: interp_type=" << param_.interp_type << " is not supported!";
      }
    }
  }

private:
  ROIWrappingParam param_;
};  // class ROIPoolingOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ROIWrappingParam param, int dtype);

#if DMLC_USE_CXX11
class ROIWrappingProp : public OperatorProperty {
public:
  std::vector<std::string> ListArguments() const override {
    return{ "data", "rois" };
  }

  std::vector<std::string> ListOutputs() const override {
    return{ "output" };
  }

  int NumOutputs() const override {
    return 1;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, rois]";

    // data: [batch_size, c, h, w]
    TShape dshape = in_shape->at(roiwrap_enum::kData);
    CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";

    // bbox: [num_rois, 5]
    TShape bshape = in_shape->at(roiwrap_enum::kBox);
    CHECK_EQ(bshape.ndim(), 2) << "bbox should be a 2D tensor of shape [batch, 5]";
    if (param_.explicit_batch) {
      CHECK_EQ(bshape[1], 5) << "when explicit_batch is on, bbox should be a 2D tensor of shape [batch, 5]";
    } else {
      CHECK_EQ(bshape[1], 4) << "when explicit_batch is off, bbox should be a 2D tensor of shape [batch, 4]";
    }
    // out: [num_rois, c, pooled_h, pooled_w]
    out_shape->clear();
    out_shape->push_back(
      Shape4(bshape[0], dshape[1], param_.pooled_size[0], param_.pooled_size[1]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2);
    int dtype = (*in_type)[0];
    CHECK_EQ(dtype, (*in_type)[1]);
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    ROIWrappingProp* roi_wrapping_sym = new ROIWrappingProp();
    roi_wrapping_sym->param_ = this->param_;
    return roi_wrapping_sym;
  }

  std::string TypeString() const override {
    return "ROIWrapping";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return{ out_grad[roiwrap_enum::kOut], in_data[roiwrap_enum::kBox], in_data[roiwrap_enum::kData] };
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const override;

private:
  ROIWrappingParam param_;
};  // class ROIWrappingProp
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ROI_WRAPPING_INL_H_

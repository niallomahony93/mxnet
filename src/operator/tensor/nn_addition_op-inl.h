/*!
*  Copyright (c) 2017 by Contributors
* \file nn_addition_op-inl.h
* \brief Function defintion of nn related operators
*/
#ifndef MXNET_OPERATOR_TENSOR_NN_ADDITION_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_NN_ADDITION_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

enum PadType {kSame, kValid};

struct LocalCorrelationParam : public dmlc::Parameter<LocalCorrelationParam> {
  TShape kernel;
  TShape dilate;
  TShape stride;
  int pad_type;
  uint64_t workspace;
  DMLC_DECLARE_PARAMETER(LocalCorrelationParam) {
    int shape[] = {1, 1};
    DMLC_DECLARE_FIELD(kernel)
    .describe("Size of the local search region.");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape(shape, shape + 2))
    .describe("Size of the dilation of the search region in lhs.");
    DMLC_DECLARE_FIELD(stride).set_default(TShape(shape, shape + 2))
    .describe("Size of the stride for rhs.");
    shape[0] = shape[1] = 0;
    DMLC_DECLARE_FIELD(pad_type).set_default(PadType::kValid)
    .add_enum("valid", PadType::kValid)
    .add_enum("same", PadType::kSame)
    .describe("Whether we need to add padding to the lhs."
      " If pad_type is \"valid\", the \"valid\" correlation will be performed."
      " Otherwise zero-padding will be used.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum tmp workspace allowed for convolution (MB).");
  }
};

struct LocalFilterParam : public dmlc::Parameter<LocalFilterParam> {
  TShape kernel;
  TShape dilate;
  TShape stride;
  int pad_type;
  uint64_t workspace;
  DMLC_DECLARE_PARAMETER(LocalFilterParam) {
    int shape[] = {1, 1};
    DMLC_DECLARE_FIELD(kernel)
    .describe("Size of the local filter kernel.");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape(shape, shape + 2))
    .describe("Dilate for the local filters.");
    DMLC_DECLARE_FIELD(stride).set_default(TShape(shape, shape + 2))
    .describe("Stride for the local filters.");
    DMLC_DECLARE_FIELD(pad_type).set_default(PadType::kValid)
    .add_enum("valid", PadType::kValid)
    .add_enum("same", PadType::kSame)
    .describe("If pad_type is valid, the \"valid\" convolution will be used. "
              "Otherwise the data will be padded to make sure that the output"
              " will be have the same height/width as the input.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum tmp workspace allowed for convolution (MB).");
  }
};

template<typename xpu>
void LocalCorrelationForward_(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const LocalCorrelationParam& param_ = nnvm::get<LocalCorrelationParam>(attrs.parsed);
  CHECK_EQ(param_.kernel.ndim(), 2);
  CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32)
    << "LocalCorrelation only support 32 bit float so far";
  int batch_size = inputs[0].shape_[0];
  int channel_num = inputs[0].shape_[1];
  int h1 = inputs[0].shape_[2];
  int w1 = inputs[0].shape_[3];
  int h2 = inputs[1].shape_[2];
  int w2 = inputs[1].shape_[3];
  int k_y = param_.kernel[0];
  int k_x = param_.kernel[1];
  CHECK(k_y % 2 == 1 && k_x % 2 == 1) << "Only support odd kernel size";
  int pad_type = param_.pad_type;
  CHECK(pad_type == PadType::kSame || pad_type == PadType::kValid);
  CHECK_EQ(batch_size, inputs[1].shape_[0]);
  CHECK_EQ(channel_num, inputs[1].shape_[1]);
  CHECK_EQ(batch_size, outputs[0].shape_[0]);
  CHECK_EQ(h1, outputs[0].shape_[1]);
  CHECK_EQ(w1, outputs[0].shape_[2]);
  CHECK_EQ(k_y, outputs[0].shape_[3]);
  CHECK_EQ(k_x, outputs[0].shape_[4]);
  mshadow::Tensor<xpu, 4, real_t> lhs = inputs[0].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 4, real_t> rhs = inputs[1].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> out = outputs[0].get_with_shape<xpu, 3, real_t>(
                                           Shape3(batch_size * h1 * w1, 1, k_y * k_x), s);
  int ele_tmp_lhs_bytes = sizeof(real_t) * (channel_num * h1 * w1);
  int ele_tmp_rhs_bytes = sizeof(real_t) * (channel_num * h1 * w1 * k_y * k_x);
  int ele_batch_dot_workspace_bytes = sizeof(real_t*) * 3 * h1 * w1;
  int workspace_ele_size = ele_tmp_lhs_bytes + ele_tmp_rhs_bytes + ele_batch_dot_workspace_bytes;
  int batch_step_ = std::min(static_cast<int>((param_.workspace << 20) / workspace_ele_size), batch_size);
  CHECK_GE(batch_step_, 1);
  mshadow::Tensor<xpu, 1, void*> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, void*>(mshadow::Shape1(batch_step_ * workspace_ele_size), s);
  if (kNullOp != req[0]) {
    for (int i = 0; i < batch_size; i += batch_step_) {
      const index_t step = std::min(batch_step_, batch_size - i);
      mshadow::Tensor<xpu, 3, real_t> tmp_lhs = mshadow::Tensor<xpu, 3, real_t>(
                                              reinterpret_cast<real_t*>(workspace.dptr_),
                                              Shape3(step * h1 * w1, 1, channel_num), s);
      mshadow::Tensor<xpu, 3, real_t> tmp_rhs = mshadow::Tensor<xpu, 3, real_t>(
                                                    reinterpret_cast<real_t*>(
                                                      workspace.dptr_ + step * ele_tmp_lhs_bytes),
                                                    Shape3(step * h1 * w1, channel_num, k_y * k_x), s);
      mshadow::Tensor<xpu, 1, real_t*> batch_dot_workspace = mshadow::Tensor<xpu, 1, real_t*>(
                                                    reinterpret_cast<real_t**>(
                                                      workspace.dptr_ + step * (ele_tmp_lhs_bytes + ele_tmp_rhs_bytes)),
                                                    Shape1(step * 3 * h1 * w1), s);
      tmp_lhs = reshape(transpose(lhs.Slice(i, i + step),
                                  Shape4(0, 2, 3, 1)),
                        Shape3(step * h1 * w1, 1, channel_num));
      if (pad_type == PadType::kValid) {
        tmp_rhs = reshape(swapaxis<1, 0>(unpack_patch2col(rhs.Slice(i, i + step),
                                                            param_.kernel[0],
                                                            param_.kernel[1],
                                                            param_.stride[0],
                                                            param_.stride[1],
                                                            param_.dilate[0],
                                                            param_.dilate[1])),
                           Shape3(step * h1 * w1, channel_num, k_y * k_x));
      } else {
        int pad_y = param_.dilate[0] * (param_.kernel[0] - 1) / 2;
        int pad_x = param_.dilate[1] * (param_.kernel[1] - 1) / 2;
        tmp_rhs = reshape(swapaxis<1, 0>(unpack_patch2col(pad(rhs.Slice(i, i + step), pad_y, pad_x),
                                                            param_.kernel[0],
                                                            param_.kernel[1],
                                                            param_.stride[0],
                                                            param_.stride[1],
                                                            param_.dilate[0],
                                                            param_.dilate[1])),
                           Shape3(step * h1 * w1, channel_num, k_y * k_x));
      }
      mshadow::BatchGEMM<false, false>(out.Slice(i * h1 * w1, (i + step) * h1 * w1), tmp_lhs, tmp_rhs, 1.0f,
                                       (kAddTo == req[0]) ? 1.0f : 0.0f,
                                       batch_dot_workspace);
    }
  }
}

template<typename xpu>
void LocalCorrelationBackward_(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const LocalCorrelationParam& param_ = nnvm::get<LocalCorrelationParam>(attrs.parsed);
  CHECK_NE(req[0], kWriteInplace);
  CHECK_NE(req[1], kWriteInplace);
  int batch_size = inputs[1].shape_[0];
  int channel_num = inputs[1].shape_[1];
  int h1 = inputs[1].shape_[2];
  int w1 = inputs[1].shape_[3];
  int h2 = inputs[2].shape_[2];
  int w2 = inputs[2].shape_[3];
  int k_y = param_.kernel[0];
  int k_x = param_.kernel[1];
  mshadow::Tensor<xpu, 3, real_t> out_grad = inputs[0].get_with_shape<xpu, 3, real_t>(
                                               Shape3(batch_size * h1 * w1, 1, k_y * k_x), s);
  mshadow::Tensor<xpu, 4, real_t> lhs = inputs[1].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 4, real_t> rhs = inputs[2].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 4, real_t> lhs_grad = outputs[0].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 4, real_t> rhs_grad = outputs[1].get<xpu, 4, real_t>(s);
  int ele_tmp_lhs_bytes = sizeof(real_t) * (channel_num * h1 * w1);
  int ele_tmp_rhs_bytes = sizeof(real_t) * (channel_num * h1 * w1 * k_y * k_x);
  int ele_batch_dot_workspace_bytes = sizeof(real_t*) * 3 * h1 * w1;
  int workspace_ele_size = ele_tmp_lhs_bytes + ele_tmp_rhs_bytes + ele_batch_dot_workspace_bytes;
  int batch_step_ = std::min(static_cast<int>((param_.workspace << 20) / workspace_ele_size), batch_size);
  CHECK_GE(batch_step_, 1);
  mshadow::Tensor<xpu, 1, void*> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, void*>(mshadow::Shape1(batch_step_ * workspace_ele_size), s);
  for (int i = 0; i < batch_size; i += batch_step_) {
    const index_t step = std::min(batch_step_, batch_size - i);
    mshadow::Tensor<xpu, 3, real_t> tmp_lhs = mshadow::Tensor<xpu, 3, real_t>(
                                            reinterpret_cast<real_t*>(workspace.dptr_),
                                            Shape3(step * h1 * w1, 1, channel_num), s);
    mshadow::Tensor<xpu, 3, real_t> tmp_rhs = mshadow::Tensor<xpu, 3, real_t>(
                                                  reinterpret_cast<real_t*>(
                                                    workspace.dptr_ + step * ele_tmp_lhs_bytes),
                                                  Shape3(step * h1 * w1, channel_num, k_y * k_x), s);
    mshadow::Tensor<xpu, 1, real_t*> batch_dot_workspace = mshadow::Tensor<xpu, 1, real_t*>(
                                                  reinterpret_cast<real_t**>(
                                                    workspace.dptr_ + step * (ele_tmp_lhs_bytes + ele_tmp_rhs_bytes)),
                                                  Shape1(step * 3 * h1 * w1), s);
    if (param_.pad_type == PadType::kValid) {
      tmp_rhs = reshape(swapaxis<1, 0>(unpack_patch2col(rhs.Slice(i, i + step),
                                                          param_.kernel[0],
                                                          param_.kernel[1],
                                                          param_.stride[0],
                                                          param_.stride[1],
                                                          param_.dilate[0],
                                                          param_.dilate[1])),
                          Shape3(step * h1 * w1, channel_num, k_y * k_x));
    } else {
      int pad_y = param_.dilate[0] * (param_.kernel[0] - 1) / 2;
      int pad_x = param_.dilate[1] * (param_.kernel[1] - 1) / 2;
      tmp_rhs = reshape(swapaxis<1, 0>(unpack_patch2col(pad(rhs.Slice(i, i + step), pad_y, pad_x),
                                                          param_.kernel[0],
                                                          param_.kernel[1],
                                                          param_.stride[0],
                                                          param_.stride[1],
                                                          param_.dilate[0],
                                                          param_.dilate[1])),
                          Shape3(step * h1 * w1, channel_num, k_y * k_x));
    }
    mshadow::BatchGEMM<false, true>(tmp_lhs, out_grad.Slice(i * h1 * w1, (i + step) * h1 * w1), tmp_rhs, 1.0f,
                                    0.0f, batch_dot_workspace);
    Assign(lhs_grad.Slice(i, i + step), req[0], transpose(reshape(tmp_lhs,
                                                                  Shape4(step, h1, w1, channel_num)),
                                                          Shape4(0, 3, 1, 2)));
    tmp_lhs = reshape(transpose(lhs.Slice(i, i + step), Shape4(0, 2, 3, 1)),
                      Shape3(step * h1 * w1, 1, channel_num));
    mshadow::BatchGEMM<true, false>(tmp_rhs, tmp_lhs, out_grad.Slice(i * h1 * w1, (i + step) * h1 * w1), 1.0f,
                                    0.0f, batch_dot_workspace);
    if (param_.pad_type == PadType::kValid) {
      Assign(rhs_grad.Slice(i, i + step), req[1],
              pack_col2patch(swapaxis<1, 0>(reshape(tmp_rhs,
                                                    Shape2(step * h1 * w1, channel_num * k_y * k_x))),
                            rhs.Slice(i, i + step).shape_,
                            param_.kernel[0],
                            param_.kernel[1],
                            param_.stride[0],
                            param_.stride[1],
                            param_.dilate[0],
                            param_.dilate[1]));
    } else {
      Shape<4> pshape = rhs.Slice(i, i + step).shape_;
      pshape[2] += param_.dilate[0] * (param_.kernel[0] - 1);
      pshape[3] += param_.dilate[1] * (param_.kernel[1] - 1);
      Assign(rhs_grad.Slice(i, i + step), req[1],
              crop(pack_col2patch(swapaxis<1, 0>(reshape(tmp_rhs,
                                                    Shape2(step * h1 * w1, channel_num * k_y * k_x))),
                                  pshape,
                                  param_.kernel[0],
                                  param_.kernel[1],
                                  param_.stride[0],
                                  param_.stride[1],
                                  param_.dilate[0],
                                  param_.dilate[1]),
                  Shape2(h2, w2)));
    }
  }
}

inline bool LocalCorrelationShape(const nnvm::NodeAttrs& attrs,
                                  std::vector<TShape> *in_attrs,
                                  std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  const LocalCorrelationParam& param_ = nnvm::get<LocalCorrelationParam>(attrs.parsed);
  CHECK_EQ(param_.kernel.ndim(), 2) << "kernel must be (ky, kx)";
  CHECK(param_.kernel[0] % 2 == 1 && param_.kernel[1] % 2 == 1)
    << "kernel size must be odd, kernel=" << param_.kernel;
  CHECK(param_.stride[0] == 1 && param_.stride[1] == 1) << "stride is not supported and must be 1.";
  TShape& lshape = (*in_attrs)[0];
  TShape& rshape = (*in_attrs)[1];
  CHECK_EQ(lshape.ndim(), 4);
  CHECK_EQ(rshape.ndim(), 4);
  int batch_size = lshape[0];
  int channel_num = lshape[1];
  int h1 = lshape[2];
  int w1 = lshape[3];
  int h2 = rshape[2];
  int w2 = rshape[3];
  // We will first try to complete the shape information
  if (batch_size == 0) {
    batch_size = rshape[0];
    CHECK_NE(batch_size, 0) << "shape is invalid! lshape=" << lshape << ", rshape=" << rshape;
  }
  if (channel_num == 0) {
    channel_num = rshape[1];
    CHECK_NE(channel_num, 0) << "shape is invalid! lshape=" << lshape << ", rshape=" << rshape;
  }
  if (h1 == 0 && h2 != 0) {
    h1 = (param_.pad_type == PadType::kValid) ? (h2 - param_.dilate[0] * (param_.kernel[0] - 1)) : h2;
  } else if (h1 != 0 && h2 == 0) {
    h2 = (param_.pad_type == PadType::kValid) ? h1 + param_.dilate[0] * (param_.kernel[0] - 1) : h1;
  }
  if (w1 == 0 && w2 != 0) {
    w1 = (param_.pad_type == PadType::kValid) ? w2 - param_.dilate[0] * (param_.kernel[0] - 1) : w2;
  } else if (w1 != 0 && w2 == 0) {
    w2 = (param_.pad_type == PadType::kValid) ? w1 + param_.dilate[1] * (param_.kernel[1] - 1) : w1;
  }
  mshadow::Shape<5> out_shape;
  out_shape[0] = batch_size;
  out_shape[1] = h1;
  out_shape[2] = w1;
  out_shape[3] = param_.kernel[0];
  out_shape[4] = param_.kernel[1];
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  batch_size = (*out_attrs)[0][0];
  h1 = (*out_attrs)[0][1];
  w1 = (*out_attrs)[0][2];
  h2 = (param_.pad_type == PadType::kValid) ? h1 + param_.dilate[0] * (param_.kernel[0] - 1) : h1;
  w2 = (param_.pad_type == PadType::kValid) ? w1 + param_.dilate[1] * (param_.kernel[1] - 1) : w1;
  CHECK(h1 != 0 && w1 != 0) << "Incomplete shape inference failed! lshape="
                            << lshape << ", rshape=" << rshape;
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::Shape4(batch_size, channel_num, h1, w1));
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::Shape4(batch_size, channel_num, h2, w2));
  return true;
}

template<typename xpu>
void LocalFilterForward_(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const LocalFilterParam& param_ = nnvm::get<LocalFilterParam>(attrs.parsed);
  CHECK_EQ(param_.kernel.ndim(), 2);
  CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32)
    << "LocalFilter only support 32 bit float so far";
  int batch_size = inputs[0].shape_[0];
  int channel_num = inputs[0].shape_[1];
  int h2 = inputs[0].shape_[2];
  int w2 = inputs[0].shape_[3];
  int h1 = inputs[1].shape_[1];
  int w1 = inputs[1].shape_[2];
  int k_y = param_.kernel[0];
  int k_x = param_.kernel[1];
  CHECK(k_y % 2 == 1 && k_x % 2 == 1) << "Only support odd kernel size";
  int pad_type = param_.pad_type;
  CHECK(pad_type == PadType::kSame || pad_type == PadType::kValid);
  CHECK_EQ(batch_size, inputs[1].shape_[0]);
  CHECK_EQ(channel_num, outputs[0].shape_[1]);
  CHECK_EQ(batch_size, outputs[0].shape_[0]);
  CHECK_EQ(h1, outputs[0].shape_[2]);
  CHECK_EQ(w1, outputs[0].shape_[3]);
  CHECK_EQ(k_y, inputs[1].shape_[3]);
  CHECK_EQ(k_x, inputs[1].shape_[4]);
  mshadow::Tensor<xpu, 4, real_t> data = inputs[0].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> weight = inputs[1].get_with_shape<xpu, 3, real_t>(
                                           Shape3(batch_size * h1 * w1, k_y * k_x, 1), s);
  mshadow::Tensor<xpu, 4, real_t> out = outputs[0].get<xpu, 4, real_t>(s);
  int ele_tmp_data_bytes = sizeof(real_t) * (channel_num * h1 * w1 * k_y * k_x);
  int ele_tmp_out_bytes = sizeof(real_t) * (channel_num * h1 * w1);
  int ele_batch_dot_workspace_bytes = sizeof(real_t*) * 3 * h1 * w1;
  int workspace_ele_size = ele_tmp_data_bytes + ele_tmp_out_bytes + ele_batch_dot_workspace_bytes;
  int batch_step_ = std::min(static_cast<int>((param_.workspace << 20) / workspace_ele_size), batch_size);
  CHECK_GE(batch_step_, 1);
  mshadow::Tensor<xpu, 1, void*> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, void*>(mshadow::Shape1(batch_step_ * workspace_ele_size), s);
  if (kNullOp != req[0]) {
    for (int i = 0; i < batch_size; i += batch_step_) {
      const index_t step = std::min(batch_step_, batch_size - i);
      mshadow::Tensor<xpu, 3, real_t> tmp_data = mshadow::Tensor<xpu, 3, real_t>(
                                              reinterpret_cast<real_t*>(workspace.dptr_),
                                              Shape3(step * h1 * w1, channel_num, k_y * k_x), s);
      mshadow::Tensor<xpu, 3, real_t> tmp_out = mshadow::Tensor<xpu, 3, real_t>(
                                                    reinterpret_cast<real_t*>(
                                                      workspace.dptr_ + step * ele_tmp_data_bytes),
                                                    Shape3(step * h1 * w1, channel_num, 1), s);
      mshadow::Tensor<xpu, 1, real_t*> batch_dot_workspace = mshadow::Tensor<xpu, 1, real_t*>(
                                                    reinterpret_cast<real_t**>(
                                                      workspace.dptr_ + step * (ele_tmp_data_bytes + ele_tmp_out_bytes)),
                                                    Shape1(step * 3 * h1 * w1), s);
      
      if (pad_type == PadType::kValid) {
        tmp_data = reshape(swapaxis<1, 0>(unpack_patch2col(data.Slice(i, i + step),
                                                           param_.kernel[0],
                                                           param_.kernel[1],
                                                           param_.stride[0],
                                                           param_.stride[1],
                                                           param_.dilate[0],
                                                           param_.dilate[1])),
                           Shape3(step * h1 * w1, channel_num, k_y * k_x));
      } else {
        int pad_y = param_.dilate[0] * (param_.kernel[0] - 1) / 2;
        int pad_x = param_.dilate[1] * (param_.kernel[1] - 1) / 2;
        tmp_data = reshape(swapaxis<1, 0>(unpack_patch2col(pad(data.Slice(i, i + step), pad_y, pad_x),
                                                            param_.kernel[0],
                                                            param_.kernel[1],
                                                            param_.stride[0],
                                                            param_.stride[1],
                                                            param_.dilate[0],
                                                            param_.dilate[1])),
                           Shape3(step * h1 * w1, channel_num, k_y * k_x));
      }
      mshadow::BatchGEMM<false, false>(tmp_out,
                                       tmp_data, weight.Slice(i * h1 * w1, (i + step) * h1 * w1), 1.0f,
                                       0.0f, batch_dot_workspace);
      Assign(out.Slice(i, i + step), req[0], transpose(reshape(tmp_out,
                                                               Shape4(step, h1, w1, channel_num)),
                                                       Shape4(0, 3, 1, 2)));
    }
  }
}

template<typename xpu>
void LocalFilterBackward_(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const LocalFilterParam& param_ = nnvm::get<LocalFilterParam>(attrs.parsed);
  CHECK_NE(req[0], kWriteInplace);
  CHECK_NE(req[1], kWriteInplace);
  int batch_size = inputs[1].shape_[0];
  int channel_num = inputs[1].shape_[1];
  int h2 = inputs[1].shape_[2];
  int w2 = inputs[1].shape_[3];
  int h1 = inputs[2].shape_[1];
  int w1 = inputs[2].shape_[2];
  int k_y = param_.kernel[0];
  int k_x = param_.kernel[1];
  mshadow::Tensor<xpu, 4, real_t> out_grad = inputs[0].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 4, real_t> data = inputs[1].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> weight = inputs[2].get_with_shape<xpu, 3, real_t>(
                                               Shape3(batch_size * h1 * w1, 1, k_y * k_x), s);
  mshadow::Tensor<xpu, 4, real_t> data_grad = outputs[0].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> weight_grad = outputs[1].get_with_shape<xpu, 3, real_t>(
                                               Shape3(batch_size * h1 * w1, 1, k_y * k_x), s);
  int ele_tmp_data_bytes = sizeof(real_t) * (channel_num * h1 * w1 * k_y * k_x);
  int ele_tmp_out_grad_bytes = sizeof(real_t) * (channel_num * h1 * w1);
  int ele_batch_dot_workspace_bytes = sizeof(real_t*) * 3 * h1 * w1;
  int workspace_ele_size = ele_tmp_data_bytes + ele_tmp_out_grad_bytes + ele_batch_dot_workspace_bytes;
  int batch_step_ = std::min(static_cast<int>((param_.workspace << 20) / workspace_ele_size), batch_size);
  CHECK_GE(batch_step_, 1);
  mshadow::Tensor<xpu, 1, void*> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, void*>(mshadow::Shape1(batch_step_ * workspace_ele_size), s);
  for (int i = 0; i < batch_size; i += batch_step_) {
    const index_t step = std::min(batch_step_, batch_size - i);
    mshadow::Tensor<xpu, 3, real_t> tmp_data = mshadow::Tensor<xpu, 3, real_t>(
                                            reinterpret_cast<real_t*>(workspace.dptr_),
                                            Shape3(step * h1 * w1, channel_num, k_y * k_x), s);
    mshadow::Tensor<xpu, 3, real_t> tmp_out_grad = mshadow::Tensor<xpu, 3, real_t>(
                                                  reinterpret_cast<real_t*>(
                                                    workspace.dptr_ + step * ele_tmp_data_bytes),
                                                  Shape3(step * h1 * w1, 1, channel_num), s);
    mshadow::Tensor<xpu, 1, real_t*> batch_dot_workspace = mshadow::Tensor<xpu, 1, real_t*>(
                                                  reinterpret_cast<real_t**>(
                                                    workspace.dptr_ + step * (ele_tmp_data_bytes + ele_tmp_out_grad_bytes)),
                                                  Shape1(step * 3 * h1 * w1), s);
    if (param_.pad_type == PadType::kValid) {
      tmp_data = reshape(swapaxis<1, 0>(unpack_patch2col(data.Slice(i, i + step),
                                                          param_.kernel[0],
                                                          param_.kernel[1],
                                                          param_.stride[0],
                                                          param_.stride[1],
                                                          param_.dilate[0],
                                                          param_.dilate[1])),
                          Shape3(step * h1 * w1, channel_num, k_y * k_x));
    } else {
      int pad_y = param_.dilate[0] * (param_.kernel[0] - 1) / 2;
      int pad_x = param_.dilate[1] * (param_.kernel[1] - 1) / 2;
      tmp_data = reshape(swapaxis<1, 0>(unpack_patch2col(pad(data.Slice(i, i + step), pad_y, pad_x),
                                                          param_.kernel[0],
                                                          param_.kernel[1],
                                                          param_.stride[0],
                                                          param_.stride[1],
                                                          param_.dilate[0],
                                                          param_.dilate[1])),
                          Shape3(step * h1 * w1, channel_num, k_y * k_x));
    }
    tmp_out_grad = reshape(transpose(out_grad.Slice(i, i + step), Shape4(0, 2, 3, 1)),
                           Shape3(step * h1 * w1, 1, channel_num));
    mshadow::BatchGEMM<false, false>(weight_grad.Slice(i * h1 * w1, (i + step) * h1 * w1),
                                     tmp_out_grad, tmp_data, 1.0f,
                                     (req[1] == kAddTo) ? 1.0f : 0.0f,
                                     batch_dot_workspace);
    mshadow::BatchGEMM<true, false>(tmp_data,
                                    tmp_out_grad, weight.Slice(i * h1 * w1, (i + step) * h1 * w1), 1.0f,
                                    0.0f, batch_dot_workspace);
    if (param_.pad_type == PadType::kValid) {
      Assign(data_grad.Slice(i, i + step), req[0],
              pack_col2patch(swapaxis<1, 0>(reshape(tmp_data,
                                                    Shape2(step * h1 * w1, channel_num * k_y * k_x))),
                            data.Slice(i, i + step).shape_,
                            param_.kernel[0],
                            param_.kernel[1],
                            param_.stride[0],
                            param_.stride[1],
                            param_.dilate[0],
                            param_.dilate[1]));
    } else {
      Shape<4> pshape = data.Slice(i, i + step).shape_;
      pshape[2] += param_.dilate[0] * (param_.kernel[0] - 1);
      pshape[3] += param_.dilate[1] * (param_.kernel[1] - 1);
      Assign(data_grad.Slice(i, i + step), req[0],
              crop(pack_col2patch(swapaxis<1, 0>(reshape(tmp_data,
                                                    Shape2(step * h1 * w1, channel_num * k_y * k_x))),
                                  pshape,
                                  param_.kernel[0],
                                  param_.kernel[1],
                                  param_.stride[0],
                                  param_.stride[1],
                                  param_.dilate[0],
                                  param_.dilate[1]),
                  Shape2(h2, w2)));
    }
  }
}

inline bool LocalFilterShape(const nnvm::NodeAttrs& attrs,
                                  std::vector<TShape> *in_attrs,
                                  std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  const LocalFilterParam& param_ = nnvm::get<LocalFilterParam>(attrs.parsed);
  CHECK_EQ(param_.kernel.ndim(), 2) << "kernel must be (ky, kx)";
  CHECK(param_.kernel[0] % 2 == 1 && param_.kernel[1] % 2 == 1)
    << "kernel size must be odd, kernel=" << param_.kernel;
  CHECK(param_.stride[0] == 1 && param_.stride[1] == 1) << "stride is not supported and must be 1.";
  TShape& data_shape = (*in_attrs)[0];
  TShape& weight_shape = (*in_attrs)[1];
  CHECK_EQ(data_shape.ndim(), 4);
  CHECK_EQ(weight_shape.ndim(), 5);
  int batch_size = data_shape[0];
  int channel_num = data_shape[1];
  int h2 = data_shape[2];
  int w2 = data_shape[3];
  int h1 = weight_shape[1];
  int w1 = weight_shape[2];
  int ky = weight_shape[3];
  int kx = weight_shape[4];
  // We will first try to complete the shape information
  if (batch_size == 0) {
    batch_size = weight_shape[0];
    CHECK_NE(batch_size, 0) << "shape is invalid! data_shape=" << data_shape << ", weight_shape=" << weight_shape;
  }
  if (h1 == 0 && h2 != 0) {
    h1 = (param_.pad_type == PadType::kValid) ? (h2 - param_.dilate[0] * (param_.kernel[0] - 1)) : h2;
  } else if (h1 != 0 && h2 == 0) {
    h2 = (param_.pad_type == PadType::kValid) ? h1 + param_.dilate[0] * (param_.kernel[0] - 1) : h1;
  }
  if (w1 == 0 && w2 != 0) {
    w1 = (param_.pad_type == PadType::kValid) ? w2 - param_.dilate[0] * (param_.kernel[0] - 1) : w2;
  } else if (w1 != 0 && w2 == 0) {
    w2 = (param_.pad_type == PadType::kValid) ? w1 + param_.dilate[1] * (param_.kernel[1] - 1) : w1;
  }
  mshadow::Shape<4> out_shape;
  out_shape[0] = batch_size;
  out_shape[1] = channel_num;
  out_shape[2] = h1;
  out_shape[3] = w1;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  batch_size = (*out_attrs)[0][0];
  channel_num = (*out_attrs)[0][1];
  h1 = (*out_attrs)[0][2];
  w1 = (*out_attrs)[0][3];
  h2 = (param_.pad_type == PadType::kValid) ? h1 + param_.dilate[0] * (param_.kernel[0] - 1) : h1;
  w2 = (param_.pad_type == PadType::kValid) ? w1 + param_.dilate[1] * (param_.kernel[1] - 1) : w1;
  CHECK(h1 != 0 && w1 != 0) << "Incomplete shape inference failed! data_shape="
                            << data_shape << ", weight_shape=" << weight_shape;
  CHECK_EQ(ky, param_.kernel[0]);
  CHECK_EQ(kx, param_.kernel[1]);
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::Shape4(batch_size, channel_num, h2, w2));
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::Shape5(batch_size, h1, w1, ky, kx));
  return true;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_MATRIX_OP_INL_H_

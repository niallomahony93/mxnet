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
#include "./local_sparse_filter.h"

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

struct LocalSparseFilterParam : public dmlc::Parameter<LocalSparseFilterParam> {
  int num_filter;
  float pad_val;
  uint64_t workspace;
  DMLC_DECLARE_PARAMETER(LocalSparseFilterParam) {
    DMLC_DECLARE_FIELD(num_filter).set_lower_bound(1)
    .describe("Number of filters.");
    DMLC_DECLARE_FIELD(pad_val).set_default(1.0)
    .describe("The padding value, indicates the state of the outside world.");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_range(0, 8192)
    .describe("Maximum tmp workspace allowed for sparse local filter (MB).");
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
void LocalSparseFilterForward_(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const LocalSparseFilterParam& param_ = nnvm::get<LocalSparseFilterParam>(attrs.parsed);
  CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32)
    << "LocalSparseFilter only support 32 bit float so far";
  mshadow::Tensor<xpu, 4, real_t> data = inputs[0].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> weight = inputs[1].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 1, real_t> bias = inputs[2].get<xpu, 1, real_t>(s);
  mshadow::Tensor<xpu, 5, real_t> values = inputs[3].get<xpu, 5, real_t>(s);
  mshadow::Tensor<xpu, 5, real_t> indices = inputs[4].get<xpu, 5, real_t>(s);
  mshadow::Tensor<xpu, 4, real_t> out = outputs[0].get<xpu, 4, real_t>(s);
  Tensor<xpu, 1, real_t> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, real_t>(Shape1(inputs[0].Size()), s);
  Tensor<xpu, 4, real_t> transposed_data = Tensor<xpu, 4, real_t>(workspace.dptr_,
    Shape4(data.shape_[0], data.shape_[2], data.shape_[3], data.shape_[1]), s);  // contain transposed data for coalescing access
  transposed_data = transpose(data, Shape4(0, 2, 3, 1));
  CHECK_EQ(transposed_data.CheckContiguous(), true);
  CHECK_EQ(weight.CheckContiguous(), true);
  CHECK_EQ(bias.CheckContiguous(), true);
  CHECK_EQ(values.CheckContiguous(), true);
  CHECK_EQ(indices.CheckContiguous(), true);
  CHECK_EQ(out.CheckContiguous(), true);
  if (req[0] == kNullOp) {
    return;
  } else if (req[0] == kWriteTo) {
    LocalSparseFilterForwardImpl<real_t>(transposed_data, weight, bias, values, indices, out, param_.pad_val);
  } else {
    LOG(FATAL) << "Not implemented, req=" << req[0];
  }
}

template<typename xpu>
void LocalSparseFilterBackward_(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const LocalSparseFilterParam& param_ = nnvm::get<LocalSparseFilterParam>(attrs.parsed);
  CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32)
    << "LocalSparseFilter only support 32 bit float so far";
  mshadow::Tensor<xpu, 4, real_t> out_grad = inputs[0].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 4, real_t> data = inputs[1].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> weight = inputs[2].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 5, real_t> values = inputs[3].get<xpu, 5, real_t>(s);
  mshadow::Tensor<xpu, 5, real_t> indices = inputs[4].get<xpu, 5, real_t>(s);
  mshadow::Tensor<xpu, 4, real_t> data_grad = outputs[0].get<xpu, 4, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> weight_grad = outputs[0].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 1, real_t> bias_grad = outputs[0].get<xpu, 1, real_t>(s);
  mshadow::Tensor<xpu, 5, real_t> values_grad = outputs[0].get<xpu, 5, real_t>(s);
  CHECK_NE(req[0], kWriteInplace);
  CHECK_NE(req[1], kWriteInplace);
  CHECK_NE(req[2], kWriteInplace);
  CHECK_NE(req[3], kWriteInplace);
  Tensor<xpu, 1, real_t> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, real_t>(Shape1(inputs[1].Size() * 2 + inputs[0].Size()), s);
  CHECK_LT(static_cast<uint64_t>(workspace.shape_.Size()) >> 20, param_.workspace);
  Tensor<xpu, 4, real_t> transposed_data = Tensor<xpu, 4, real_t>(workspace.dptr_,
    Shape4(data.shape_[0], data.shape_[2], data.shape_[3], data.shape_[1]), s);  // contain transposed data for coalescing access
  Tensor<xpu, 4, real_t> transposed_data_grad = Tensor<xpu, 4, real_t>(workspace.dptr_,
    Shape4(data.shape_[0], data.shape_[2], data.shape_[3], data.shape_[1]), s);  // contain transposed data for coalescing access
  Tensor<xpu, 4, real_t> transposed_out_grad = Tensor<xpu, 4, real_t>(workspace.dptr_,
    Shape4(out_grad.shape_[0], out_grad.shape_[2], out_grad.shape_[3], out_grad.shape_[1]), s);  // contain transposed data for coalescing access
  transposed_data = transpose(data, Shape4(0, 2, 3, 1));
  transposed_out_grad = transpose(out_grad, Shape4(0, 2, 3, 1));
  Assign(bias_grad, req[2], sumall_except_dim<1>(data));
  if (req[0] == kWriteTo) {
    transposed_data_grad = scalar<real_t>(0.0f);
  }
  if (req[1] == kWriteTo) {
    weight_grad = scalar<real_t>(0.0f);
  }
  if (req[3] == kWriteTo) {
    values_grad = scalar<real_t>(0.0f);
  }
  LocalSparseFilterBackwardAccImpl(transposed_out_grad, transposed_data, weight, values, indices,
                                   transposed_data_grad, weight_grad, values_grad,
                                   req[0] != kNullOp, req[1] != kNullOp, req[3] != kNullOp, param_.pad_val);
  LOG(FATAL) << "Not Implemented Error";
}

inline bool LocalSparseFilterShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape> *in_attrs,
                                   std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 5);
  CHECK_EQ(out_attrs->size(), 1);
  const LocalSparseFilterParam& param_ = nnvm::get<LocalSparseFilterParam>(attrs.parsed);
  TShape& data_shape = (*in_attrs)[0];
  TShape& values_shape = (*in_attrs)[3];
  CHECK_EQ(data_shape.ndim(), 4);
  int B = data_shape[0];
  int inC = data_shape[1];
  int H = data_shape[2];
  int W = data_shape[3];
  int L = values_shape[1];
  int K = values_shape[2];
  int outC = param_.num_filter;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape4(B, outC, H, W));
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::Shape3(L, outC, inC));
  SHAPE_ASSIGN_CHECK(*in_attrs, 2, mshadow::Shape1(outC));
  SHAPE_ASSIGN_CHECK(*in_attrs, 3, mshadow::Shape5(B, L, K, H, W));
  SHAPE_ASSIGN_CHECK(*in_attrs, 4, mshadow::Shape5(B, L, K, H, W));
  return true;
}

struct BinaryStochasticNeuronParam : public dmlc::Parameter<BinaryStochasticNeuronParam> {
  bool stochastic_train;
  bool stochastic_test;
  DMLC_DECLARE_PARAMETER(BinaryStochasticNeuronParam) {
    DMLC_DECLARE_FIELD(stochastic_train).set_default(true)
      .describe("Whether to draw samples in the training process.");
    DMLC_DECLARE_FIELD(stochastic_test).set_default(false)
      .describe("Whether to draw samples in the testing process. (is_train=False)");
  }
};

template<typename xpu>
void BinaryStochasticNeuronCompute(const nnvm::NodeAttrs& attrs,
  const OpContext& ctx,
  const std::vector<TBlob>& inputs,
  const std::vector<OpReqType>& req,
  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const BinaryStochasticNeuronParam& param = nnvm::get<BinaryStochasticNeuronParam>(attrs.parsed);
  // Use sigmoid activation to compute the probability of the Bernoulli distribution
  if ((ctx.is_train && param.stochastic_train) || (!ctx.is_train && param.stochastic_test)) {
    CHECK(req[0] == kWriteTo || req[0] == kWriteInplace);
    CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32) << "only support float32 rnd so far";
    Tensor<xpu, 1, float> bsn_out = outputs[0].FlatTo1D<xpu, float>(s);
    Tensor<xpu, 1, float> sigmoid_out = outputs[1].FlatTo1D<xpu, float>(s);
    sigmoid_out = F<mshadow_op::sigmoid>(inputs[0].FlatTo1D<xpu, float>(s));
    mshadow::Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
    prnd->SampleUniform(&bsn_out, 0.0, 1.0);
    bsn_out = F<mshadow_op::lt>(bsn_out, sigmoid_out);
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> bsn_out = outputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> sigmoid_out = outputs[1].FlatTo1D<xpu, DType>(s);
      sigmoid_out = F<mshadow_op::sigmoid>(inputs[0].FlatTo1D<xpu, DType>(s));
      ASSIGN_DISPATCH(bsn_out, req[0], F<mshadow_op::round>(sigmoid_out));
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_MATRIX_OP_INL_H_

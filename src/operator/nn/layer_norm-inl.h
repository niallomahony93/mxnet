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
 * \file layer_norm-inl.h
 * \brief Implements Ba et. al, Layer Normalization (https://arxiv.org/abs/1607.06450).
*/
#ifndef MXNET_OPERATOR_NN_LAYER_NORM_INL_H_
#define MXNET_OPERATOR_NN_LAYER_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/base.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../tensor/broadcast_reduce_op.h"

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

namespace mxnet {
namespace op {

namespace layernorm {
enum LayerNormOpInputs {kData, kGamma, kBeta};  // kGamma: scaling parameters, kBeta: shift biases
enum LayerNormOpOutputs {kOut, kMean, kStd};  // req, out_data
}  // namespace layernorm

struct LayerNormParam : public dmlc::Parameter<LayerNormParam> {
  int axis;
  float eps;
  bool output_mean_var;
  DMLC_DECLARE_PARAMETER(LayerNormParam) {
    DMLC_DECLARE_FIELD(axis).set_default(1)
      .describe("The axis to perform layer normalization. "
                "Usually, this should be be axis of the channel dimension. "
                "Negative values means indexing from right to left. ");
    DMLC_DECLARE_FIELD(eps).set_default(1e-5f)
      .describe("An `epsilon` parameter to prevent division by 0.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
      .describe("Output the mean and std calculated along the given axis");
  }
};


template<typename xpu>
void LayerNormCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx, const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  int axis = param.axis;
  if (axis < 0) {
    axis += static_cast<int>(inputs[0].ndim());
  }
  CHECK(axis >= 0 && axis < inputs[0].ndim()) << "Channel axis out of range: " << param.axis;
  CHECK_EQ(inputs.size(), 3U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // Initialize the output to be the same as the input
  mxnet_op::copy(s, outputs[0], inputs[0]);
  // Compute necessary data for the reduce operation.
  TShape red_src_shape, red_dst_shape;
  BroadcastReduceShapeCompact(inputs[0].shape_, outputs[layernorm::kMean].shape_,
                              &red_src_shape, &red_dst_shape);
  const TBlob in_data = inputs[0].reshape(red_src_shape);
  const TBlob mean_data = outputs[layernorm::kMean].reshape(red_dst_shape);
  const TBlob std_data = outputs[layernorm::kStd].reshape(red_dst_shape);
  int normalize_size = red_src_shape.Size() / red_dst_shape.Size();
  // Get workspace
  Tensor<xpu, 1, char> workspace;
  size_t workspace_size = 0;
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim,{
      workspace_size = broadcast::ReduceWorkspaceSize<NDim, DType>(s, mean_data, req[0], in_data);
    });
  });
  workspace = ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
  // Calculate mean
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    broadcast::Reduce<red::sum, NDim, DType, op::mshadow_op::identity>(
      s, mean_data, req[0], workspace, in_data);
    Tensor<xpu, 2, DType> mean_data_tensor = mean_data.FlatTo2D<xpu, DType>(s);
    mean_data_tensor /= scalar<DType>(normalize_size);
  });
  // Calculate data = data - mean
  BinaryBroadcastCompute<xpu, op::mshadow_op::minus>(attrs, ctx,
                                                     {outputs[0], outputs[layernorm::kMean]},
                                                     {kWriteTo}, {outputs[0]});
  // Calculate std
  const TBlob centered_out = outputs[0].reshape(red_src_shape);
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    broadcast::Reduce<red::sum, NDim, DType, op::mshadow_op::square>(
      s, std_data, req[0], workspace, centered_out);
    Tensor<xpu, 2, DType> std_data_tensor = std_data.FlatTo2D<xpu, DType>(s);
    std_data_tensor = F<mshadow_op::square_root>(std_data_tensor / scalar<DType>(normalize_size) + scalar<DType>(param.eps));
  });
  // Calculate data = data / std
  BinaryBroadcastCompute<xpu, op::mshadow_op::div>(attrs, ctx,
                                                   {outputs[0], outputs[layernorm::kStd]},
                                                   {kWriteTo}, {outputs[0]});
}

template<typename xpu>
void LayerNormGradCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx, const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 5U);
  LOG(FATAL) << "Not implemented";
}

}  // namespace op
}  // namespace mxnet
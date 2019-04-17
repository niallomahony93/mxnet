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

namespace mxnet {
namespace op {


void LayerNormCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx, const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  return LayerNormComputeGeneral<gpu>(attrs, ctx, inputs, req, outputs);
}

NNVM_REGISTER_OP(LayerNorm)
.set_attr<FCompute>("FCompute<gpu>", LayerNormCompute);

NNVM_REGISTER_OP(_backward_LayerNorm)
.set_attr<FCompute>("FCompute<gpu>", LayerNormGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

/*!
*  Copyright (c) 2017 by Contributors
* \file nn_addition_op.cc
* \brief CPU Implementation of nn additional operations
*/
// this will be invoked by gcc and compile CPU version
#include "./nn_addition_op-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(LocalCorrelationParam);
DMLC_REGISTER_PARAMETER(LocalFilterParam);

NNVM_REGISTER_OP(LocalCorrelation)
.MXNET_DESCRIBE("Calculate the inner product between the vector lhs_{:, i,j} and rhs_{:, N(i), N(j)}."
                " lhs will be of shape (B, C, H_1, W_1), rhs will be of shape (B, C, H_2, W_2)."
                " The output will have shape (B, H_1, W_1, ksize_y, ksize_x).")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LocalCorrelationParam>)
.set_attr<nnvm::FInferShape>("FInferShape", LocalCorrelationShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", LocalCorrelationForward_<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_LocalCorrelation"})
.add_argument("lhs", "NDArray", "lhs data, the query input data that we want to find the correlation weights")
.add_argument("rhs", "NDArray", "rhs data, the search data that we will try to attend the lhs on.");

NNVM_REGISTER_OP(_backward_LocalCorrelation)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<LocalCorrelationParam>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LocalCorrelationBackward_<cpu>);

/*
NNVM_REGISTER_OP(LocalFilter)
.MXNET_DESCRIBE("Calculate the local convolution between data and weight."
                " data will be of shape (B, C, H_2, W_2), weight will be of shape (B, H_1, W_1, ksize_y, ksize_x)."
                " The output will have shape (B, C, H_1, W_1)."
                " It can actually be viewed as the reverse operation of LocalCorrelation.")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LocalFilterParam>)
.set_attr<nnvm::FInferShape>("FInferShape", LocalFilterShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", LocalFilterForward_<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_LocalFilter"})
.add_argument("data", "NDArray", "The data input, shape (B, C, H_2, W_2)")
.add_argument("weight", "NDArray", "The weight input, shape (B, H_1, W_1, k_h, k_w)");

NNVM_REGISTER_OP(_backward_LocalFilter)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<LocalFilterParam>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LocalFilterBackward_<cpu>);
*/
}  // namespace op
}  // namespace mxnet

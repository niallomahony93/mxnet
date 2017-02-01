/*!
 *  Copyright (c) 2016 by Contributors
 * \file sample_op.cc
 * \brief CPU Implementation of sample op
 */
#include "./sample_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SampleUniformParam);
DMLC_REGISTER_PARAMETER(SampleNormalParam);
DMLC_REGISTER_PARAMETER(BinaryStochasticNeuronParam);

MXNET_OPERATOR_REGISTER_SAMPLE(uniform, SampleUniformParam)
.add_alias("_sample_uniform")
.describe("Sample a uniform distribution")
.set_attr<FCompute>("FCompute<cpu>", SampleUniform_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(normal, SampleNormalParam)
.add_alias("_sample_normal")
.describe("Sample a normal distribution")
.set_attr<FCompute>("FCompute<cpu>", SampleNormal_<cpu>);

// Binary Stochastic Neurons
NNVM_REGISTER_OP(BSN)
.MXNET_DESCRIBE("Binary Stochastic Neurons with the Straight-Through Estimator."
" The input will be first mapped to [0, 1] using the sigmoid activation,"
" which will be further converted to a hard {0, 1} by stochastic sample or"
" deterministic rounding"
" See \"[Arxiv2016]Hierarchical Multiscale Recurrent Neural Networks\""
"for more detail")
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr_parser(ParamParser<BinaryStochasticNeuronParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 2>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 2>)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs", [](const NodeAttrs& attrs) { return 1; })
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{ {0, 1}};
})
.set_attr<FResourceRequest>("FResourceRequest", SampleResource)
.set_attr<FCompute>("FCompute<cpu>", BinaryStochasticNeuronCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", [](const nnvm::NodePtr& n,
  const std::vector<nnvm::NodeEntry>& ograds) {
  std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.begin() + 1);
  for (auto& h : n->inputs) {
    heads.push_back(h);
  }
  return MakeGradNode("_backward_Activation", n, heads, {{"act_type", "sigmoid"}});
})
.add_argument("data", "NDArray", "Source input")
.add_arguments(BinaryStochasticNeuronParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

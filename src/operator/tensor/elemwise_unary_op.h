/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_unary_op-inl.h
 * \brief Function defintion of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../special_functions-inl.h"

namespace mxnet {
namespace op {
template<typename GRAD_OP>
struct unary_bwd {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a*GRAD_OP::Map(b));
  }
};

template<typename xpu, typename OP>
void UnaryCompute(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<OP>(inputs[0].FlatTo1D<xpu, DType>(s)));
  });
}


template<typename xpu>
void IdentityCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (req[0] == kNullOp) return;
  if (req[0] == kWriteInplace) {
    CHECK_EQ(inputs[0].dptr_, outputs[0].dptr_); return;
  }
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<mshadow_op::identity>(inputs[0].FlatTo1D<xpu, DType>(s)));
  });
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

#define MXNET_OPERATOR_REGISTER_UNARY(name)                         \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray", "Source input")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_

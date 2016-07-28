/*!
 *  Copyright (c) 2015 by Contributors
 * \file k_smallest_flags-inl.h
 * \brief Function defintion of matrix related operators
 */
#ifndef MXNET_OPERATOR_K_SMALLEST_FLAGS_INL_H_
#define MXNET_OPERATOR_K_SMALLEST_FLAGS_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "./mshadow_op.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {
    
struct KSmallestFlagsParam : public dmlc::Parameter<KSmallestFlagsParam> {
    int k;
    DMLC_DECLARE_PARAMETER(KSmallestFlagsParam) {
        DMLC_DECLARE_FIELD(k).set_default(1).set_lower_bound(1)
        .describe("Kth smallest.");
    }
};


template<typename xpu>
void KSmallestFlagsForward_(const TBlob& data,
                            const EnvArguments& env,
                            TBlob *ret,
                            OpReqType req,
                            RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  KSmallestFlagsParam param;
  param.Init(env.kwargs);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(data.shape_.ndim(), 2) << "K Smallest Flags only supports input data has ndim=2";
  Tensor<xpu, 2, real_t> out = ret->FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2, real_t> dat = data.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2, real_t> workspace = env.resource[0].get_space_typed<xpu, 2, real_t>(
    mshadow::Shape2(2, data.Size()), s);
  Tensor<xpu, 1, real_t> sorted_dat = workspace[0];
  Tensor<xpu, 1, real_t> indices = workspace[1];
  Copy(sorted_dat, Tensor<xpu, 1, real_t>(dat.dptr_, Shape1(data.Size()), s), s);
  indices = range<real_t>(0, data.shape_[0], 1, data.shape_[1]);
  CHECK_EQ(out.CheckContiguous(), true);
  CHECK_EQ(dat.CheckContiguous(), true);
  CHECK_EQ(sorted_dat.CheckContiguous(), true);
  CHECK_EQ(indices.CheckContiguous(), true);
  VectorizedSort(sorted_dat, indices);
  ASSIGN_DISPATCH(out, req,
    scalar<real_t>(1.0f) -
    F<mshadow_op::threshold>(
      broadcast_keepdim(slice<1>(Tensor<xpu, 2, real_t>(sorted_dat.dptr_, dat.shape_, s),
                                 param.k - 1, param.k), 1, dat.shape_[1]), dat));
}

template<typename xpu>
void KSmallestFlagsBackward_(const OutputGrad& out_grad,
                                  const Input0& inputdata,
                                  const EnvArguments& env,
                                  TBlob* inputdata_grad,
                                  OpReqType req_inputdata_grad,
                                  RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputdata.data.shape_.ndim(), 2) << "K Smallest Flags only supports input data has ndim=2";

  mshadow::Tensor<xpu, 2, real_t> mout_grad = out_grad.data.get<xpu, 2, real_t>(s);
  mshadow::Tensor<xpu, 2, real_t> dat = inputdata.data.get<xpu, 2, real_t>(s);
  mshadow::Tensor<xpu, 2, real_t> dat_grad = inputdata_grad->get<xpu, 2, real_t>(s);
  ASSIGN_DISPATCH(dat_grad, req_inputdata_grad, ScalarExp<real_t>(0.0f));
}

    
inline TShape KSmallestFlagsShape(const TShape& src,
                                  const EnvArguments& env) {
  CHECK_EQ(src.ndim(), 2) << "K Smallest Flags only supports input data has ndim=2";
  return src;
}


// batch_cconv
MXNET_REGISTER_SIMPLE_OP(k_smallest_flags, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, KSmallestFlagsForward_<XPU>, kNoInplace, kRegisterSymbolic)
.set_shape_function(KSmallestFlagsShape)
.set_gradient(XPU::kDevMask, KSmallestFlagsBackward_<XPU>, kNoInplace)
.set_resource_request(ResourceRequest::kTempSpace)
.describe("Set the K Smallest elements in a matrix to 1 and other elements to 0. ")
.add_arguments(KSmallestFlagsParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_K_SMALLEST_FLAGS_INL_H_

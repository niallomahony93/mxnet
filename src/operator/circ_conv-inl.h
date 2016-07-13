/*!
 *  Copyright (c) 2015 by Contributors
 * \file circ_conv-inl.h
 * \brief Function defintion of matrix related operators
 * \author Xingjian Shi
 */
#ifndef MXNET_OPERATOR_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_MATRIX_OP_INL_H_

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

template<typename xpu>
void CircularConvolutionForward_(const TBlob& lhs,
                                 const TBlob& rhs,
                                 const EnvArguments& env,
                                 TBlob *ret,
                                 OpReqType req,
                                 RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(lhs.shape_.ndim(), 2) << "Circular Convolution only supports lhs has ndim=2";
  CHECK_EQ(rhs.shape_.ndim(), 2) << "Circular Convolution only supports rhs has ndim=2";
  Tensor<xpu, 2, real_t> out = ret->FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2, real_t> dat = lhs.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2, real_t> weight = rhs.FlatTo2D<xpu, real_t>(s);
  CHECK_EQ(out.CheckContiguous(), true);
  CHECK_EQ(dat.CheckContiguous(), true);
  CHECK_EQ(weight.CheckContiguous(), true);
  if (req != kAddTo) {
    out = 0.0f;
  }
  CircularConvolution1DForwardImpl_(out, dat, weight);
}

template<typename xpu>
void CircularConvolutionBackward_(const OutputGrad& out_grad,
                                  const Input0& lhs,
                                  const Input1& rhs,
                                  const EnvArguments& env,
                                  TBlob* lhs_grad,
                                  TBlob* rhs_grad,
                                  OpReqType req_lhs_grad,
                                  OpReqType req_rhs_grad,
                                  RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(lhs.data.shape_.ndim(), 2) << "Circular Convolution only supports lhs has ndim=2";
  CHECK_EQ(rhs.data.shape_.ndim(), 2) << "Circular Convolution only supports rhs has ndim=2";

  mshadow::Tensor<xpu, 2, real_t> mout_grad = out_grad.data.get<xpu, 2, real_t>(s);
  mshadow::Tensor<xpu, 2, real_t> dat = lhs.data.get<xpu, 2, real_t>(s);
  mshadow::Tensor<xpu, 2, real_t> weight = rhs.data.get<xpu, 2, real_t>(s);
  mshadow::Tensor<xpu, 2, real_t> dat_grad = lhs_grad->get<xpu, 2, real_t>(s);
  mshadow::Tensor<xpu, 2, real_t> weight_grad = rhs_grad->get<xpu, 2, real_t>(s);
  if (req_lhs_grad != kAddTo) {
    dat_grad = 0.0f;
  }
  if (req_rhs_grad != kAddTo) {
    weight_grad = 0.0f;
  }
  CircularConvolution1DBackwardImpl_(mout_grad, dat_grad, weight_grad, dat, weight);
}


inline TShape CircularConvolutionShape(const TShape& lshape,
                                       const TShape& rshape,
                                       const EnvArguments& env) {
  CHECK_EQ(lshape.ndim(), 2) << "Circular Convolution only supports lhs has ndim=2";
  CHECK_EQ(rshape.ndim(), 2) << "Circular Convolution only supports rhs has ndim=2";
  return lshape;
}


// batch_cconv
MXNET_REGISTER_SIMPLE_OP(batch_cconv, XPU)
.set_function(XPU::kDevMask, CircularConvolutionForward_<XPU>, kNoInplace, kRegisterSymbolic)
.set_shape_function(CircularConvolutionShape)
.set_gradient(XPU::kDevMask, CircularConvolutionBackward_<XPU>, kNoInplace)
.describe("Calculate batched circular convolution of two matrix. Follows Matlab cconv syntax"
          " (Batchsize, N) conv (Batchsize, K) -> (Batchsize, N)");
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MATRIX_OP_INL_H_

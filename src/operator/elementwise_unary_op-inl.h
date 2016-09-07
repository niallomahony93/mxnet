/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_unary_op-inl.h
 * \brief Function defintion of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_ELEMENTWISE_UNARY_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_UNARY_OP_INL_H_

#include <mxnet/operator_util.h>
#include "./mshadow_op.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

template<typename xpu, typename OP>
void UnaryForward_(const TBlob& src,
                   const EnvArguments& env,
                   TBlob *ret,
                   OpReqType req,
                   RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, src.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> out = ret->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req, F<OP>(src.FlatTo1D<xpu, DType>(s)));
  });
}

// backward function that takes input value of the op
template<typename xpu, typename OP>
void UnaryBackwardUseIn_(const OutputGrad& out_grad,
                         const Input0& in_data0,
                         const EnvArguments& env,
                         TBlob *in_grad,
                         OpReqType req,
                         RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  CHECK_EQ(in_grad->type_flag_, in_data0.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> igrad = in_grad->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    (F<OP>(in_data0.data.FlatTo1D<xpu, DType>(s)) *
                     out_grad.data.FlatTo1D<xpu, DType>(s)));
  });
}

// backward function that takes output value of the op
template<typename xpu, typename OP>
void UnaryBackwardUseOut_(const OutputGrad& out_grad,
                          const OutputValue& out_value,
                          const EnvArguments& env,
                          TBlob *in_grad,
                          OpReqType req,
                          RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  CHECK_EQ(in_grad->type_flag_, out_value.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> igrad = in_grad->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    (F<OP>(out_value.data.FlatTo1D<xpu, DType>(s)) *
                     out_grad.data.FlatTo1D<xpu, DType>(s)));
    });
}

// Compute conj(x)
template<typename xpu>
void ConjForward_(const TBlob& src,
  const EnvArguments& env,
  TBlob *ret,
  OpReqType req,
  RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, src.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req, conj(src.FlatTo2D<xpu, DType>(s)));
  });
}

// backward function of the complex conjugate operator
template<typename xpu>
void ConjBackward_(const OutputGrad& out_grad,
  const EnvArguments& env,
  TBlob* in_grad,
  OpReqType req,
  RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req, conj(out_grad.data.FlatTo2D<xpu, DType>(s)));
  });
}

// Compute the square of the absolute value of the complex tensor A: |A|^2
template<typename xpu>
void ComplexAbsSquareForward_(const TBlob& src,
  const EnvArguments& env,
  TBlob *ret,
  OpReqType req,
  RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, src.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req, complex_abs_square(src.FlatTo2D<xpu, DType>(s)));
  });
}

inline TShape ComplexAbsSquareShape_(const TShape& ishape,
  const EnvArguments& env) {
  TShape ret = ishape;
  ret[ret.ndim() - 1] /= 2;
  return ret;
}

// backward function of the complex_abs_square operator
template<typename xpu>
void ComplexAbsSquareBackward_(const OutputGrad& out_grad,
  const Input0& in_data0,
  const EnvArguments& env,
  TBlob* in_grad,
  OpReqType req,
  RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  CHECK_EQ(in_grad->type_flag_, in_data0.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> ograd = out_grad.data.FlatTo2D<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> idata = in_data0.data.FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req, scalar<DType>(2) * complex_mul_rc(ograd, idata));
  });
}

MXNET_REGISTER_SIMPLE_OP(abs, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::abs>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::sign>, kInplaceOutIn)
.describe("Take absolute value of the src");
// sign
MXNET_REGISTER_SIMPLE_OP(sign, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::sign>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::sign_grad>, kInplaceOutIn)
.describe("Take sign value of the src");
// round
MXNET_REGISTER_SIMPLE_OP(round, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::round>, kInplaceInOut)
.describe("Take round value of the src");
// ceil
MXNET_REGISTER_SIMPLE_OP(ceil, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::ceil>, kInplaceInOut)
.describe("Take ceil value of the src");
// floor
MXNET_REGISTER_SIMPLE_OP(floor, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::floor>, kInplaceInOut)
.describe("Take floor value of the src");
// square
MXNET_REGISTER_SIMPLE_OP(square, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::square>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::square_grad>, kInplaceOutIn)
.describe("Take square of the src");
// sqrt
MXNET_REGISTER_SIMPLE_OP(sqrt, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::square_root>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, mshadow_op::square_root_grad>, kInplaceOutIn)
.describe("Take sqrt of the src");
// rsqrt
MXNET_REGISTER_SIMPLE_OP(rsqrt, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::reciprocal_square_root>, kInplaceInOut)
.set_gradient(XPU::kDevMask,
              UnaryBackwardUseIn_<XPU, mshadow_op::reciprocal_square_root_grad>, kInplaceOutIn)
.describe("Take rsqrt of the src");
// exp
MXNET_REGISTER_SIMPLE_OP(exp, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::exp>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, mshadow_op::identity>, kInplaceOutIn)
.describe("Take exp of the src");
// log
MXNET_REGISTER_SIMPLE_OP(log, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::log>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::log_grad>, kInplaceOutIn)
.describe("Take log of the src");
// cos
MXNET_REGISTER_SIMPLE_OP(cos, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::cos>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::cos_grad>, kInplaceOutIn)
.describe("Take cos of the src");
// sin
MXNET_REGISTER_SIMPLE_OP(sin, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::sin>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::sin_grad>, kInplaceOutIn)
.describe("Take sin of the src");
// clip_zero_one
MXNET_REGISTER_SIMPLE_OP(clip_zero_one, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::clip_zero_one>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::clip_zero_one_grad>, kInplaceOutIn)
.describe("Clip the src to 0 and 1");
// conj
MXNET_REGISTER_SIMPLE_OP(conj, XPU)
.set_function(XPU::kDevMask, ConjForward_<XPU>, kInplaceInOut)
.set_gradient(XPU::kDevMask, ConjBackward_<XPU>, kInplaceOutIn)
.describe("Take complex conjugate of the src");
// complex_abs_square
MXNET_REGISTER_SIMPLE_OP(complex_abs_square, XPU)
.set_shape_function(ComplexAbsSquareShape_)
.set_function(XPU::kDevMask, ComplexAbsSquareForward_<XPU>, kNoInplace)
.set_gradient(XPU::kDevMask, ComplexAbsSquareBackward_<XPU>, kNoInplace)
.describe("Take square of the absolute value of the src, which is a complex tensor");

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_UNARY_OP_INL_H_

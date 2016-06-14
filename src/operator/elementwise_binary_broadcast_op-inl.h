/*!
 *  Copyright (c) 2016 by Contributors
 * \file elementwise_binary_broadcast_op-inl.h
 * \brief Function defintion of elementwise binary operators with broadcast
 *
 * For example,
 *
 *   [1, 2] + [1,  = [2, 3;
 *             2 ]    3, 4]
 *
 * The shapes broacast of the above example
 *
 *   A      (2d tensor):  1 x 2
 *   B      (1d tensor):  2 x 1
 *   Result (2d tensor):  2 x 2
 *
 * More examples
 *
 *   A      (3d tensor):  15 x 3 x 5
 *   B      (2d tensor):   1 x 1 x 5
 *   Result (3d tensor):  15 x 3 x 5
 *
 *   A      (3d tensor):  15 x 3 x 5
 *   B      (2d tensor):   1 x 3 x 1
 *   Result (3d tensor):  15 x 3 x 5
 *
 * Here are examples of shapes that do not broadcast:
 *
 *   A      (1d tensor):  3
 *   B      (1d tensor):  4 # trailing dimensions do not match
 *
 *   A      (2d tensor):  1 x 2 x 1
 *   B      (3d tensor):  8 x 4 x 3 # second from last dimensions mismatched
 *
 * When no broadcast is need, it falls back to elementwise_binary_op-inl.h
 */
#ifndef MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <vector>
#include "./mshadow_op.h"
#include "./broadcast_reduce_op_common.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

inline bool IsBroadcastNeeded_(const TShape& lhs,
                              const TShape& rhs) {
  // force ndim to be equal. do not smartly padding dims with 1s, which may confuse users
  CHECK_EQ(lhs.ndim(), rhs.ndim());
  for (index_t i = 0; i < lhs.ndim(); ++i) {
    if (lhs[i] != rhs[i]) return true;
  }
  return false;
}


inline TShape BinaryBroadcastShape_(const TShape& lhs,
                                    const TShape& rhs,
                                    const EnvArguments& env) {
  if (!IsBroadcastNeeded_(lhs, rhs)) return lhs;
  std::vector<index_t> ret(lhs.ndim());
  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = std::max(lhs[i], rhs[i]);
  }
  return TShape(ret.begin(), ret.end());
}


template<typename xpu, typename OP>
void BinaryBroadcastForward_(const TBlob& lhs,
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
  CHECK_EQ(lhs.shape_.ndim(), rhs.shape_.ndim()) << "the ndim of lhs and rhs must be equal,"
    " shape of lhs=" << lhs.shape_ << " shape of rhs=" << rhs.shape_;
  if (!IsBroadcastNeeded_(lhs.shape_, rhs.shape_)) {
    // no broadcast
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
      ASSIGN_DISPATCH(out, req,
        F<OP>(lhs.FlatTo2D<xpu, DType>(s),
        rhs.FlatTo2D<xpu, DType>(s)));
    });
    return;
  }
  CHECK(lhs.shape_.ndim() < MXNET_MAX_RANGE_SWITCH_DIM) << "Only support input dimension up to " << MXNET_MAX_RANGE_SWITCH_DIM;
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    MXNET_RANGE_SWITCH(ret->ndim(), NDIM, {
      Tensor<xpu, NDIM, DType> out = ret->get<xpu, NDIM, DType>(s);
      Tensor<xpu, NDIM, DType> mlhs = lhs.get<xpu, NDIM, DType>(s);
      Tensor<xpu, NDIM, DType> mrhs = rhs.get<xpu, NDIM, DType>(s);
      ASSIGN_DISPATCH(out, req,
        F<OP>(broadcast_to(mlhs, ret->shape_), broadcast_to(mrhs, ret->shape_)));
    });
  });
}


template<typename xpu, typename LHS_OP, typename RHS_OP>
void BinaryBroadcastBackward_(const OutputGrad& out_grad,
                              const EnvArguments& env,
                              TBlob* lhs_grad,
                              TBlob* rhs_grad,
                              OpReqType req_lhs_grad,
                              OpReqType req_rhs_grad,
                              RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(out_grad.data.type_flag_, lhs_grad->type_flag_)
    << "Binary function only support ingrad/outgrad with the same type";
  CHECK_EQ(out_grad.data.type_flag_, rhs_grad->type_flag_)
    << "Binary function only support ingrad/outgrad with the same type";
  CHECK_EQ(rhs_grad->shape_.ndim(), rhs_grad->shape_.ndim()) <<
    "the ndim of lhs_grad and rhs_grad must be equal,"
    " shape of lhs_grad=" << lhs_grad->shape_ << " shape of rhs_grad=" << rhs_grad->shape_;
  if (!IsBroadcastNeeded_(lhs_grad->shape_, rhs_grad->shape_)) {
    // no broadcast
    MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
        Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
        ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, F<LHS_OP>(mout_grad));
        ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, F<RHS_OP>(mout_grad));
      });
    return;
  }
  CHECK(lhs_grad->ndim() <= MXNET_MAX_RANGE_SWITCH_DIM)
    << "Only support input dimension up to " << MXNET_MAX_RANGE_SWITCH_DIM;
  MSHADOW_REAL_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
    MXNET_RANGE_SWITCH(lhs_grad->ndim(), NDIM, {
      Tensor<xpu, NDIM, DType> mout_grad = out_grad.data.get<xpu, NDIM, DType>(s);
      Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_grad->Size()), s);
      Tensor<xpu, 1, DType> mrhs_grad =
        rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_grad->Size()), s);
      reduce_to_assign<red::sum>(mlhs_grad, req_lhs_grad, lhs_grad->shape_, F<LHS_OP>(mout_grad));
      reduce_to_assign<red::sum>(mrhs_grad, req_rhs_grad, rhs_grad->shape_, F<RHS_OP>(mout_grad));
    });
  });
}

template<typename xpu>
void BroadcastMulBackward_(const OutputGrad& out_grad,
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
  if (!IsBroadcastNeeded_(lhs_grad->shape_, rhs_grad->shape_)) {
    MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mlhs_data = lhs.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mrhs_data = rhs.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, mlhs_data * mout_grad);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, mrhs_data * mout_grad);
    });
    return;
  }
  
  MSHADOW_REAL_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
    MXNET_RANGE_SWITCH(lhs_grad->ndim(), NDIM, {
      mshadow::Tensor<xpu, NDIM, DType> mout_grad = out_grad.data.get<xpu, NDIM, DType>(s);
      mshadow::Tensor<xpu, NDIM, DType> mlhs_data = lhs.data.get<xpu, NDIM, DType>(s);
      mshadow::Tensor<xpu, NDIM, DType> mrhs_data = rhs.data.get<xpu, NDIM, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_grad->Size()), s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad =
        rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_grad->Size()), s);
      reduce_to_assign<red::sum>(mrhs_grad, req_rhs_grad, rhs_grad->shape_,
        broadcast_to(mlhs_data, out_grad.data.shape_) * mout_grad);
      reduce_to_assign<red::sum>(mlhs_grad, req_lhs_grad, lhs_grad->shape_,
        broadcast_to(mrhs_data, out_grad.data.shape_) * mout_grad);
    });
  });
}

template<typename xpu>
void BroadcastDivBackward_(const OutputGrad& out_grad,
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
  if (!IsBroadcastNeeded_(lhs_grad->shape_, rhs_grad->shape_)) {
    MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mlhs_data = lhs.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mrhs_data = rhs.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
        F<mshadow_op::negation>(mout_grad * mlhs_data) /
        F<mshadow_op::square>(mrhs_data));
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, mout_grad / mrhs_data);
    });
    return;
  }
  MSHADOW_REAL_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
    MXNET_RANGE_SWITCH(lhs_grad->ndim(), NDIM, {
      mshadow::Tensor<xpu, NDIM, DType> mout_grad = out_grad.data.get<xpu, NDIM, DType>(s);
      mshadow::Tensor<xpu, NDIM, DType> mlhs_data = lhs.data.get<xpu, NDIM, DType>(s);
      mshadow::Tensor<xpu, NDIM, DType> mrhs_data = rhs.data.get<xpu, NDIM, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_grad->Size()), s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad =
        rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_grad->Size()), s);
      reduce_to_assign<red::sum>(mrhs_grad, req_rhs_grad, rhs_grad->shape_,
        F<mshadow_op::negation>(mout_grad * broadcast_to(mlhs_data, out_grad.data.shape_)) /
        F<mshadow_op::square>(broadcast_to(mrhs_data, out_grad.data.shape_)));
      reduce_to_assign<red::sum>(mlhs_grad, req_lhs_grad, lhs_grad->shape_, mout_grad /
        broadcast_to(mrhs_data, out_grad.data.shape_));
    });
  });
}


MXNET_REGISTER_SIMPLE_OP(broadcast_plus, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::plus>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BinaryBroadcastBackward_<
              XPU, mshadow_op::identity, mshadow_op::identity>, kNoInplace)
.describe("lhs add rhs with broadcast");

MXNET_REGISTER_SIMPLE_OP(broadcast_minus, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::minus>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BinaryBroadcastBackward_<
              XPU, mshadow_op::identity, mshadow_op::negation>, kNoInplace)
.describe("lhs minus rhs with broadcast");

MXNET_REGISTER_SIMPLE_OP(broadcast_mul, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::mul>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BroadcastMulBackward_<XPU>, kNoInplace)
.describe("lhs multiple rhs with broadcast");

MXNET_REGISTER_SIMPLE_OP(broadcast_div, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::div>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BroadcastDivBackward_<XPU>, kNoInplace)
.describe("lhs divide rhs with broadcast");

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_

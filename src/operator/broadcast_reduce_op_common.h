/*!
* Copyright (c) 2016 by Contributors
* \file broadcast_reduce_op_common.h
* \brief common function used for broadcasting and reducing
* \author Xingjian Shi
*/
#ifndef MXNET_OPERATOR_BROADCAST_REDUCE_OP_COMMON_H_
#define MXNET_OPERATOR_BROADCAST_REDUCE_OP_COMMON_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <vector>

namespace mxnet {
namespace op {

/*!
* \brief a reduce over multiple axes and assign to the output tensor.
* \param out output tensor, must have dim 1
* \param src the source expression
* \param axes the given axes, should be in increasing order
* \tparam Reducer type of the reducing operation
* \tparam xpu
* \tparam SrcExp the src expression template
* \tparam etype type of expression
*/
template<typename Reducer, typename xpu, typename SrcExp, typename DType>
void reduce_multi_axes_assign(mshadow::Tensor<xpu, 1, DType> out, const OpReqType req,
  const SrcExp &src_, const TShape &axes) {
  using namespace mshadow;
  using namespace mshadow::expr;
  static const int dimsrc = ExpInfo<SrcExp>::kDim;
  CHECK(axes.ndim() <= dimsrc);
  Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src_);

  // 1. Check if the axes has size 0, if so, no reducing is needed.
  if (0 == axes.ndim()) {
    ASSIGN_DISPATCH(out, req, reshape(src_, Shape1(src_shape.ProdShape(0, dimsrc))));
    return;
  }

  // 2. Check if we want to reduce over contiguous axes and get the reducing size.
  //  e.g. (1,2,3) --> contiguous, (1,3) --> noncontiguous
  bool is_contiguous_axes = true;
  index_t reducing_size = 1;
  for (index_t i = 0; i < axes.ndim(); ++i) {
    reducing_size *= src_shape[axes[i]];
    if (i > 0) {
      is_contiguous_axes = is_contiguous_axes && (axes[i] == (axes[i - 1] + 1));
      CHECK(axes[i - 1] < axes[i]) << "axes must be in increasing order, received axes=" << axes;
    }
  }

  // 3. For contiguous axes, we can always reshape them to (leading, reducing_size, trailing)
  //  and we can then simplify the combination of mshadow symbols.
  if (is_contiguous_axes) {
    index_t leading = 1;
    index_t trailing = 1;
    for (index_t i = 0; i < dimsrc; ++i) {
      if (i < axes[0]) {
        leading *= src_shape[i];
      } else if (i > axes[axes.ndim() - 1]) {
        trailing *= src_shape[i];
      }
    }
    if (1 == leading) {
      ASSIGN_DISPATCH(out, req,
        (reduce_except_dim<1, Reducer>(reshape(src_, Shape2(reducing_size, trailing)))));
    } else if (1 == trailing) {
      ASSIGN_DISPATCH(out, req,
        (reduce_except_dim<0, Reducer>(reshape(src_, Shape2(leading, reducing_size)))));
    } else {
      ASSIGN_DISPATCH(out, req, (reduce_except_dim<0, Reducer>(
        reshape(swapaxis<2, 1>(reshape(src_, Shape3(leading, reducing_size, trailing))),
        Shape2(leading * trailing, reducing_size)))));
    }
    return;
  }
  // 4. For non-contiguous axes, we need to push axes to the front of the shape vector then reduce.
  //   E.g axes = (1, 2), dim = 6 => transpose_shape = (1, 2, 0, 3, 4, 5)
  Shape<dimsrc> transpose_shape = src_shape;
  index_t remaining_size = 1;
  for (index_t i = 0; i < axes.ndim(); ++i) {
    transpose_shape[i] = axes[i];
    if (i > 0) {
      for (index_t j = axes[i - 1] + 1; j < axes[i]; ++j) {
        transpose_shape[axes.ndim() - i + j] = j;
        remaining_size *= src_shape[j];
      }
    }
    if (axes.ndim() - 1 == i) {
      for (index_t j = axes[axes.ndim() - 1] + 1; j < dimsrc; ++j) {
        transpose_shape[j] = j;
        remaining_size *= src_shape[j];
      }
    }
    if (0 == i) {
      for (index_t j = 0; j < axes[0]; ++j) {
        transpose_shape[axes.ndim() - i + j] = j;
        remaining_size *= src_shape[j];
      }
    }
  }
  ASSIGN_DISPATCH(out, req,
    (reduce_except_dim<1, Reducer>(reshape(transpose(src_, transpose_shape),
    Shape2(reducing_size, remaining_size)))));
}

/*!
* \brief a reduce to the given shape and assign to the output tensor.
* \param out output tensor, must have dim 1
* \param src the source expression
* \param target_shape shape of the target tensor, must have size 1 for the reduction axes
* \tparam Reducer type of the reducing operation
* \tparam xpu
* \tparam SrcExp the src expression template
* \tparam etype type of expression
*/
template<typename Reducer, typename xpu, typename SrcExp, typename DType>
void reduce_to_assign(mshadow::Tensor<xpu, 1, DType> out, const OpReqType req,
  const TShape &target_shape, const SrcExp &src_) {
  using namespace mshadow;
  using namespace mshadow::expr;
  static const int dimsrc = ExpInfo<SrcExp>::kDim;
  std::vector<index_t> axes_vec;
  Shape<dimsrc> src_shape = ShapeCheck<dimsrc, SrcExp>::Check(src_);
  CHECK_EQ(target_shape.ndim(), dimsrc);
  for (int i = 0; i < dimsrc; ++i) {
    if (src_shape[i] != target_shape[i]) {
      CHECK_EQ(target_shape[i], 1) << "reducing axis must have size 1, received src_shape="
        << src_shape << " target_shape=" << target_shape;
      axes_vec.push_back(i);
    }
  }
  TShape axes = TShape(axes_vec.begin(), axes_vec.end());
  reduce_multi_axes_assign<Reducer>(out, req, src_, axes);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BROADCAST_REDUCE_OP_COMMON_H_

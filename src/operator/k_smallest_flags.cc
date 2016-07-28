/*!
 *  Copyright (c) 2015 by Contributors
 * \file k_smallest_flags.cc
 * \brief CPU Implementation of k smallest flag
 * \author Jiani Zhang
 */
// this will be invoked by gcc and compile CPU version
#include "./k_smallest_flags-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(KSmallestFlagsParam);
}  // namespace op
}  // namespace mxnet
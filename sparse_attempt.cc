      // Transpose the grad_in here since the dst of csrmm will be transposed
      Tensor<xpu, 2, DType> grad_in = in_grad[embedding::kWeight].get_with_shape<xpu, 2, DType>(
        Shape2(param_.output_dim, param_.input_dim), s);
      // Parse the data into csr format using tempspace
      index_t word_num = static_cast<index_t>(param_.input_dim);
      Tensor<xpu, 1, int> workspace = ctx.requested[embedding::kTempSpace]
                                         .get_space_typed<xpu, 1, int>(
                                          Shape1(data.shape_.Size() * 2 + 1), s);
      Tensor<xpu, 1, int> data_col_ind = Tensor<xpu, 1, int>(workspace.dptr_,
                                                             Shape1(data.shape_.Size()),
                                                             data.shape_.Size(),
                                                             s);
      Tensor<xpu, 1, int> data_row_ptr = Tensor<xpu, 1, int>(workspace.dptr_ + data.shape_.Size(),
                                                             Shape1(data.shape_.Size() + 1),
                                                             word_num + 1,
                                                             s);
      data_row_ptr = range<int>(0, data.shape_.Size() + 1);
      data_col_ind = tcast<int>(data);
      data = scalar<DType>(1.0f);
      LOG(INFO) << "embedding: kWriteTo";
      // grad_in = scalar<DType>(0.0f);
      // AddTakeGrad(grad_in, data, data_row_ptr, data_col_ind, grad_out);
      csrmm<true, false>(data, data_row_ptr, data_col_ind, grad_out,
                         DType(1.0f), DType(0.0f), grad_in);
      data = tcast<DType>(data_col_ind);
  /*! \brief cusparse handle */
  cusparseHandle_t sparse_handle_;
  /*! \brief cusparse handle ownership */
  HandleState sparse_handle_ownership_;
  st->CreateSparseHandle();
  stream->DestorySparseHandle();
  /*!
  * \brief return actual cusparseHandle
  * \param pointer to GPU stream
  */
  inline static cusparseHandle_t GetSparseHandle(Stream<gpu> *stream) {
    if (stream == NULL) {
      return 0;
    }
    else {
      CHECK_NE(stream->sparse_handle_ownership_, NoHandle)
        << "No handle exist in source stream";
      return stream->sparse_handle_;
    }
  }
  /*! \brief Destory cusparse handle if own it */
  inline void DestorySparseHandle() {
    if (sparse_handle_ownership_ == OwnHandle) {
      cusparseStatus_t err = cusparseDestroy(sparse_handle_);
      sparse_handle_ownership_ = NoHandle;
      CHECK_EQ(err, CUSPARSE_STATUS_SUCCESS) << "Destory cusparse handle failed";
    }
  }
  /*! \brief Destory original sparse handle and create a new one */
  inline void CreateSparseHandle() {
    this->DestorySparseHandle();
    cusparseStatus_t err = cusparseCreate(&sparse_handle_);
    sparse_handle_ownership_ = OwnHandle;
    CHECK_EQ(err, CUSPARSE_STATUS_SUCCESS) << "Create cusparse handle failed";
  }
/*!
 * \brief CPU/GPU: dst^T = alpha * op(A) * op(B) + beta * dst^T;
                   op(A) = trans_A ? A^T : A; op(B) = trans_B ? B^T : B;
                   A is stored in Compressed Sparse Row Format (CSR)
                   Refer to https://en.wikipedia.org/wiki/Sparse_matrix
 * \param A_val value of the nnz elements in A
 * \param A_row_ptr pointer to row elements
 * \param A_col_ind column indices of the nnz elements
 * \param B dense matrix
 * \param alpha
 * \param beta
 * \param dst destination, during computation, the dst will be transposed
 */
template<bool trans_A, bool trans_B, typename DType>
inline void csrmm(const Tensor<cpu, 1, DType> &A_val,
                  const Tensor<cpu, 1, int> &A_row_ptr,
                  const Tensor<cpu, 1, int> &A_col_ind,
                  const Tensor<cpu, 2, DType> &B,
                  DType alpha,
                  DType beta,
                  Tensor<cpu, 2, DType> dst);
/*!
 * \brief CPU/GPU: dst = alpha * op(A) * op(B) + beta * dst;
                   op(A) = trans_A ? A^T : A; op(B) = trans_B ? B^T : B;
                   A is stored in Compressed Sparse Row Format (CSR)
                   Refer to https://en.wikipedia.org/wiki/Sparse_matrix
 * \param A_val value of the nnz elements in A
 * \param A_row_ptr pointer to row elements
 * \param A_col_ind column indices of the nnz elements
 * \param B dense matrix
 * \param alpha
 * \param beta
 * \param dst destination, during computation, the dst will be transposed
 */
template<bool trans_A, bool trans_B, typename DType>
inline void csrmm(const Tensor<gpu, 1, DType> &A_val,
                  const Tensor<gpu, 1, int> &A_row_ptr,
                  const Tensor<gpu, 1, int> &A_col_ind,
                  const Tensor<gpu, 2, DType> &B,
                  DType alpha,
                  DType beta,
                  Tensor<gpu, 2, DType> dst);


// Follows http://www.netlib.org/utk/people/JackDongarra/etemplates/node382.html
template<bool trans_A, bool trans_B, typename DType>
inline void csrmm(const Tensor<cpu, 1, DType> &A_val,
                  const Tensor<cpu, 1, int> &A_row_ptr,
                  const Tensor<cpu, 1, int> &A_col_ind,
                  const Tensor<cpu, 2, DType> &B,
                  DType alpha,
                  DType beta,
                  Tensor<cpu, 2, DType> dst) {
  using namespace mshadow::expr;
  int m = A_row_ptr.shape_.Size() - 1;
  int k = trans_A ? dst.size(1) : (trans_B ? B.size(1) : B.size(0));
  int n = trans_B ? B.size(0) : B.size(1);
  int ldb = B.size(1);
  int nnz = A_val.shape_.Size();
  int ldc = dst.size(1);
  LOG(INFO) << "m = " << m << ", k = " << k << ", n = " << n << ", nnz = " << nnz;
  LOG(INFO) << "Shape: A_val = " << A_val.shape_ <<" A_row_ptr = " << A_row_ptr.shape_ << " A_col_ind = " << A_col_ind.shape_ << " B = " << B.shape_ << " dst = " << dst.shape_;
  CHECK((dst.size(1) == (trans_A ? k : m)) && (dst.size(0) == n)) << "Shape error,"
    << " need dst = (" << n << ", " << (trans_A ? k : m) << "),"
    << " get " << dst.shape_;
  dst *= ScalarExp<DType>(beta);
  if (trans_A) {
    for (index_t j = 0; j < m; ++j) {
      for (index_t k = 0; k < n; ++k) {
        for (index_t i = A_row_ptr[j]; i < A_row_ptr[j + 1]; ++i) {
          dst[k][A_col_ind[i]] += alpha * A_val[i] * (trans_B ? B[k][j] : B[j][k]);
        }
      }
    }
  } else {
    for (index_t j = 0; j < m; j++) {
      for (index_t k = 0; k < n; k++) {
        for (index_t i = A_row_ptr[j]; i < A_row_ptr[j + 1]; ++i) {
          dst[k][i] += alpha * A_val[j] * (trans_B ? B[k][A_col_ind[i]] : B[A_col_ind[i]][k]);
        }
      }
    }
  }
}


// Use cuSPARSE
template<bool trans_A, bool trans_B>
inline void csrmm(const Tensor<gpu, 1, float> &A_val,
                  const Tensor<gpu, 1, int> &A_row_ptr,
                  const Tensor<gpu, 1, int> &A_col_ind,
                  const Tensor<gpu, 2, float> &B,
                  float alpha,
                  float beta,
                  Tensor<gpu, 2, float> dst) {
  CHECK_EQ(A_val.CheckContiguous(), true);
  CHECK_EQ(A_row_ptr.CheckContiguous(), true);
  CHECK_EQ(A_col_ind.CheckContiguous(), true);
  CHECK_EQ(B.CheckContiguous(), true);
  int m = A_row_ptr.shape_.Size() - 1;
  int k = trans_A ? dst.size(1) : (trans_B ? B.size(1) : B.size(0));
  int n = trans_B ? B.size(0) : B.size(1);
  int ldb = B.size(1);
  int nnz = A_val.shape_.Size();
  int ldc = dst.size(1);
  LOG(INFO) << "m = " << m << ", k = " << k << ", n = " << n << ", nnz = " << nnz;
  LOG(INFO) << "Shape: A_val = " << A_val.shape_ << " A_row_ptr = " << A_row_ptr.shape_ << " A_col_ind = " << A_col_ind.shape_ << " B = " << B.shape_ << " dst = " << dst.shape_;
  CHECK((dst.size(1) == (trans_A ? k : m)) && (dst.size(0) == n)) << "Shape error,"
    << " need dst = (" << n << ", " << (trans_A ? k : m) << "),"
    << " get " << dst.shape_;
  CHECK_EQ(A_col_ind.shape_.Size(), nnz) << "col_ind shape: " << A_col_ind.shape_
    << "val shape:" << A_val.shape_;
  cusparseStatus_t err = cusparseSetStream(Stream<gpu>::GetSparseHandle(dst.stream_),
                          Stream<gpu>::GetStream(dst.stream_));
  CHECK_EQ(err, CUSPARSE_STATUS_SUCCESS) << "cusparse: set stream failed";
  cusparseMatDescr_t descr = NULL;
  CHECK_EQ(cusparseCreateMatDescr(&descr), CUSPARSE_STATUS_SUCCESS) << "cusparse: create descr";
  CHECK_EQ(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL), CUSPARSE_STATUS_SUCCESS)
    << "cusparse: set mat type failed";
  CHECK_EQ(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO), CUSPARSE_STATUS_SUCCESS)
    << "cusparse: set mat index failed";
  err = cusparseScsrmm2(Stream<gpu>::GetSparseHandle(dst.stream_),
                       trans_A ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                       trans_B ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE,
                       m, n, k, nnz, &alpha, descr,
                       (const float*)A_val.dptr_,
                       (const int*)A_row_ptr.dptr_,
                       (const int*)A_col_ind.dptr_,
                       (const float*)B.dptr_, ldb, &beta, dst.dptr_, ldc);
  CHECK_EQ(err, CUSPARSE_STATUS_SUCCESS) << "cusparse: csrmm fail";
  CHECK_EQ(cusparseDestroyMatDescr(descr), CUSPARSE_INDEX_BASE_ZERO)
    << "cusparse: destroy failed!";
  descr = NULL;
}

// Use cuSPARSE
template<bool trans_A, bool trans_B>
inline void csrmm(const Tensor<gpu, 1, double> &A_val,
                  const Tensor<gpu, 1, int> &A_row_ptr,
                  const Tensor<gpu, 1, int> &A_col_ind,
                  const Tensor<gpu, 2, double> &B,
                  double alpha,
                  double beta,
                  Tensor<gpu, 2, double> dst) {
  LOG(FATAL) << "Not implmented!";
}

// Use cuSPARSE
template<bool trans_A, bool trans_B>
inline void csrmm(const Tensor<gpu, 1, half::half_t> &A_val,
                  const Tensor<gpu, 1, int> &A_row_ptr,
                  const Tensor<gpu, 1, int> &A_col_ind,
                  const Tensor<gpu, 2, half::half_t> &B,
                  half::half_t alpha,
                  half::half_t beta,
                  Tensor<gpu, 2, half::half_t> dst) {
  LOG(FATAL) << "Not implmented!";
}
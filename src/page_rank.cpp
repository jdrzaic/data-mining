#include "page_rank.h"
#include "lapack.h"


#include <iostream>


namespace {


void scale_matrix(CsrMatrix<real, DataHost> *A)
{
    for (int r = 0; r < A->n_rows; ++r) {
        int start = A->row_ptr[r];
        int end = A->row_ptr[r+1];
        real val = 1.0 / (end - start);
        for (int i = start; i < end; ++i) {
            A->val[i] = val;
        }
    }
}


template <typename MultOperator>
int get_krylov_subspace(cublasHandle_t handle,
        const MultOperator &mult, const Array<real, DataDev> &v,
        int k, Array<real, DataDev> *Q, Array<real, DataHost> *H,
        Array<real, DataDev> *z)
{
    int n = v.size;
    int ldh = k + 1;
    for (int i = 0; i < H->size; ++i) {
        (*H)[i] = 0.0;
    }
    real alpha;
    CHECK(cublasDnrm2(handle, n, v.data, 1, &alpha));
    alpha = 1.0 / alpha;
    CHECK(cublasDcopy(handle, n, v.data, 1, Q->data, 1));
    CHECK(cublasDscal(handle, n, &alpha, Q->data, 1));
    for (int j = 0; j < k; ++j) {
        mult(Q->data + j*n, z->data);
        for (int i = 0; i <= j; ++i) {
            CHECK(cublasDdot(handle, n, Q->data + i*n, 1, z->data, 1,
                             &(*H)[i+j*ldh]));
            real scal = -(*H)[i+j*ldh];
            CHECK(cublasDaxpy(handle, n, &scal, Q->data + i*n, 1, z->data, 1));
        }
        CHECK(cublasDnrm2(handle, n, z->data, 1, &(*H)[j+1 + j*ldh]));
        if ((*H)[j+1 + j*ldh] == 0.0) {
            return j;
        }
        real scal = 1.0 / (*H)[j+1 + j*ldh];
        CHECK(cublasDcopy(handle, n, z->data, 1, Q->data + (j+1)*n, 1));
        CHECK(cublasDscal(handle, n, &scal, Q->data + (j+1)*n, 1));
    }
    return k;
}


struct pagerank_spmv {
    pagerank_spmv(context ctx_, const CsrMatrix<real, DataHost> &hA,
                  real alpha_)
        : ctx(ctx_), A(hA), alpha(alpha_) 
    {
        Array<real, DataHost> hd;
        hd.init(hA.n_rows);
        Array<real, DataHost> he;
        he.init(hA.n_rows);
        for (int i = 0; i < hA.n_rows; ++i) {
            hd[i] = (hA.row_ptr[i] == hA.row_ptr[i+1]) ? 1.0 : 0.0;
            he[i] = 1.0;
        }
        d = hd;
        e = he;
    }

    void operator ()(real *x, real *y) const
    {
        real zero = 0.0, one = 1.0;
        cusparseMatDescr_t dsc;
        CHECK(cusparseCreateMatDescr(&dsc));
        CHECK(cusparseDcsrmv(
                    ctx.cusparse, CUSPARSE_OPERATION_TRANSPOSE, A.n_rows,
                    A.n_cols, A.nnz(), &one, dsc, A.val.data, A.row_ptr.data,
                    A.col_idx.data, x, &zero, y));
        real xavg = 0.0;
        CHECK(cublasDdot(ctx.cublas, A.n_rows, e.data, 1, x, 1, &xavg));
        xavg /= A.n_rows;
        CHECK(cublasDaxpy(ctx.cublas, A.n_rows, &xavg, d.data, 1, y, 1));
        real scal = 1.0 - alpha;
        CHECK(cublasDscal(ctx.cublas, A.n_rows, &scal, y, 1));
        scal = xavg * alpha;
        CHECK(cublasDaxpy(ctx.cublas, A.n_rows, &scal, e.data, 1, y, 1));
    }

    context ctx;
    const CsrMatrix<real, DataDev> A;
    Array<real, DataDev> d;
    Array<real, DataDev> e;
    real alpha;
};


}  // namespace


void get_pagerank_vector_power(
        context ctx, CsrMatrix<real, DataHost> *A, real alpha,
        Array<real, DataHost> *v, real tol, int max_iter)
{
    scale_matrix(A);
    std::cout << "A = [\n" << *A << "];" << std::endl;
    int n = v->size;
    Array<real, DataDev> dv = *v;
    Array<real, DataDev> y;
    y.init(n);
    pagerank_spmv spmv(ctx, *A, alpha);
    real nrm = 0.0;
    CHECK(cublasDnrm2(ctx.cublas, n, dv.data, 1, &nrm));
    real scal = 1.0 / nrm;
    CHECK(cublasDscal(ctx.cublas, n, &scal, dv.data, 1));
    bool done = false;
    for (int i = 0; !done && i < max_iter; ++i) {
        spmv(dv.data, y.data);
        real alpha = -1.0;
        CHECK(cublasDaxpy(ctx.cublas, n, &alpha, y.data, 1, dv.data, 1));
        CHECK(cublasDnrm2(ctx.cublas, n, dv.data, 1, &nrm));
        done = nrm < tol;
        std::cout << "iter = " << i+1 << "; err = " << nrm << std::endl;
        CHECK(cublasDnrm2(ctx.cublas, n, y.data, 1, &nrm));
        CHECK(cublasDcopy(ctx.cublas, n, y.data, 1, dv.data, 1));
        scal = 1.0 / nrm;
        CHECK(cublasDscal(ctx.cublas, n, &scal, dv.data, 1));
        std::cout << "v = [\n" << dv << "];" << std::endl;
    }
    *v = dv;
}


void get_pagerank_vector_arnoldi(
        context ctx,  CsrMatrix<real, DataHost> *A, real alpha, int k,
        Array<real, DataHost> *v, real tol, int max_iter)
{
    scale_matrix(A);
    std::cout << "A = [\n" << *A << "];" << std::endl;
    Array<real, DataDev> dv = *v;
    real zero = 0.0, one = 1.0;
    int ldh = k+1;
    Array<real, DataDev> work;
    work.init(v->size);
    Array<real, DataDev> Q;
    Q.init(v->size * (k+1));
    Array<real, DataHost> H;
    H.init(k * ldh);
    Array<real, DataHost> S;
    S.init(k);
    Array<real, DataHost> hwork;
    real optwork = 1.0;
    char jobu = 'N', jobvt = 'O';
    int m = k+1, n = k;
    int calc_work = -1;
    int info;
    dgesvd_(&jobu, &jobvt, &m, &n, H.data, &ldh, S.data, nullptr, &ldh,
            nullptr, &ldh, &optwork, &calc_work, &info);
    CHECK(info);
    hwork.init((int)optwork);
    Array<real, DataDev> x;
    x.init(k);
    pagerank_spmv spmv(ctx, *A, alpha);
    for (int i = 0; i < max_iter; ++i) {
        int tk = get_krylov_subspace(ctx.cublas, spmv, dv, k, &Q, &H, &work);
        for (int i = 0; i < tk; ++i) {
            H[i + i*ldh] -= 1.0;
        }
        m = tk+1;
        n = tk;
        dgesvd_(&jobu, &jobvt, &m, &n, H.data, &ldh, S.data, nullptr, &ldh,
                nullptr, &ldh, hwork.data, &hwork.size, &info);
        CHECK(info);
        CHECK(cublasSetVector(tk, sizeof(real), H.data+tk-1, ldh, x.data, 1));
        CHECK(cublasDgemv(ctx.cublas, CUBLAS_OP_N, dv.size, tk, &one, Q.data,
                          dv.size, x.data, 1, &zero, dv.data, 1));
        std::cout << "Q  = [\n" << Q << "]; " << std::endl;
        std::cout << "S  = [\n" << S << "]; " << std::endl;
        std::cout << "VT = [\n" << H << "]; " << std::endl;
        std::cout << "x  = [\n" << x << "]; " << std::endl;
        std::cout << "v  = [\n" << dv << "]; " << std::endl;
        std::cout << "iter = " << i+1 << "; err = " << S[tk-1] << std::endl;
        if (S[tk-1] < tol) break;
    }
    *v = dv;
}


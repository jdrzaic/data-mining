#ifndef PAGE_RANK_H_
#define PAGE_RANK_H_


#include "constants.h"
#include "error.h"
#include "matrix.h"


#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>


struct context {
    void create()
    {
        CHECK(cublasCreate(&cublas));
        CHECK(cusparseCreate(&cusparse));
    }

    void destroy()
    {
        CHECK(cublasDestroy(cublas));
        CHECK(cusparseDestroy(cusparse));
    }

    cublasHandle_t cublas;
    cusparseHandle_t cusparse;
};


void get_pagerank_vector_arnoldi(
        context ctx, CsrMatrix<real, DataHost> *A, real alpha, int k,
        Array<real, DataHost> *v, real tol, int max_iter);


void get_pagerank_vector_power(
        context ctx, CsrMatrix<real, DataHost> *A, real alpha,
        Array<real, DataHost> *v, real tol, int max_iter);

#endif  // PAGE_RANK_H_


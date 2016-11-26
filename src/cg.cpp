#include <iostream>
#include <fstream>


#include "matrix.h"


using namespace std;


typedef double eltype;


void copy(int n, eltype *x, eltype *y)
{
    for (int i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}


void waxpby(int n, eltype alpha, eltype *x, eltype beta, eltype *y, eltype *w)
{
    for (int i = 0; i < n; ++i) {
        w[i] = alpha * x[i] + beta * y[i];
    }
}


eltype dot(int n, eltype *x, eltype *y)
{
    eltype res(0);
    for (int i = 0; i < n; ++i) {
        res += x[i] * y[i];
    }
    return res;
}


void spmv(const CsrMatrix<double, DataHost> &A, eltype *x, eltype *y)
{
    for (int i = 0; i < A.n_rows; ++i) {
        eltype res(0);
        int rstart = A.row_ptr[i];
        int rend = A.row_ptr[i+1];
        for (int j = rstart; j < rend; ++j) {
            int c = A.col_idx[j];
            res += A.val[j] * x[c];
        }
        y[i] = res;
    }
}


int cg(const CsrMatrix<double, DataHost> &A, eltype *b, eltype *x, eltype tol,
       int max_iter)
{
    int n = A.n_rows;
    eltype *r = new eltype[n];
    eltype *d = new eltype[n];
    eltype *w = new eltype[n];
    spmv(A, x, d);
    waxpby(n, 1, b, -1, d, r);
    copy(n, r, d);
    eltype delta0 = dot(n, r, r);
    eltype delta = delta0;
    int info = 1;
    for (int i = 0; i < max_iter; ++i) {
        spmv(A, d, w);
        eltype alpha = delta / dot(n, d, w);
        waxpby(n, alpha, d, 1, x, x);
        waxpby(n, -alpha, w, 1, r, r);
        eltype new_delta = dot(n, r, r);
        std::cout << new_delta << std::endl;
        if (new_delta / delta0 <= tol) {
            info = 0;
            break;
        }
        eltype beta = new_delta / delta;
        waxpby(n, 1, r, beta, d, d);
        delta = new_delta;
    }
    delete[] r;
    delete[] d;
    delete[] w;
    return info;
}


int main(int argc, char *argv[])
{
    cout.precision(6);
    cout << scientific << showpos;
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " FILENAME" << std::endl;
        return -1;
    }
    ifstream fin(argv[1]);
    CsrMatrix<eltype, DataHost> A;
    read_mtx(fin, &A);
    // example of data copy to device
    CsrMatrix<eltype, DataDev> dA = A;
    // copy back to host
    A = dA;
    eltype *b = new eltype[A.n_rows];
    eltype *x = new eltype[A.n_rows];
    for (int i = 0; i < A.n_rows; ++i) {
        b[i] = eltype(1);
        x[i] = eltype(0);
    }

    cout << cg(A, b, x, 1e-6, 1000) << endl;
    cout << A << endl;
    return 0;
}

